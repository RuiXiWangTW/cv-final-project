import os
from itertools import chain
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageNet
from tqdm import tqdm
from PIL import Image
from typing import Tuple, Union, Optional, List

      
class PatchEmbedSPP(nn.Module):
    def __init__(self, nin, dim, pyramid_levels=[1,2,4,8]):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(nin, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.levels = pyramid_levels

    def forward(self, x):
        fmap = self.stem(x)   # (B, dim, H, W)
        B, C, _, _ = fmap.shape
        tokens = []
        for l in self.levels:
            p = F.adaptive_avg_pool2d(fmap, (l, l))      # (B, C, l, l)
            # reshape to (B, l*l, C)
            cells = p.view(B, C, l*l).transpose(1,2)     # (B, l^2, C)
            tokens.append(cells)
        # concat all levels: (B, sum(l^2), C)
        out = torch.cat(tokens, dim=1)
        return out
    def get_embeddings(self, x):
        with torch.no_grad():
            fmap = self.stem(x)   # (B, dim, H, W)
            return torch.mean(fmap, dim=1)[0]



class AttentionHead(nn.Module):
    def __init__(self, dim: int, n_hidden: int, use_relative=False, max_len=-1):
        super().__init__()
        self.W_K = nn.Linear(dim, n_hidden) # W_K weight matrix
        self.W_Q = nn.Linear(dim, n_hidden) # W_Q weight matrix
        self.W_V = nn.Linear(dim, n_hidden) # W_V weight matrix
        self.n_hidden = n_hidden
        self.use_relative = use_relative
        if use_relative:
            self.max_len = max_len
            self.relative_emb = nn.Parameter(torch.randn(max_len, n_hidden))
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        out, alpha = None, None
        B, T, _ = x.shape
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        if self.use_relative:
            E_rel = self.relative_emb[self.max_len - T:, :].transpose(0, 1)
            Q_rel = torch.matmul(Q, E_rel)
            S_rel = self.skew(Q_rel)
            alpha = (torch.matmul(Q, K.transpose(1, 2)) + S_rel) / (self.n_hidden ** 0.5)
        else:
            alpha = torch.matmul(Q, K.transpose(1, 2)) / (self.n_hidden ** 0.5)
        if attn_mask is not None:
            attn_mask = attn_mask.cuda()
            alpha = alpha.masked_fill(attn_mask == 0, float('-inf'))
        alpha = torch.softmax(alpha, dim=-1)
        attn_output = torch.matmul(alpha, V)
        return attn_output, alpha
    def skew(self, Q_rel):
        padded = F.pad(Q_rel, (1,0))
        B, r, c  = padded.shape
        return padded.reshape(B, c, r)[:, 1:, :]
class MultiHeadedAttention(nn.Module):
    def __init__(self, dim: int, n_hidden: int, num_heads: int, use_relative=False, max_len=-1):
        super().__init__()
        self.attention_heads = nn.ModuleList([
            AttentionHead(dim, n_hidden, use_relative=use_relative, max_len=max_len)
            for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * n_hidden, dim)
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_alphas = [], []
        for head in self.attention_heads:
            output, alpha = head(x, attn_mask)
            attn_output.append(output)
            attn_alphas.append(alpha)
        attn_output = self.linear(torch.cat(attn_output, dim=-1))
        attn_alphas = torch.stack(attn_alphas, dim=1)
        return attn_output, attn_alphas
class FFN(nn.Module):
    def __init__(self, dim: int, n_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, dim),
        )
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return self.net(x)
class AttentionResidual(nn.Module):
    def __init__(self, dim: int, attn_dim: int, mlp_dim: int, num_heads: int, use_relative=False, max_len=-1):
        super().__init__()
        self.attn = MultiHeadedAttention(dim, attn_dim, num_heads, use_relative=use_relative, max_len=max_len)
        self.ffn = FFN(dim, mlp_dim)
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, alphas = self.attn(x=x, attn_mask=attn_mask)
        x = attn_out + x
        x = self.ffn(x) + x
        return x, alphas
class Transformer(nn.Module):
    def __init__(self, dim: int, attn_dim: int, mlp_dim: int, num_heads: int, num_layers: int, use_relative=False, max_len=-1):
        super().__init__()
        self.layers = nn.ModuleList([
          AttentionResidual(dim, attn_dim, mlp_dim, num_heads, use_relative=use_relative, max_len=max_len)
          for _ in range(num_layers)
        ])
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, return_attn=False)-> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        output, collected_attns = None, None
        output = x
        if return_attn:
          collected_attns = []
        for layer in self.layers:
          output, alphas = layer(output, attn_mask)
          if return_attn:
            collected_attns.append(alphas)
        if return_attn:
          collected_attns = torch.stack(collected_attns, dim=1)
        return output, collected_attns
class RelativeVisionTransformerSPP(nn.Module):
    def __init__(self, n_channels, nout, dim, attn_dim,
                 mlp_dim, num_heads, num_layers,
                 pyramid_levels=[1,2,4,8]):
        super().__init__()
        self.patch_embed = PatchEmbedSPP(n_channels, dim, pyramid_levels)
        self.cls_token   = nn.Parameter(torch.randn(1,1,dim))
        self.transformer = Transformer(
            dim, attn_dim, mlp_dim,
            num_heads, num_layers,
            use_relative=True,
            max_len=sum(l*l for l in pyramid_levels)+1
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, nout)
        )
    def forward(self, img, return_attn=False):
        B    = img.size(0)
        embs = self.patch_embed(img)
        cls  = self.cls_token.expand(B, -1, -1)
        x    = torch.cat([cls, embs], dim=1)
        x, attn = self.transformer(x, None, return_attn)
        return self.head(x)[:, 0], attn
    
    def get_embeddings(self, img):
        with torch.no_grad():
            img = img.unsqueeze(0)
            return self.patch_embed.get_embeddings(img)

