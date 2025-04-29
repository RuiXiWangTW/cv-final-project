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

class SizeFilterDataset(Dataset):
    def __init__(self, dataset, min_width=256, min_height=256, num_workers=8):
        self.dataset = dataset
        self.min_width = min_width
        self.min_height = min_height
        self.num_workers = num_workers
        self.valid_indices = self._find_valid_indices()

    def _check_size(self, idx):
        path, _ = self.dataset.samples[idx]
        try:
            with Image.open(path) as img:
                w, h = img.size
                if w >= self.min_width and h >= self.min_height:
                    return idx
        except Exception: pass
        return None

    def _find_valid_indices(self):
        indices = list(range(len(self.dataset)))
        valid = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for result in executor.map(self._check_size, indices):
                if result is not None:
                    valid.append(result)

        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        return self.dataset[real_idx]

class TransformedDataset(Dataset):
    def __init__(self, base_ds, transform): 
        self.ds, self.tf = base_ds, transform

    def __len__(self): 
        return len(self.ds)

    def __getitem__(self, i):
        img, lbl = self.ds[i]
        return self.tf(img), lbl

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

class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, nin: int, nout: int):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.num_patches = (img_size // patch_size)**2
        self.hidden = 3
        self.conv = nn.Conv2d(nin, nout, patch_size, stride=patch_size)
    def forward(self, x: torch.Tensor):
        out = self.conv(x).flatten(start_dim=2).transpose(1,2)
        return out
class RelativeVisionTransformer(nn.Module):
    def __init__(self, n_channels: int, nout: int, img_size: int, patch_size: int, dim: int, attn_dim: int,
                 mlp_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, nin=n_channels, nout=dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # learned class embedding
        self.transformer = Transformer(
            dim=dim, attn_dim=attn_dim, mlp_dim=mlp_dim, num_heads=num_heads, num_layers=num_layers, use_relative=True,max_len=1600)

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, nout)
        )
    def forward(self, img: torch.Tensor, return_attn=False) ->Tuple[torch.Tensor, Optional[torch.Tensor]]:
        embs = self.patch_embed(img) # patch embedding
        B, T, _ = embs.shape
        cls_token = self.cls_token.expand(len(embs), -1, -1)
        x = torch.cat([cls_token, embs], dim=1)
        x, alphas = self.transformer(x, attn_mask=None, return_attn=return_attn)
        out = self.head(x)[:, 0]
        return out, alphas
