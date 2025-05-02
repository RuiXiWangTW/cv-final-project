from torch import nn
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Union, Optional, List
import torchvision.transforms as transforms
import numpy as np
from fastai.vision.all import *
from torch.utils.data import Dataset, DataLoader
from itertools import chain
import tqdm
import pandas as pd
import torchvision
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed


class AverageMeter():
    def __init__(self):
        self.num = 0
        self.tot = 0
    def update(self, val: float, sz: float):
        self.num += val*sz
        self.tot += sz
    def calculate(self) -> float:
        return self.num/self.tot

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
        except Exception:
            pass  # Skip corrupted or unreadable images
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
    def __init__(self, dataset, transform):
        self.dataset=dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, lable = self.dataset[idx]
        transformed_img = self.transform(img)
        return transformed_img, lable

    
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
class PatchEmbedCNN(nn.Module):
    def __init__(self,patch_size: int, nin: int, nout: int):
        super().__init__()
        self.num_patches = (64//patch_size)**2
        self.downsampling=nn.Sequential(
            nn.Conv2d(nin,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((64,64))
        )
        self.out_conv = nn.Conv2d(64,nout, patch_size, stride=patch_size)
    def forward(self, x: torch.Tensor):
        out = self.out_conv(self.downsampling(x)).flatten(start_dim=2).transpose(1,2)
        return out
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, num_embeddings, dim):
        super().__init__()
        assert(dim % 2 == 0)
        self.dim = dim
        self.embeddings = torch.zeros(num_embeddings, dim)
        even_indices = torch.arange(0, dim, 2)
        log_term = torch.log(torch.tensor(10000.0)) / dim
        div_term = torch.exp(even_indices * -log_term)
        embed_indices = torch.arange(num_embeddings).unsqueeze(1)
        self.embeddings[:, 0::2] = torch.sin(embed_indices * div_term)
        self.embeddings[:, 1::2] = torch.cos(embed_indices * div_term)

    def forward(self, pos_ids):
        return self.embeddings.to(pos_ids.device)[pos_ids]
class SinusoidalVisionTransformer(nn.Module):
    def __init__(self, n_channels: int, nout: int, img_size: int, patch_size: int, dim: int, attn_dim: int,
                 mlp_dim: int, num_heads: int, num_layers: int, max_len: int = -1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, nin=n_channels, nout=dim)
        if max_len < 0:
          self.pos_E = SinusoidalPositionEmbedding((img_size//patch_size)**2, dim) # positional embedding matrix
        else:
          self.pos_E = SinusoidalPositionEmbedding((img_size//patch_size)**2, dim) # positional embedding matrix
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # learned class embedding
        self.transformer = Transformer(
            dim=dim, attn_dim=attn_dim, mlp_dim=mlp_dim, num_heads=num_heads, num_layers=num_layers,max_len=576)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, nout)
        )
    def forward(self, img: torch.Tensor, return_attn=False) ->Tuple[torch.Tensor, Optional[torch.Tensor]]:
        embs = self.patch_embed(img) # patch embedding
        B, T, _ = embs.shape
        pos_ids = torch.arange(T).expand(B, -1).to(embs.device)
        embs += self.pos_E(pos_ids) # positional embedding
        cls_token = self.cls_token.expand(len(embs), -1, -1)
        x = torch.cat([cls_token, embs], dim=1)
        x, alphas = self.transformer(x, attn_mask=None, return_attn=return_attn)
        out = self.head(x)[:, 0]
        return out, alphas
class SinusoidalVisionTransformerCNN(nn.Module):
    def __init__(self, n_channels: int, nout: int, patch_size: int, dim: int, attn_dim: int,
                 mlp_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.patch_embed = PatchEmbedCNN(patch_size=patch_size, nin=n_channels, nout=dim)
        self.pos_E = SinusoidalPositionEmbedding(256, dim) # positional embedding matrix
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # learned class embedding
        self.transformer = Transformer(
            dim=dim, attn_dim=attn_dim, mlp_dim=mlp_dim, num_heads=num_heads, num_layers=num_layers,max_len=256)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, nout)
        )
    def forward(self, img: torch.Tensor, return_attn=False) ->Tuple[torch.Tensor, Optional[torch.Tensor]]:
        embs = self.patch_embed(img) # patch embedding
        B, T, _ = embs.shape
        pos_ids = torch.arange(T).expand(B, -1).to(embs.device)
        embs += self.pos_E(pos_ids) # positional embedding
        cls_token = self.cls_token.expand(len(embs), -1, -1)
        x = torch.cat([cls_token, embs], dim=1)
        x, alphas = self.transformer(x, attn_mask=None, return_attn=return_attn)
        out = self.head(x)[:, 0]
        return out, alphas
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
        # img          the input image. shape: (B, nin, img_size, img_size)
        # return_attn  whether to return the attention alphas
        #
        # Outputs
        # out          the output of the vision transformer. shape: (B, nout)
        # alphas       the attention weights for all heads and layers. None if return_attn is False, otherwise
        #              shape: (B, num_layers, num_heads, num_patches + 1, num_patches + 1)

        # generate embeddings
        embs = self.patch_embed(img) # patch embedding
        B, T, _ = embs.shape

        cls_token = self.cls_token.expand(len(embs), -1, -1)
        x = torch.cat([cls_token, embs], dim=1)

        x, alphas = self.transformer(x, attn_mask=None, return_attn=return_attn)
        out = self.head(x)[:, 0]
        return out, alphas
class RelativeVisionTransformerCNN(nn.Module):
    def __init__(self, n_channels: int, nout: int, patch_size: int, dim: int, attn_dim: int,
                 mlp_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.patch_embed = PatchEmbedCNN(patch_size=patch_size, nin=n_channels, nout=dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # learned class embedding
        self.transformer = Transformer(
            dim=dim, attn_dim=attn_dim, mlp_dim=mlp_dim, num_heads=num_heads, num_layers=num_layers, use_relative=True, max_len=256)

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, nout)
        )


    def forward(self, img: torch.Tensor, return_attn=False) ->Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # img          the input image. shape: (B, nin, img_size, img_size)
        # return_attn  whether to return the attention alphas
        #
        # Outputs
        # out          the output of the vision transformer. shape: (B, nout)
        # alphas       the attention weights for all heads and layers. None if return_attn is False, otherwise
        #              shape: (B, num_layers, num_heads, num_patches + 1, num_patches + 1)

        # generate embeddings
        embs = self.patch_embed(img) # patch embedding
        B, T, _ = embs.shape

        cls_token = self.cls_token.expand(len(embs), -1, -1)
        x = torch.cat([cls_token, embs], dim=1)

        x, alphas = self.transformer(x, attn_mask=None, return_attn=return_attn)
        out = self.head(x)[:, 0]
        return out, alphas
if __name__=="__main__":
    device = torch.device("cuda:1")
    img_transform_96=transforms.Compose([
        transforms.RandomCrop((256,256)),
        transforms.Resize((96,96)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    img_transform_128=transforms.Compose([
        transforms.RandomCrop((256,256)),
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    img_transform_160=transforms.Compose([
        transforms.RandomCrop((256,256)),
        transforms.Resize((160,160)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    img_transform_256=transforms.Compose([
        transforms.RandomCrop((256,256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    # train_dataset_96=ImagenetteDataset(train_data,transform=img_transform_96)
    # val_dataset_96=ImagenetteDataset(val_data,transform=img_transform_96)
    # train_dataset_128=ImagenetteDataset(train_data,transform=img_transform_128)
    # val_dataset_128=ImagenetteDataset(val_data,transform=img_transform_128)
    # train_dataset_160=ImagenetteDataset(train_data,transform=img_transform_160)
    # val_dataset_160=ImagenetteDataset(val_data,transform=img_transform_160)
    # val_dataset_256=ImagenetteDataset(val_data,transform=img_transform_256)

    train_dataset = SizeFilterDataset(torchvision.datasets.ImageNet("imagenet", split="train"), min_height=256, min_width=256, num_workers=64)
    val_dataset = SizeFilterDataset(torchvision.datasets.ImageNet("imagenet", split="val"), min_height=256, min_width=256, num_workers=64)

    train_dataset_96 = TransformedDataset(train_dataset, transform=img_transform_96)
    val_dataset_96 = TransformedDataset(val_dataset, transform=img_transform_96)
    train_dataset_128 = TransformedDataset(train_dataset, transform=img_transform_128)
    val_dataset_128 = TransformedDataset(val_dataset, transform=img_transform_128)
    train_dataset_160 = TransformedDataset(train_dataset, transform=img_transform_160)
    val_dataset_160 = TransformedDataset(val_dataset, transform=img_transform_160)
    val_dataset_256 = TransformedDataset(val_dataset, transform=img_transform_256)

    # val_dataset_96 = SizeFilterDataset(torchvision.datasets.ImageNet("imagenet", split="val"), transform=img_transform_96, num_workers=64)
    # train_dataset_128 = SizeFilterDataset(torchvision.datasets.ImageNet("imagenet", split="train"), transform=img_transform_128, num_workers=64)
    # val_dataset_128 = SizeFilterDataset(torchvision.datasets.ImageNet("imagenet", split="val"), transform=img_transform_128, num_workers=64)
    # train_dataset_160 = SizeFilterDataset(torchvision.datasets.ImageNet("imagenet", split="train"), transform=img_transform_160, num_workers=64)
    # val_dataset_160 = SizeFilterDataset(torchvision.datasets.ImageNet("imagenet", split="val"), transform=img_transform_160, num_workers=64)
    # val_dataset_256 = SizeFilterDataset(torchvision.datasets.ImageNet("imagenet", split="val"), transform=img_transform_256, num_workers=64)




    batch_size=176
    val_dataloader_256=torch.utils.data.DataLoader(val_dataset_256,batch_size=batch_size,shuffle=False,num_workers=32)
    train_dataloader_96=torch.utils.data.DataLoader(train_dataset_96, batch_size=batch_size, shuffle=True, num_workers=32)
    val_dataloader_96=torch.utils.data.DataLoader(val_dataset_96, batch_size=batch_size, shuffle=False, num_workers=32)
    train_dataloader_128=torch.utils.data.DataLoader(train_dataset_128, batch_size=batch_size, shuffle=True, num_workers=32)
    val_dataloader_128=torch.utils.data.DataLoader(val_dataset_128, batch_size=batch_size, shuffle=False, num_workers=32)
    train_dataloader_160=torch.utils.data.DataLoader(train_dataset_160, batch_size=batch_size, shuffle=True, num_workers=32)
    val_dataloader_160=torch.utils.data.DataLoader(val_dataset_160, batch_size=batch_size, shuffle=False, num_workers=32)
    train_dataloaders=[train_dataloader_96,train_dataloader_128,train_dataloader_160]
    val_dataloaders=[val_dataloader_96,val_dataloader_128,val_dataloader_160]
    # model = SinusoidalVisionTransformer(n_channels=3, nout=10, img_size=160, patch_size=4, dim=128, attn_dim=64, mlp_dim=256, num_heads=4, num_layers=8).cuda()
    model = SinusoidalVisionTransformerCNN(n_channels=3, nout=1000, patch_size=4, dim=128, attn_dim=64, mlp_dim=256, num_heads=4, num_layers=8).cuda()
    #model = RelativeVisionTransformerCNN(n_channels=3,nout=10,patch_size=4,dim=128,attn_dim=64,mlp_dim=128,num_heads=3,num_layers=6).cuda()
    criterion = nn.CrossEntropyLoss()
    NUM_EPOCHS = 100
    optimizer = optim.AdamW(model.parameters(), lr=0.0003)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    def evaluate_model_256(model, criterion):
        is_train = model.training
        model.eval()
        with torch.no_grad():
            loss_meter, acc_meter = AverageMeter(), AverageMeter()
            for img, labels in  tqdm.tqdm(val_dataloader_256):
                img = img.cuda()
                labels = labels.cuda()
                outputs, _ = model(img)
                loss_meter.update(criterion(outputs, labels).item(), len(img))
                acc = (outputs.argmax(-1) == labels).float().mean().item()
                acc_meter.update(acc, len(img))
        model.train(is_train)
        return loss_meter.calculate(), acc_meter.calculate()
    def evaluate_model(model, criterion):
        is_train = model.training
        model.eval()
        with torch.no_grad():
            loss_meter, acc_meter = AverageMeter(), AverageMeter()
            for img, labels in  tqdm.tqdm(val_loader_chain,total=val_len):
                img = img.cuda()
                labels = labels.cuda()
                outputs, _ = model(img)
                loss_meter.update(criterion(outputs, labels).item(), len(img))
                acc = (outputs.argmax(-1) == labels).float().mean().item()
                acc_meter.update(acc, len(img))
        model.train(is_train)
        return loss_meter.calculate(), acc_meter.calculate()
    training_acc=[]
    valid_acc=[]
    for epoch in range(NUM_EPOCHS):  #
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        train_loader_chain=chain(train_dataloader_96,train_dataloader_128,train_dataloader_160)
        val_loader_chain=chain(val_dataloader_96,val_dataloader_128,val_dataloader_160)
        train_len=len(train_dataloader_96)+len(train_dataloader_128)+len(train_dataloader_160)
        val_len=len(val_dataloader_96)+len(val_dataloader_128)+len(val_dataloader_160)
        for img, labels in tqdm.tqdm(train_loader_chain,total=train_len):
            img, labels = img.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs, _ = model(img)
            # print(outputs.shape, labels.shape)
            # raise KeyboardInterrupt
            loss = criterion(outputs, labels)
            loss_meter.update(loss.item(), len(img))
            acc = (outputs.argmax(-1) == labels).float().mean().item()
            acc_meter.update(acc, len(img))
            loss.backward()
            optimizer.step()
        scheduler.step()
        loss_val=loss_meter.calculate()
        acc_val=acc_meter.calculate()
        training_acc.append((loss_val,acc_val))
        print(f"Train Epoch: {epoch}, Loss: {loss_val}, Acc: {acc_val}")
        if epoch % 10 == 0:
            val_loss, val_acc = evaluate_model(model, criterion)
            valid_acc.append((val_loss,val_acc))
            print(f"Val Epoch: {epoch}, Loss: {val_loss}, Acc: {val_acc}")
            torch.save(model.state_dict(),'dl_project_sinusoidal_cnn.pt')
    val_loss, val_acc = evaluate_model_256(model, criterion)
    print(f"Val Epoch: {epoch}, Loss: {val_loss}, Acc: {val_acc}")
    print('Finished Training')
    torch.save(model.state_dict(),'dl_project_sinusoidal_cnn.pt')
    print(training_acc)
    print(valid_acc)