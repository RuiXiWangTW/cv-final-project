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
from normal_vit import RelativeVisionTransformer
from spp_vit import RelativeVisionTransformerSPP
import argparse
import logging
from train import SizeFilterDataset, TransformedDataset
import cv2
import matplotlib.pyplot as plt


def add_train_args(parser):
    parser.add_argument("--model", default="normal", action="store", choices=["normal", "special"])
    return parser

def get_args():
    parser = argparse.ArgumentParser()
    parser = add_train_args(parser)
    return parser.parse_args()

if __name__=="__main__":
    args = get_args()
    kwargs = args.__dict__
    device = torch.device("cuda:3") 

    # Using resolutions 64x64, 128x128, 192x192, and 256x256 seem like a good idea for training. 64x64, 128x128, 256x256, 384x384, and 512x512 should work for evaluation.
    img_transform_64=transforms.Compose([
        transforms.RandomCrop((256,256)),
        transforms.Resize((64,64)),
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
    img_transform_192=transforms.Compose([
        transforms.RandomCrop((256,256)),
        transforms.Resize((192,192)),
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

    val_dataset = SizeFilterDataset(torchvision.datasets.ImageNet("imagenet", split="val"), min_height=256, min_width=256, num_workers=64)
    if kwargs["model"] == "normal":
        state_dict = torch.load("model/RelativeVisionTransformer.pt")
        model = RelativeVisionTransformer(n_channels=3, nout=1000, img_size=256, patch_size=4, dim=128, attn_dim=64, mlp_dim=256, num_heads=4, num_layers=8).to(device)
    else:
        state_dict = torch.load("model/RelativeVisionTransformerSPP_124.pt")
        model = RelativeVisionTransformerSPP(n_channels=3, nout=1000, dim=128, attn_dim=64, mlp_dim=256, num_heads=4, num_layers=8).to(device)
    model.load_state_dict(state_dict)
    val_dataset_64 = TransformedDataset(val_dataset, transform=img_transform_64)
    val_dataset_128 = TransformedDataset(val_dataset, transform=img_transform_128)
    val_dataset_160 = TransformedDataset(val_dataset, transform=img_transform_160)
    val_dataset_192 = TransformedDataset(val_dataset, transform=img_transform_192)
    val_dataset_256 = TransformedDataset(val_dataset, transform=img_transform_256)

    datasets = [val_dataset_64, val_dataset_128, val_dataset_160, val_dataset_192, val_dataset_256]
    embeddings = []
    images = []
    for dataset in datasets:
        img, label = dataset[0]
        img = img.to(device)
        img_min, img_max = torch.min(img), torch.max(img)
        normalized_img = ((img-img_min)/(img_max-img_min)).permute(1, 2, 0).cpu().numpy()
        normalized_img = (normalized_img * 255).astype(np.uint8)
        embedding = model.get_embeddings(img)
        min_val, max_val = torch.min(embedding), torch.max(embedding)
        embedding = (embedding - min_val)/(max_val-min_val)
        embedding = embedding.cpu().numpy()
        embedding = (embedding * 255).astype(np.uint8)
        embeddings.append(embedding)
        images.append(normalized_img)
    resolutions = [64, 128, 160, 192, 256]
    model_name = kwargs["model"]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows, 5 columns

    for i in range(5):
        axes[0, i].imshow(images[i])
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Original images - res={resolutions[i]}x{resolutions[i]}")

        axes[1, i].imshow(embeddings[i], cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Embeddings - res={resolutions[i]}x{resolutions[i]}")

    plt.tight_layout()
    plt.savefig("embedings.pdf")
    plt.show()

    for emb, img, res in zip(embeddings, images, resolutions):
        cv2.imwrite(f"{model_name}_{res}.png", emb)
        cv2.imwrite(f"{res}.png", img)


    

