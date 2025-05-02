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

def add_train_args(parser):
    parser.add_argument("--model", default="normal", action="store", choices=["normal", "special"])
    parser.add_argument("--pyramid-level", default=1, action="store", choices=[1, 2, 4])

    return parser

def get_args():
    parser = argparse.ArgumentParser()
    parser = add_train_args(parser)
    return parser.parse_args()

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


    train_dataset = SizeFilterDataset(torchvision.datasets.ImageNet("imagenet", split="val"), min_height=256, min_width=256, num_workers=64)
    val_dataset = SizeFilterDataset(torchvision.datasets.ImageNet("imagenet", split="val"), min_height=256, min_width=256, num_workers=64)
    train_dataset_64 = TransformedDataset(train_dataset, transform=img_transform_64)
    train_dataset_128 = TransformedDataset(train_dataset, transform=img_transform_128)
    train_dataset_192 = TransformedDataset(train_dataset, transform=img_transform_192)
    train_dataset_256 = TransformedDataset(train_dataset, transform=img_transform_256)
    val_dataset_64 = TransformedDataset(val_dataset, transform=img_transform_64)
    val_dataset_128 = TransformedDataset(val_dataset, transform=img_transform_128)
    val_dataset_160 = TransformedDataset(val_dataset, transform=img_transform_160)

    val_dataset_192 = TransformedDataset(val_dataset, transform=img_transform_192)
    val_dataset_256 = TransformedDataset(val_dataset, transform=img_transform_256)

    batch_size=64
    train_dataloader_64=torch.utils.data.DataLoader(train_dataset_64, batch_size=batch_size, shuffle=True, num_workers=32)
    train_dataloader_128=torch.utils.data.DataLoader(train_dataset_128, batch_size=batch_size, shuffle=True, num_workers=32)
    train_dataloader_192=torch.utils.data.DataLoader(train_dataset_192, batch_size=batch_size, shuffle=True, num_workers=32)
    train_dataloader_256=torch.utils.data.DataLoader(train_dataset_256, batch_size=batch_size, shuffle=True, num_workers=32)

    val_dataloader_64=torch.utils.data.DataLoader(val_dataset_64,batch_size=batch_size,shuffle=False,num_workers=32)
    val_dataloader_128=torch.utils.data.DataLoader(val_dataset_128,batch_size=batch_size,shuffle=False,num_workers=32)
    val_dataloader_160=torch.utils.data.DataLoader(val_dataset_160,batch_size=batch_size,shuffle=False,num_workers=32)
    val_dataloader_192=torch.utils.data.DataLoader(val_dataset_192,batch_size=batch_size,shuffle=False,num_workers=32)
    val_dataloader_256=torch.utils.data.DataLoader(val_dataset_256,batch_size=batch_size,shuffle=False,num_workers=32)

    # train_dataloaders=[train_dataloader_64,train_dataloader_128,train_dataloader_192,train_dataloader_256]
    train_dataloaders = [train_dataloader_256]
    val_dataloaders=[val_dataloader_64,val_dataloader_128,val_dataset_160,val_dataloader_192,val_dataloader_256]
    resolutions = [64, 128, 160, 192, 256]

    pyramid_level = kwargs["pyramid_level"]
    if kwargs["model"] == "normal":
        model = RelativeVisionTransformer(n_channels=3, nout=1000, img_size=256, patch_size=4, dim=256, attn_dim=256, mlp_dim=256, num_heads=4, num_layers=8).to(device)
        val_dataloaders = val_dataloaders[:-1]
    else:
        model = RelativeVisionTransformerSPP(n_channels=3, nout=1000, dim=256, attn_dim=256, mlp_dim=256, num_heads=4, num_layers=8, pyramid_levels=pyramid_level).to(device)
    # model = SinusoidalVisionTransformerCNN(n_channels=3, nout=1000, patch_size=4, dim=128, attn_dim=64, mlp_dim=256, num_heads=4, num_layers=8).cuda()
    criterion = nn.CrossEntropyLoss()
    NUM_EPOCHS = 10
    if kwargs["model"] == "normal":
        model_name = "model/RelativeVisionTransformer.pt"
    else:
        model_name = f"model/RelativeVisionTransformerSPP{pyramid_level}.pt"
    optimizer = optim.AdamW(model.parameters(), lr=0.0003)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    def evaluate_model(model, criterion, val_loader):
        is_train = model.training
        model.eval()
        with torch.no_grad():
            loss_meter, acc_meter = AverageMeter(), AverageMeter()
            for img, labels in  tqdm.tqdm(val_loader,total=len(val_loader)):
                img = img.to(device)
                labels = labels.to(device)
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
        train_loader_chain=chain(*train_dataloaders)
        # val_loader_chain=chain(*val_dataloaders)
        train_len = sum([len(loader) for loader in train_dataloaders])
        for img, labels in tqdm.tqdm(train_loader_chain,total=train_len):
            img, labels = img.to(device), labels.to(device)

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
        for res, val_loader in zip(resolutions, val_dataloaders):
            val_loss, val_acc = evaluate_model(model, criterion, val_loader)
            valid_acc.append((val_loss,val_acc))
            print(f"Val Epoch: {epoch}, Resolution: {res}, Loss: {val_loss}, Acc: {val_acc}")
        torch.save(model.state_dict(),model_name)
    print('Finished Training')

    torch.save(model.state_dict(),model_name)
    print(training_acc)
    print(valid_acc)