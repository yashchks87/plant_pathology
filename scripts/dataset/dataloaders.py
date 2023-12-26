import torch
import torchvision
import os
import sys
sys.path.append('../')
from dataset.dataset import read_csv
from torch.utils.data import Dataset, DataLoader

class PlantDataset(Dataset):
    def __init__(self, df, img_paths):
        self.df = df
        self.paths = df['paths'].values.tolist()
        self.labels = df['labels'].values.tolist()
        self.img_paths = img_paths
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img = torchvision.io.read_file(self.paths[idx])
        img = torchvision.io.decode_jpeg(img)
        img = torchvision.transforms.functional.resize(img, (self.img_paths[0], self.img_paths[1])) # (C, H, W)
        label = torch.Tensor(self.labels[idx])
        img = img / 255
        return img, label
    

def direct_dataloader():
    pass

def all_loaders():
    pass