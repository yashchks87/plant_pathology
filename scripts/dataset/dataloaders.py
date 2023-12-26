import torch
import torchvision
import os
import sys
sys.path.append('../')
from dataset.dataset import ReadCSV
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

class PlantDataset(Dataset):
    def __init__(self, df, img_paths):
        self.df = df
        self.paths = df['paths'].values.tolist()
        self.labels = df['encoded_labels'].values.tolist()
        self.img_paths = img_paths
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img = torchvision.io.read_file(self.paths[idx])
        img = torchvision.io.decode_jpeg(img)
        img = torchvision.transforms.functional.resize(img, (self.img_paths[0], self.img_paths[1])) # (C, H, W)
        label = torch.Tensor([self.labels[idx]]).long()
        img = img / 255
        return img, label
    

def direct_dataloader():
    pass

class GetLoaders():
    def __init__(self, train, val, test, num_workers = 22, batch_size = 32, img_size = (224, 224)):
        self.train = train
        self.val = val
        self.test = test
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
    
    def create_sets(self):
        self.train_set = PlantDataset(self.train, self.img_size)
        self.val_set = PlantDataset(self.val, self.img_size)
        self.test_set = PlantDataset(self.test, self.img_size)
    
    def create_loaders(self):
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=22, prefetch_factor=2, persistent_workers=True)
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=22, prefetch_factor=2, persistent_workers=True)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=22, prefetch_factor=2, persistent_workers=True)

    def get_object(self):
        return self
    
    def return_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader