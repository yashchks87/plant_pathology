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
    # def __init__(self, df, img_size, class_list = None, augment = True):
    def __init__(self, df, img_size, augment = True, cut_out_params = None):
        self.df = df
        self.paths = df['paths'].values.tolist()
        self.labels = df['encoded_labels'].values.tolist()
        self.img_size = img_size
        self.augment = augment
        self.norm = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   
        self.cut_out_params = cut_out_params 
        if augment:
            # Cut out params needed are
            # 1. size of the cutout
            # 2. number of holes    
            if cut_out_params != None:
                self.size_of_cutout = cut_out_params['size_of_cutout']
                self.num_of_cutout_holed = cut_out_params['num_of_cutout_holes']
            self.horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
            self.random_rotation = torchvision.transforms.RandomRotation(degrees=45)
            self.jitter = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    
    def __len__(self):
        return len(self.df)

    def cutout(self, img:torch.Tensor, size:int, n_holes:int) -> torch.Tensor:
        h, w = img.shape[1:]
        for _ in range(n_holes):
            y = torch.randint(low=0, high = h - size + 1, size=(1,))
            x = torch.randint(low=0, high = w - size + 1, size=(1,))
            img[:, y:y+size, x:x+size] = 0
        return img
    
    def __getitem__(self, idx):
        # img = torchvision.io.read_file(self.paths[idx])
        img = torchvision.io.read_image(self.paths[idx])
        label = torch.Tensor([self.labels[idx]]).long()
        # img = torchvision.io.decode_jpeg(img)
        # if self.augment and (self.class_list is not None) and (float(self.labels[idx]) in self.class_list):
        if self.augment:
            img = self.horizontal_flip(img)
            img = self.random_rotation(img)
            img = self.jitter(img)
            if self.cut_out_params != None: 
                img = self.cutout(img, self.size_of_cutout, self.num_of_cutout_holed)
        if self.img_size != 224:
            img = torchvision.transforms.functional.resize(img, (self.img_size[0], self.img_size[1])) # (C, H, W)
        img = img / 255.0
        img = img.float()
        img = self.norm(img)
        return img, label

    
def direct_dataloader():
    pass

class GetLoaders():
    def __init__(self, train, val, test, img_size = (224, 224)):
        self.train = train
        self.val = val
        self.test = test
        self.batch_size = None
        self.img_size = img_size
        self.num_workers = None
    
    def create_sets(self, train_augment = True, val_augment = False, test_augment = False, cut_out_params = None):
        self.train_augment, self.val_augment, self.test_augment = train_augment, val_augment, test_augment
        self.cut_out_params = cut_out_params
        self.train_set = PlantDataset(self.train, self.img_size, train_augment, cut_out_params)
        self.val_set = PlantDataset(self.val, self.img_size, val_augment)
        self.test_set = PlantDataset(self.test, self.img_size, test_augment)
    
    def create_loaders(self, batch_size = 32, num_workers = 24, shuffle_train = True, shuffle_val = False):
        self.batch_size = batch_size
        self.num_workers = num_workers  
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_workers)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_workers)

    def get_object(self):
        return self
    
    def return_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader