import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import multiprocessing as mp
import torch, torchvision
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def split_datasets(df, test_size):
    train, test = train_test_split(df, test_size=test_size, random_state=42)
    train, val = train_test_split(train, test_size=test_size, random_state=42)
    return train, val, test

def check_img(img_path):
    try:
        img = torchvision.io.read_file(img_path)
        img = torchvision.io.decode_jpeg(img)
        return True
    except:
        return img_path

def get_class_weight(labels):
    if ~isinstance(labels, np.ndarray):
        labels = np.array(labels)
    return compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)

class ReadCSV():
    def __init__(self, csv_path, img_paths):
        self.csv_path = csv_path
        self.img_paths = img_paths

    def read_csv(self):
        df = pd.read_csv(self.csv_path)
        df['paths'] = df['image'].apply(lambda x: self.img_paths + x)
        self.df = df
    
    def check_paths(self) -> bool or list:
        issue_paths = []
        for path in self.df['paths'].values.tolist():
            if os.path.exists(path) == False:
                issue_paths.append(path)
        self.issue_paths = issue_paths
    
    def encode_labels(self):
        label_encoder = LabelEncoder()
        label_encoder.fit(self.df['labels'].values)
        encoded_labels = label_encoder.transform(self.df['labels'].values)
        self.df['encoded_labels'] = encoded_labels
        self.label_encoder = label_encoder
    
    def one_hot_encoder(self):
        ohe = OneHotEncoder()
        ohe.fit(self.df['encoded_labels'].values.reshape(-1,1))
        one_hot_encoded = ohe.transform(self.df['encoded_labels'].values.reshape(-1,1)).toarray()
        self.df['one_hot_encoded'] = one_hot_encoded.tolist()
        self.ohe = ohe
    
    def split_datasets(self, test_size = 0.01):   
        self.train, self.val, self.test = split_datasets(self.df, test_size=test_size)
    
    def check_image_valids(self, cpu_count = 4):
        img_pahts = self.df['paths'].values.tolist()
        with mp.Pool(cpu_count) as p:
            valids = list(p.map(check_img, img_pahts))
        self.valids = valids

    def get_object(self):
        return self