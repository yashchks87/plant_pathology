import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append('../')
from tqdm import tqdm
from train.loss import loss_fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, val_loader, epochs=10):
    model = model.to(device)
    data_loaders = {
        'train': train_loader,
        'val': val_loader
    }
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        train_loss, val_loss = 0.0, 0.0
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            with tqdm(data_loaders[phase], unit="batch") as tepoch:
                for img, label in tepoch:
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        img = img.to(device)
                        label = label.to(device)
                        outputs = model(img)
                        loss = loss_fn(outputs, label)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        running_loss += loss.item()
                    tepoch.set_postfix(loss = loss.item())
            if phase == 'train':
                train_loss = running_loss / len(train_loader)
                print(f'train loss: {train_loss}')
            else:
                val_loss = running_loss / len(val_loader)  
                print(f'val loss: {val_loss}')