import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append('../')
from tqdm import tqdm
from train.loss import loss_fn
from metrics.metrics import generate_metrics
import wandb
import itertools
from train.save_model import save_model

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
multiple_gpus = False
# if torch.cuda.is_available():
    # if torch.cuda.device_count() > 1:
    #     multiple_gpus = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_loader, epochs=10, wandb_dict=None, load_model = None, save_path='../../weights/res32/'):
    if wandb_dict == None:
        wandb.init(project='plant_pathology')
    else:
        wandb.init(project='plant_pathology', config=wandb_dict)
    data_loaders = {
        'train': train_loader,
        'val': val_loader
    }
    if load_model != None:
        model.load_state_dict(torch.load(load_model)['model_state_dict'])
    if multiple_gpus:
        model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        train_loss, val_loss = 0.0, 0.0
        train_prec, val_prec = torch.zeros(1, 12), torch.zeros(1, 12)
        train_rec, val_rec = torch.zeros(1, 12), torch.zeros(1, 12)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss, running_prec, running_rec = 0.0, torch.zeros(1, 12), torch.zeros(1, 12)
            with tqdm(data_loaders[phase], unit="batch") as tepoch:
                for img, label in tepoch:
                    img = img.to(device)
                    label = label.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(img)
                        loss = loss_fn(outputs, label)
                        c_m, p, r = generate_metrics(outputs, label, 12)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item()
                    running_prec += p
                    running_rec += r
                    tepoch.set_postfix(loss = loss.item())
            if phase == 'train':
                train_prec = train_prec[0]
                train_rec = train_rec[0]
                train_loss = running_loss / len(train_loader)
                train_prec = train_prec / len(train_loader) 
                train_rec = train_rec / len(train_loader)
                print(f'train loss: {train_loss}')
            else:
                val_prec = running_prec[0]
                val_rec = running_rec[0]
                val_loss = running_loss / len(val_loader)  
                val_prec = val_prec / len(val_loader)
                val_rec = val_rec / len(val_loader)
                print(f'val loss: {val_loss}')
        keys = ['train_loss', 'val_loss']
        ids = itertools.chain.from_iterable([[f'train_prec_{x}' for x in range(12)],
        [f'val_prec_{x}' for x in range(12)],
        [f'train_rec_{x}' for x in range(12)],
        [f'val_rec_{x}' for x in range(12)]])
        keys = keys + list(ids)
        # return train_prec, val_prec, train_rec, val_rec
        metric_list = torch.cat([train_prec, val_prec, train_rec, val_rec]).tolist()
        # return metric_list, train_loss, val_loss 
        wandb.log(dict(zip(keys, [train_loss, val_loss] + metric_list)))
        save_model(model, epoch, optimizer, multiple_gpus, save_path)


def eval_dataset(model, dataset, weights_path = None, return_preds = False):
    if weights_path != None:
        model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    model = model.to(device)
    model.eval()
    loss = 0.0 
    preds = []
    for img, label in dataset:
        img = img.to(device)
        label = label.to(device)
        with torch.no_grad():
            outputs = model(img)
            preds.append(outputs.cpu())
            loss += loss_fn(outputs, label)
    new_preds = torch.Tensor([y.numpy() for x in preds for y in x])
    if return_preds:
        return loss / len(dataset), new_preds
    return loss / len(dataset)