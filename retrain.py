import torch
from torch import nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from model import TimeSiries
from preprocess import Shuffle, StandardScaler, CreateMatrix
device = 'cuda' if torch.cuda.is_available() else 'cpu'

timesteps = 30
future = 9
hidden_size = 128
dropout = 0.2
lr = 1e-4
weight_decay = 1e-4
teacher_force_ratio = 0.5

eemd = [np.load(f'EEMD/EEMD{i:04d}.npy').reshape(11, -1) for i in range(1501)]

for mode in range(int(input('start:')), int(input('end:'))):
    epochs = 300 if mode in [0, 1] else 1000

    x_train, y_train, x_val, y_val, x_test, y_test = CreateMatrix(
            mode, eemd, timesteps, future)
    x_train, y_train = Shuffle(x_train, y_train)
    
    norm = StandardScaler(y_train)
    x_train = norm.transform(x_train)
    y_train = norm.transform(y_train)
    x_val = norm.transform(x_val)
    y_val = norm.transform(y_val)
    x_test = norm.transform(x_test)
    
    x_train = torch.FloatTensor(x_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    x_val = torch.FloatTensor(x_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    x_test = torch.FloatTensor(x_test)
    
    model = torch.load(f'models/mode{mode}.pt')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 
            lr=lr, weight_decay=weight_decay)
    
    scaler = torch.cuda.amp.GradScaler()
    
    model.eval()
    with torch.no_grad():
        y_pred = model(x_val, future, None, 0)
        best_weights = criterion(y_pred, y_val).item()
    history = {'loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        with torch.cuda.amp.autocast():
            y_pred = model(x_train, future, y_train[:, -1-future:], teacher_force_ratio)
            loss = criterion(y_pred, y_train)
    
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
            
        model.eval()
        with torch.no_grad():
            y_pred = model(x_val, future, None, 0)
            val_loss = criterion(y_pred, y_val)
        
        print(f'mode: {mode:02d} | epoch: {epoch+1:04d}/{epochs}', end=' | ')
        print(f'loss: {loss.item():.7f} | val_loss: {val_loss.item():.7f}'
                , end=' | ')
    
        history['loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        
        if val_loss.item() <= best_weights:
            torch.save(model, f'models/mode{mode}.pt')
            best_weights = val_loss.item()
            print('weights saved')
        else:
            print('')
    
    model = torch.load(f'models/mode{mode}.pt')
    model.eval()
    x_test = x_test.to(device)
    with torch.no_grad():
        y_pred = model(x_test, 0, None, 0).cpu().numpy()[:, -1:]
        y_pred = norm.inverse_transform(y_pred)
    
    plt.figure(figsize=(16, 8))
    plt.plot(y_test, label='ACTUAL')
    plt.plot(y_pred, label='PREDICT')
    plt.title(f'MODE: {mode}-MAELOSS: {mean_absolute_error(y_pred, y_test)}')
    plt.legend()
    plt.savefig(f'train/predict_mode{mode:02d}.jpg')
    plt.close()
    
    plt.figure(figsize=(16, 8))
    plt.plot(history['loss'], label='LOSS')
    plt.plot(history['val_loss'], label='VAL_LOSS')
    plt.title(f'MODE: {mode}-HISTORY')
    plt.legend()
    plt.savefig(f'train/history_mode{mode:02d}.jpg')
    plt.close()
            
