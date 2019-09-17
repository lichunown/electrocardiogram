import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils import train_set
from model import ConvModel


epochs = 10
batch_size = 16
n_cpu = 1



if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_cpu)
    
    net = ConvModel(1024, 2).to(device)
    optimizer = torch.optim.Adam(net.parameters())
    
    
    loss_func = F.binary_cross_entropy
    
    print('start trainning ......')
    for e in range(epochs):
        for i, batchs in enumerate(dataloader):
            sig, other, label = batchs['sig'].to(device), batchs['other'].to(device), batchs['label'].to(device)
    
            out = net(sig, other)        
            loss = loss_func(out, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f'[{e}/{epochs}] b:{i}   loss: {loss.cpu().data.numpy()}')