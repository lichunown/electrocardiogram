import torch, sys
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F

from utils import train_set, Log, f1_score
from model import ConvModel


epochs = 10
batch_size = 32
n_cpu = 6

save_dir = 'save_model'

def valid(dataloader, model, device, log):
    with torch.no_grad():
        model.eval()
        f1_all, acc_all, recall_all = [], [], []
        for i, batchs in enumerate(dataloader):
            sig, other, label = batchs['sig'].to(device), batchs['other'].to(device), batchs['label'].to(device)
            out = net(sig, other)
            f1, accuracy, recall = f1_score(out.cpu().data.numpy(), label.cpu().data.numpy())
            f1_all.append(f1)
            acc_all.append(accuracy)
            recall_all.append(recall)
            log.logging_flush(f'      validing      b:{i}/{len(dataloader)}  ')
        f1_all, acc_all, recall_all = np.mean(f1_all), np.mean(acc_all), np.mean(recall_all)
        log.logging(f'f1:{f1_all}  acc:{acc_all}  recall:{recall_all}')
        return f1_all
        
        
if __name__ == '__main__':
    log = Log(save_dir)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.logging('using cuda backend.')
    else:
        device = torch.device("cpu")
    
    all_split = np.arange(len(train_set))
    train_split, val_split = all_split[:int(len(all_split)*0.8)], all_split[int(len(all_split)*0.8):]
    
    dataloader = DataLoader(train_set, batch_size=batch_size, num_workers=n_cpu, sampler=RandomSampler(train_split))
    valid_dataloader = DataLoader(train_set, batch_size=batch_size, num_workers=n_cpu, sampler=RandomSampler(val_split))
    
    
    net = ConvModel(1024, 2).to(device)
    optimizer = torch.optim.Adam(net.parameters())
    
    
    loss_func = F.binary_cross_entropy
    
    print('start trainning ......')
    for e in range(epochs):
        net.train()
        for i, batchs in enumerate(dataloader):
            sig, other, label = batchs['sig'].to(device), batchs['other'].to(device), batchs['label'].to(device)
    
            out = net(sig, other)        
            loss = loss_func(out, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log.logging_flush(f'[{e}/{epochs}] b:{i}/{len(dataloader)}   loss: {loss.cpu().data.numpy()}')
        
        log.logging('')
        f1_sorce = valid(valid_dataloader, net, device, log)
        log.save_model(net, f1_sorce)
    
    log.save_info()
            
            

        
        