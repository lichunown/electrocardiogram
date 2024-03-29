# -*- coding: utf-8 -*-

import torch, sys, os
import numpy as np

from torch.utils.data import DataLoader, RandomSampler

from utils import test_set


model_path = 'save_model/model.h5'
write_to_path = 'save_model/subA.txt'
batch_size = 128    
n_cpu = 1



if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('using cuda backend.')
    else:
        device = torch.device("cpu")
        
    with open(model_path, 'rb') as f:
        model = torch.load(f).to(device)
        print(f'loaded model: {model_path}.')
    
    label_info = test_set.label_info
    label_info = dict([(v,k) for k,v in label_info.items()])
    
    
    save_result = []
    with torch.no_grad():
        model.eval()
        dataloader = DataLoader(test_set, batch_size=batch_size, num_workers=n_cpu, shuffle=False)
        for nums, batchs in enumerate(dataloader):
            sig, other, label = batchs['sig'].to(device), batchs['other'].to(device), batchs['label'].to(device)
            out = model(sig, other).cpu().data.numpy()
            for index, result in zip(batchs['index'], out):
                info = test_set.index_info(index)
                
                name = info['name']
                age = info['age']
                m_or_f = info['m_or_f']
                
                write_d = [name, age, m_or_f]
                select_id = np.where(result > 0.5)[0]
                if len(select_id) == 0:
                    select_id = [np.argmax(result)]
                    
                write_d.extend([label_info[i] for i in select_id])
                save_result.append('\t'.join([str(item) for item in write_d]).replace('None', ''))

            sys.stdout.write(f'\r writing {nums}/{len(dataloader)}')
    
    result_sorted = sorted(save_result, key=lambda x:int(x.split('.')[0]))
    
    f = open(write_to_path, 'w', encoding='utf8')
    for line in result_sorted:
        f.write(line)
        f.write('\n')
    f.close()