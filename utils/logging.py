# -*- coding: utf-8 -*-
import os, sys
import torch
import json


class Log(object):
    def __init__(self, save_model_dir):
        self.save_model_dir = save_model_dir
        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)
            
        if os.path.exists(os.path.join(self.save_model_dir, 'info.json')):
            with open(os.path.join(self.save_model_dir, 'info.json'), 'r') as f:
                self.info = json.loads(''.join(f.readlines()))
        else:
            self.info = {
                'eval_f1': 0,
            }
    
    def __del__(self):
        self.save_info()
        
    def save_info(self):
        with open(os.path.join(self.save_model_dir, 'info.json'), 'w') as f:
            f.writelines(json.dumps(self.info))
            
    def save_model(self, model, eval_f1):
        if eval_f1 > self.info['eval_f1']:
            self._save_model(model)
            self.info['eval_f1'] = eval_f1
            
    def _save_model(self, model):
        with open(os.path.join(self.save_model_dir, 'model.h5'), 'wb') as f:
            torch.save(model, f)
    
    def add_model_info(self, info:dict):
        for k, v in info.items():
            self.info[k] = v
            
    def logging_flush(self, msg):
        sys.stdout.write('\r' + msg)
        
    def logging(self, msg):
        sys.stdout.write(msg + '\n')