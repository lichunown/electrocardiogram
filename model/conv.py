# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class ResConv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResConv1, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, 9, padding=4)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 9, padding=4)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = None
        if out_channels != in_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
            
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity) 
        out += identity
        out = self.relu(out)
        
        return out
    
    
class SigConvs(nn.Module):
    def __init__(self, encode_dim=1024):
        super(SigConvs, self).__init__()
        
        self.conv1 = nn.Conv1d(8, 64, 81)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        res_layers = []
        res_layers.append(ResConv1(64, 128))
        for _ in range(1):
            res_layers.append(ResConv1(128, 128))
        res_layers.append(ResConv1(128, 256))
        for _ in range(1):
            res_layers.append(ResConv1(256, 256))
        res_layers.append(ResConv1(256, 512))
        for _ in range(1):
            res_layers.append(ResConv1(512, 512))
        self.res_layers = nn.Sequential(*res_layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(512, encode_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.res_layers(x)

            
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    


class ConvModel(nn.Module):
    def __init__(self, sig_encode_dim=1024, other_encode_dim=2, div_dim=55):
        super(ConvModel, self).__init__()
        
        self.sig_conv = SigConvs(sig_encode_dim)
        self.pre_layer = nn.Linear(2, other_encode_dim)
        self.relu = nn.ReLU(True)
        self.dence_layers = nn.Sequential(
                    nn.Linear(sig_encode_dim + other_encode_dim, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, div_dim),
                    nn.Sigmoid(),
                )
        
    def forward(self, sig, other):
        sig = self.relu(self.sig_conv(sig))
        other = self.relu(self.pre_layer(other))
        return self.dence_layers(torch.cat([sig, other], 1))
        

