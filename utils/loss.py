# -*- coding: utf-8 -*-
import numpy as np



def f1_score(y_, y):
    y_ = (y_ > 0.5).astype('int')
    y = (y > 0.5).astype('int')
    
    TP = np.sum(y_ & y)
    TN = np.sum((1-y_) & (1-y))
    FP = np.sum(y_ & (1-y))
    FN = np.sum((1-y_) & y)
    
    accuracy = (TP+TN)/(TP + TN + FP + FN)
    recall = TP / (TP + FN)
    
    f1 = 2 * accuracy*recall / (accuracy + recall)
    
    return f1, accuracy, recall

