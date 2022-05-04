

"""
Cutmix and mixup. We use torchvision.tranforms.randomcrop() to do cutout.

"""

import torch
import numpy as np

def mixup(inputs,labels,max_lam=0.15):
        
    n = inputs.shape[0]
    indices = torch.randperm(n)
    
    lam = np.random.uniform(0,max_lam)
    new_inputs = (1-lam)*inputs + lam*inputs[indices]
    
    new_labels = torch.zeros(size=(n,10))
    for i in range(n):
        new_labels[i,labels[i]] = 1-lam
        new_labels[i,labels[indices][i]] = lam
       
    return new_inputs,new_labels


def randcut(x=32,y=32,rate=0.15):
    lx,ly = int(x*rate),int(y*rate)
    midx,midy = np.random.randint(x),np.random.randint(y)
    
    x1 = max(int(midx-lx/2),0)
    x2 = min(int(midx+lx/2),x-1)
    y1 = max(int(midy-ly/2),0)
    y2 = min(int(midy+ly/2),y-1)
    
    return x1,x2,y1,y2


def cutmix(inputs,labels,rate=0.15):
    
    n = inputs.shape[0]
    indices = torch.randperm(n)
    
    new_inputs = torch.zeros(size=inputs.shape)
    new_inputs.copy_(inputs)
    
    new_labels = torch.zeros(size=(n,10))
    perm_inputs = inputs[indices] 
    
    for i in range(n):
        x1,x2,y1,y2 = randcut(rate=rate)
        lam = (x2-x1)*(y2-y1)/32**2
        new_labels[i,labels[i]] = 1-lam
        new_labels[i,labels[indices][i]] = lam
        
        new_inputs[i,:,x1:x2+1,y1:y2+1] = perm_inputs[i,:,x1:x2+1,y1:y2+1]
        
    return new_inputs,new_labels
        
        
        
