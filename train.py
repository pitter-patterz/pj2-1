
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import *
from data_augmentation import *
# from tensorboardX import SummaryWriter

import numpy as np,matplotlib.pyplot as plt
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

methods = ['none','cutout','cutmix','mixup']
m = str(sys.argv[-1])

if m not in methods:
    m = 'none'
print('\nThe method being applied:',m,'\n')

# hyper parameters

EPOCH = 10
BATCH_SIZE = 128

# load and preprocess the dataset

transform_train = transforms.Compose([
    transforms.ToTensor()
])

if m == 'cutout':
    transform_train = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR100(root='dataset100', train=True, download=False, transform=transform_train)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR100(root='dataset100', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# A half of the pictures in the testloader is used as valid data.


# train a ResNet18
net = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)

# writer = SummaryWriter('record_'+m+str(LR))
iteration = 0
for epoch in range(EPOCH):
    print('\n*****Epoch:',epoch+1)
    
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    
    LS_train,LS_val = [],[]
    AC_train,AC_val = [],[]
    
    for data in trainloader:
        
        # inputs and labels
        
        length = len(trainloader)
        inputs, labels = data
               
        if m == 'mixup':
            inputs, labels = mixup(inputs,labels)  # do mixup
        
        if m == 'cutmix':
            inputs, labels = cutmix(inputs,labels)  # do cutmix
        
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        
        #forward & backward
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        
        #print accuracy & loss in each batch
        
        labels = data[1].to(device)  
        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        
        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '%(epoch+1,iteration,loss.item(),100.0*correct/total))
        iteration += 1
        
        # writer.add_scalars(m+'_LR='+str(LR)+'/acc',{'train':100.0*correct/total},iteration)
        # writer.add_scalars(m+'_LR='+str(LR)+'/loss',{'train':loss.item()},iteration)
        
    # compute the accuracy on valid dataset after each epoch
    
    print('\n*****Waiting Test...')
    
    with torch.no_grad():
        correct = 0
        total = 0
        loss_valid = 0
        for i, data in enumerate(testloader,0):
            if i>len(testloader)/2:
                break
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss_valid += 2*criterion(outputs,labels)/len(testloader)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        
        print('Test Accuracy:',correct.item()/total)
        print('Test Loss:',loss_valid)
        # writer.add_scalars(m+'_LR='+str(LR)+'/acc',{'valid':100.0*correct/total},iteration)
        # writer.add_scalars(m+'_LR='+str(LR)+'/loss',{'valid':loss_valid},iteration)
        
       
print('Train has finished.')

# writer.close()
# torch.save(net,m+'_'+str(LR)+'.pth')
