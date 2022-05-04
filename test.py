import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import *

import numpy as np,matplotlib.pyplot as plt
import random


transform_test = transforms.Compose([
    transforms.ToTensor()
])

testset = torchvision.datasets.CIFAR10(root='dataset', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model_name):
    
    print('\nBegin Test...Model name:',model_name)
    net = ResNet18()
    net = torch.load('nets/'+model_name)
    
    net = net.to(device)
    
    total,correct = 0,0
    
    ERR = []
       
    for i, data in enumerate(testloader,0):   
             
        if i<=len(testloader)/2:
            continue
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
    
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

        # for i in range(len(labels)):
        #     if labels[i] != predicted[i]:
        #         ERR.append((images[i],labels[i],predicted[i]))
    print('\nTest Accuracy:',correct.item()/total)
        
    return ERR


for model_name in ['none.pth','mixup.pth','cutmix.pth','cutout.pth']:
   ERR = test(model_name)


# print('\nInspecting some images misclassified by Cutout')

# for i in range(3):
#     img,label,p_label = random.choice(ER1)
    
#     img = np.asarray(img.cpu())
    
#     img = img.transpose(1,2,0)
#     plt.figure(dpi=300)
#     plt.imshow(img)
    
#     string = classes[label]+'(true)'
#     string += '  '+classes[p_label]+'(false)'
    
#     plt.xlabel(string)
    



