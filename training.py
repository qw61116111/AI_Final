import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import csv    
from torchvision import transforms
import random
import time

count_month=torch.from_numpy(np.load('C:\\Users\Q56091087\Desktop\data_array.npy'))

class AutoEncoder(nn.Module):
    def __init__(self,num_feature,hidden_size, num_layers):
        super().__init__()
        self.lstm=nn.LSTM(
            input_size=num_feature,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
            )
        self.out = nn.Sequential( 

                   nn.Linear(1024, 2048),
                   nn.ReLU(inplace=True),
                   nn.Linear(2048, 5096),
                   nn.ReLU(inplace=True),
                   nn.Linear(5096, 10192),
                   nn.ReLU(inplace=True),
                   nn.Linear(10192, 22170),
                   )

    def forward(self,inputs):
        out,(h_n,c_n)=self.lstm(inputs, None)
        outputs=self.out(h_n[1])

        return  outputs
#%%
num_pred_month=1
num_train_set=31
class dataset(torch.utils.data.Dataset):
    def __init__(self,is_train=True,shop=25):
        if is_train:
            self.shop=shop
           
            self.pre_data=count_month
        else:
            self.shop=shop
            self.pre_data=count_month
    def __len__(self):
        return num_train_set-12+1
    def __getitem__(self, index):

        self.data=self.pre_data[index:index+12,self.shop]
        self.label=count_month[index+12,self.shop]
        return self.data,self.label
    
def val(shop):
        
    data=count_month[21:33,shop]
    label=count_month[33,shop]
    return data,label

def test(shop):
        
    data=count_month[22:34,shop]
   
    return data
#%%
net=AutoEncoder(22170,1024,2)
#net.cuda()
#%%
optimizer = torch.optim.Adam(net.parameters(), lr=0.001,weight_decay=0.001)
criterion = nn.MSELoss(reduction='sum')
mean_criterion = nn.MSELoss(reduction='mean')

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=100, verbose=True)
#%%
shop_list=[0,1,8,9,11,13,17,20,23,27,29,30,32,33,40,43,51,54]
for shop in range(60):
    z=0
    trainloader=DataLoader(dataset(is_train=True,shop=shop),batch_size=4,shuffle=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001,weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=100, verbose=True)
    val_data,val_label=val(shop)
    val_data,val_label=torch.unsqueeze(val_data,0).float().cuda(),torch.unsqueeze(val_label,0).float().cuda()
    if shop not in shop_list:
        for epoch in range(4):
            z=0
            for num_batch,data in enumerate(trainloader,0):
                net.train()
                inputs,label=data
                #inputs,label=(inputs).float().cuda(),(label).float().cuda()
                inputs,label=(inputs).float(),(label).float()
                out=net(inputs)
                
                loss=torch.sqrt(criterion(out,label))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                z+=loss.item()
                
            #if epoch%10==0:
                #val_out=net(val_data)
                #loss=torch.sqrt(mean_criterion(val_out,val_label))
                #print('val loss= ',loss.item())
            scheduler.step(loss)
            print('shop %d: %.2f,  '%(shop,(z/(num_batch+1))))

        torch.save(net,'%dsave.pt'%shop)
        
