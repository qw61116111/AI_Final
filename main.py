import pandas as pd
import numpy as np

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import csv    
from torchvision import transforms
import random
import time


count_month=torch.from_numpy(np.load('C:\\Users\Q56091087\Desktop\data_array.npy'))

submit=pd.read_csv('C://Users/Q56091087/Desktop/data/test.csv')
temp=np.zeros([60,22170])
shop_list=[0,1,8,9,11,13,17,20,23,27,29,30,32,33,40,43,51,54]

def test(shop):
    data=count_month[22:34,shop]
    return data
model=torch.load('C://Users/Q56091087/.spyder-py3/Class_AI/Final_hw/save.pt', map_location=torch.device('cpu'))
model.eval()
for shop in range(60):
    if shop not in shop_list:

        test_data=test(shop).float()
        test_data=torch.unsqueeze(test_data,0)
        out=model(test_data)
        out[torch.where(out<0)]=0
        out[torch.where(out>20)]=20
        temp[shop]=out.detach().numpy()

with open('C://Users/Q56091087/.spyder-py3/Class_AI/Final_hw/submission.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(submit)):
        if i!=0:
            writer.writerow([i,temp[submit['shop_id'][i],submit['item_id'][i]]])
        else:
            writer.writerow(['ID','item_cnt_month'])
            writer.writerow([i,temp[submit['shop_id'][i],submit['item_id'][i]]])
