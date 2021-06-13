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


path="C://Users/Q56091087/Desktop/data/sales_train.csv"
data=pd.read_csv(path)

data = data[data['item_price'] < 300000]
data = data[data['item_cnt_day'] < 1000]
data=data[data['item_cnt_day'] >0]

count_month=np.zeros([34,60,22170])

for i in range(len(data['date_block_num'])):
    try:
        count_month[data['date_block_num'][i]][data['shop_id'][i]][data['item_id'][i]]+=data['item_cnt_day'][i]
    except:
        pass
   
count_month=torch.from_numpy(count_month)
np.save('data_array', count_month)