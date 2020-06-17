import pandas as pd
import torch
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
import math
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import matplotlib.pyplot as plt

def load_data(file_path,NL_MCI_AD=False):
    if NL_MCI_AD:
        X_roi = pd.read_csv(file_path+'X_roi.csv', index_col=0)
        X_roi = torch.tensor(X_roi.values,dtype=torch.double,requires_grad=False)

        X_snp = pd.read_csv(file_path+'X_snp.csv', index_col=0)
        X_snp = torch.tensor(X_snp.values,dtype=torch.double,requires_grad=False)

        Y = np.load(file_path+'Label.npy')
        Y_Label = torch.tensor(Y,dtype=torch.double, requires_grad=False)

    else:
        X_roi = pd.read_csv(file_path+'X_roi.csv',index_col=0)
        X_roi = torch.tensor(X_roi.values,dtype=torch.double,requires_grad=False)

        X_snp = np.load(file_path+'X_snp.npy')
        X_snp = torch.tensor(X_snp,dtype=torch.double,requires_grad=False)

        Y_Label = pd.read_csv(file_path+'Label.csv',index_col=0)
        Y_Label = torch.tensor(Y_Label.values,dtype=torch.double,requires_grad=False)

    return X_roi,X_snp,Y_Label

def load_parameter_directory(file_path):
    u = np.load(file_path+'u.npy')
    u = torch.tensor(u)
    
    u_roi = np.load(file_path+'u_roi.npy')
    u_roi = torch.tensor(u_roi)
    
    v_roi = np.load(file_path+'v_roi.npy')
    v_roi = torch.tensor(v_roi)
    
    u_snp = np.load(file_path+'u_snp.npy')
    u_snp = torch.tensor(u_snp)
    
    v_snp = np.load(file_path+'v_snp.npy')
    v_snp = torch.tensor(v_snp)
    
    w = np.load(file_path+'w.npy')
    w = torch.tensor(w)
    
    b = np.load(file_path+'b.npy')
    b = torch.tensor(b)
    
    return u,u_roi,v_roi,u_snp,v_snp,w,b

subject = ['NL_MCI','NL_AD','NL_MCI_AD']
file_path = ['./data/'+subject[0]+'_Data/', './data/'+subject[1]+'_Data/', './data/'+subject[2]+'_Data/']

# NL_MCI
# Data & Parameter Load
directory = './Result/'+subject[0]+'/'
f = file_path[0]

u,u_roi,v_roi,u_snp,v_snp,w,b = load_parameter_directory(directory)
X_roi,X_snp,Y_Label = load_data(f,NL_MCI_AD=False)

train_size = int(X_roi.shape[0]*0.9)
sample_size = X_roi.shape[0]

y__ = Y_Label[train_size:sample_size,:]
u_test = u.detach()[train_size:sample_size,:]

prediction__ = torch.sigmoid(torch.matmul(u_test,w)+b)
test_auc = roc_auc_score(y__.detach(),prediction__.detach())

print('NL_MCI_AUC: ',test_auc)

# NL_AD
# Data & Parameter Load
directory = './Result/'+subject[1]+'/'
f = file_path[1]

u,u_roi,v_roi,u_snp,v_snp,w,b = load_parameter_directory(directory)
X_roi,X_snp,Y_Label = load_data(f,NL_MCI_AD=False)

train_size = int(X_roi.shape[0]*0.9)
sample_size = X_roi.shape[0]

y__ = Y_Label[train_size:sample_size,:]
u_test = u.detach()[train_size:sample_size,:]

prediction__ = torch.sigmoid(torch.matmul(u_test,w)+b)
test_auc = roc_auc_score(y__.detach(),prediction__.detach())

print('NL_AD_AUC: ',test_auc)

# NL_MCI_AD
# Data & Parameter Load
directory = './Result/'+subject[2]+'/'
f = file_path[2]

u,u_roi,v_roi,u_snp,v_snp,w,b = load_parameter_directory(directory)
X_roi,X_snp,Y_Label = load_data(f,NL_MCI_AD=True)

train_size = int(X_roi.shape[0]*0.9)
sample_size = X_roi.shape[0]

y__ = Y_Label[train_size:sample_size,:]
u_test = u.detach()[train_size:sample_size,:]

prediction__ = F.softmax(torch.matmul(u_test,w)+b, dim=1)
test_auc = roc_auc_score(y__.detach(),prediction__.detach())

print('NL_MCI_AD',test_auc)
