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

def sigmoid(x):
    return 1/(1+torch.exp(-x))

# Term2 Loss & Regularization
def frob(z):
    vec_i = torch.reshape(z,[-1])
    return torch.sum(torch.mul(vec_i,vec_i))

# Term1 Loss
# 1 Dimension Loss
def logistic_loss(label,y):
    return -torch.mean(label*torch.log(y)+(1-label)*(torch.log(1-y)))

# One-Hot Loss
def logistic_loss2(label,y):
    return -torch.mean(torch.sum(label*torch.log(y),axis=1))

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

def svd_initialization(X_roi, X_snp, du1, du2, NL_MCI_AD=False):
    train_size = int(X_roi.shape[0]*0.9)
    
    u_roi_svd1, _, v_roi_svd1 = np.linalg.svd(X_roi,full_matrices=False)
    u_roi_svd2, _, v_roi_svd2 = np.linalg.svd(u_roi_svd1,full_matrices=False)
    
    # SNP => One-Hot-Encoding
    if NL_MCI_AD:
        u_snp_svd1, _, v_snp_svd1 = np.linalg.svd(X_snp,full_matrices=False)
        u_snp_svd2, _, v_snp_svd2 = np.linalg.svd(u_snp_svd1,full_matrices=False)
        
        
        v_snp = torch.tensor(v_snp_svd1[0:du1, :], dtype=torch.double, requires_grad=True)
        u_snp = torch.tensor(u_snp_svd2[0:du2,0:du1], dtype=torch.double, requires_grad=True)
        
        w = torch.empty(du2, 3,dtype=torch.double,requires_grad=True)
        b = torch.zeros(1,3,dtype=torch.double, requires_grad=True)
    
    # SNP => Category Value: 0 or 1 or 2
    else:
        v_snp = np.zeros((du1,X_snp.shape[1],3))
        u_snp = np.zeros((du2,du1,3))
        
        for i in range(3):
            u_snp_svd1, _, v_snp_svd1 = np.linalg.svd(X_snp[:,:,i],full_matrices=False)
            u_snp_svd2, _, v_snp_svd2 = np.linalg.svd(u_snp_svd1,full_matrices=False)
            
            v_snp[:,:,i] = v_snp_svd1[0:du1, :]
            u_snp[:,:,i] = u_snp_svd2[0:du2,0:du1]
        
        v_snp = torch.tensor(v_snp,dtype=torch.double, requires_grad=True)
        u_snp = torch.tensor(u_snp,dtype=torch.double, requires_grad=True)
        
        w = torch.empty(du2, 1,dtype=torch.double,requires_grad=True)
        b = torch.zeros(1,dtype=torch.double, requires_grad=True)
        
    v_roi = torch.tensor(v_roi_svd1[0:du1, :],dtype=torch.double, requires_grad=True)
    u_roi = torch.tensor(u_roi_svd2[0:du2,0:du1],dtype=torch.double, requires_grad=True)
    
    u = torch.tensor(u_roi_svd2[:,0:du2],dtype=torch.double, requires_grad=True)
    
    # Xavier Initialization
    nn.init.xavier_uniform_(w)
    
    b = torch.tensor(0.1,dtype=torch.double,requires_grad=True)
    
    return u,u_roi,v_roi,u_snp,v_snp,w,b
def save_parameter(max_steps, tol, file_path, du1, du2, alpha1=1, alpha2=1, NL_MCI_AD=False):
    # Data Load
    X_roi,X_snp,Y_Label = load_data(file_path,NL_MCI_AD)
    # Train Test split parameter => 0.9:1
    train_size = int(X_roi.shape[0]*0.9)
    sample_size = X_roi.shape[0]
    
    # SVD Initialization
    u,u_roi,v_roi,u_snp,v_snp,w,b = svd_initialization(X_roi,X_snp,du1,du2, NL_MCI_AD)
    # Label Split
    y_ = Y_Label[0:train_size,:]
    y__ = Y_Label[train_size:sample_size,:]
    
    # Optimizer
    optimizer = torch.optim.Adam([u,u_roi,v_roi,u_snp,v_snp,w,b], lr=1e-4)
    
    # For Model Performance
    funval = [0]
    for i in range(max_steps+1):
        # Overfitting => 1. Dropout
        u_train = u[0:train_size,:]
        u_train = torch.dropout(u_train,p=0.3,train=True)
        if NL_MCI_AD:
            roi_ = torch.matmul(u,torch.sigmoid(torch.matmul(u_roi,v_roi)))
            snp_ = torch.matmul(u,torch.sigmoid(torch.matmul(u_snp,v_snp)))
            prediction_ = F.softmax(torch.matmul(u_train,w)+b, dim=1)
            
            # Model Loss
            prediction_loss = logistic_loss2(y_,prediction_)+alpha1*frob(X_roi-roi_)+alpha2*frob(X_snp-snp_)
            
        else:
            roi_ = torch.sigmoid(torch.matmul(u,torch.square(torch.matmul(u_roi,v_roi))))
            # Becuas of One-Hot-Encoding
            snp_0 = torch.sigmoid(torch.matmul(u,torch.square(torch.matmul(u_snp[:,:,0],v_snp[:,:,0]))))
            snp_1 = torch.sigmoid(torch.matmul(u,torch.square(torch.matmul(u_snp[:,:,1],v_snp[:,:,1]))))
            snp_2 = torch.sigmoid(torch.matmul(u,torch.square(torch.matmul(u_snp[:,:,2],v_snp[:,:,2]))))
            snp_ = torch.stack([snp_0,snp_1,snp_2],axis=2)
            
            prediction_ = torch.sigmoid(torch.matmul(u_train,w)+b)
        
            # Model Loss
            prediction_loss = logistic_loss(y_,prediction_)+alpha1*frob(X_roi-roi_)+alpha2*frob(X_snp-snp_)
        
        # Overfitting => 2. L2 Regularization
        regularization_loss = 0.01*frob(u_roi) + 0.01*frob(u_snp)+ 0.01*frob(u) + 0.01*frob(v_roi) + 0.01*frob(v_snp)
        
        # Total Loss
        total_loss = prediction_loss+regularization_loss
        
        # Weight Update
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        
        # When there is no improvement in model performance
        total_loss_digit = total_loss.detach().item()
        funval.append(total_loss_digit)
        
        if abs(funval[i+1]-funval[i]) < tol:
            train_auc = roc_auc_score(y_.detach(),prediction_.detach())
            print('Early Stopping')
            train_auc = roc_auc_score(y_.detach(),prediction_.detach())
            print('Iteration: ',i,' Train_AUC:',train_auc)
            
            if NL_MCI_AD:
                print("Label: ",y_.detach()[25:30])
                print("Model Prediction: ", prediction_.detach()[25:30],'\n')
                
            else:
                print("Label: ",y_.detach()[10:15].T)
                print("Model Prediction: ", prediction_.detach()[10:15].T,'\n')
            break
            
        # Wrong Model Loss
        if math.isnan(total_loss):
            print("Totla Loss Exception2")
            print(funval)
            break
    
        # Metric: AUC Score
        # 1. Train AUC
        if i%5000 == 0:
            train_auc = roc_auc_score(y_.detach(),prediction_.detach())
            print('Iteration: ',i,' Train_AUC:',train_auc)
            
            if NL_MCI_AD:
                print("Label: ",y_.detach()[25:30])
                print("Model Prediction: ", prediction_.detach()[25:30],'\n')
                
            else:
                print("Label: ",y_.detach()[10:15].T)
                print("Model Prediction: ", prediction_.detach()[10:15].T,'\n')
    
    # 2. Test AUC
    u_test = u.detach()[train_size:sample_size,:]
    
    if NL_MCI_AD:
        prediction__ = F.softmax(torch.matmul(u_test,w)+b, dim=1)
    else:
        prediction__ = torch.sigmoid(torch.matmul(u_test,w)+b)
    
    test_auc = roc_auc_score(y__.detach(),prediction__.detach())
    
    return train_auc,test_auc,(u,u_roi,v_roi,u_snp,v_snp,w,b)

def save_paramet_directory(u,u_roi,v_roi,u_snp,v_snp,w,b,directory):
    np.save(directory+'u.npy',u.detach().numpy())
    np.save(directory+'u_roi.npy',u_roi.detach().numpy())
    np.save(directory+'v_roi.npy',v_roi.detach().numpy())
    np.save(directory+'u_snp.npy',u_snp.detach().numpy())
    np.save(directory+'v_snp.npy',v_snp.detach().numpy())
    np.save(directory+'w.npy',w.detach().numpy())
    np.save(directory+'b.npy',b.detach().numpy())

# File Path
subject = ['NL_MCI','NL_AD','NL_MCI_AD']
file_path = ['./data/'+subject[0]+'_Data/', './data/'+subject[1]+'_Data/', './data/'+subject[2]+'_Data/']

# Hyperparameter
max_iter = 20000
tol = 1e-7

print('Save parameter....')


# NL_MCI_Parameter_Save
directory = './Result'+subject[0]+'/'
f = file_path[0]

if not os.path.exists(directory):
    os.makedirs(directory)

train_auc,test_auc,(u,u_roi,v_roi,u_snp,v_snp,w,b) = save_parameter(max_iter,tol,f,100,110,alpha1=0.001,alpha2=1)
save_paramet_directory(u,u_roi,v_roi,u_snp,v_snp,w,b,directory)

# NL_AD_Parameter_Save
directory = './Result/'+subject[1]+'/'
f = file_path[1]

if not os.path.exists(directory):
    os.makedirs(directory)
    
train_auc,test_auc,(u,u_roi,v_roi,u_snp,v_snp,w,b) = save_parameter(max_iter,tol,f,100,110,alpha1=1,alpha2=1)
save_paramet_directory(u,u_roi,v_roi,u_snp,v_snp,w,b,directory)


# NL_MCI_AD_Parameter_Save
directory = './Result'+subject[2]+'/'
f = file_path[2]

if not os.path.exists(directory):
    os.makedirs(directory)
    
train_auc,test_auc,(u,u_roi,v_roi,u_snp,v_snp,w,b) = save_parameter(max_iter,tol,f,150,110,alpha1=1,alpha2=1,NL_MCI_AD=True)
save_paramet_directory(u,u_roi,v_roi,u_snp,v_snp,w,b,directory)

