#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt


# In[2]:


# Do not use pandas.DataFrame for calculation!
# It will be too slow!
def lin_reg(x,wb):
    X = np.array(x)
    S = np.zeros(X.shape)
    for i in range(2):
        if(i==0): continue
        S += wb[i] * (X**i)
    return wb[0] + np.sum(S)


# In[3]:


def loss(sets,wb):
    loss = 0
    for st in sets:
        x, y = st
        loss += (y-lin_reg(x,wb)) ** 2
    return (loss / len(sets)) ** 0.5


# In[4]:


def gd_grad(sets,wb):
    grad = [0] + [np.zeros((18,9))]
    for st in sets:
        x, y = st
        X = np.array(x)
        for i in range(2):
            # derivative
            if(i==0): grad[0] += 2 * (lin_reg(x,wb) - y)
            else: grad[i] += 2 * (lin_reg(x,wb) - y) * (X**i)
    return grad


# In[5]:


train = pd.read_csv('./train.csv',encoding='big5').replace('NR',0).drop(['日期','測站','測項'],'columns').astype('float64')
# I think NR means 'no rainfall', and first 3 columns 日期,測站,測項 are not needed.
def grab_train(mm, dd, hh):
    if(mm<1 or mm>12 or dd<1 or dd>20 or hh<0 or hh>23 or (dd==20 and hh>14)):
        print('no data for given date {0}/{1} {2}:00 !'.format(mm,dd,hh))
        return None
    offset = (18*(dd-1+(mm-1)*20))
    x = train.iloc[offset:offset+18,hh:min(hh+10,24)].reset_index(drop=True)
    if(hh>14): # concatenate tomorrow
        x = pd.concat([x , train.iloc[offset+18:offset+36,0:hh-14].reset_index(drop=True)], 'columns')
    y = x.iloc[9,9]                   # PM2.5
    x = x.iloc[:,0:9].values.tolist() # slice 10th column
    return x, y


# In[6]:


# if we take continuous 10 hours as a set,
# 1 hour spacing like 00,1/1~09,1/1 ; 18,12/19~03,12/20 ...
# there will be (24*20-9)*12 = 5652 sets.
all_sets = []
for m in range(1,13):
    for d in range(1,21):
        if(d!=20):
            for h in range(0,24): all_sets.append(grab_train(m,d,h))
        else:
            for h in range(0,15): all_sets.append(grab_train(m,d,h))


# In[7]:


# generate 90% training sets and 10% validation sets
# should not be affected by seasons
train_sets = []
valid_sets = []
for i in range(len(all_sets)):
    if(i%10 == 0): valid_sets.append(all_sets[i])
    else: train_sets.append(all_sets[i])


# In[8]:


# feature scaling
train_x_min = train_x_max = train_sets[0][0]
for ts in train_sets:
    # https://stackoverflow.com/questions/39277638/
    train_x_min = np.minimum.reduce([ts[0],train_x_min])
    train_x_max = np.maximum.reduce([ts[0],train_x_max])
for i in range(len(train_sets)):
    train_sets[i] = ((train_sets[i][0] - train_x_min) / (train_x_max - train_x_min)).tolist(), train_sets[i][1]


# In[9]:


# initiate weight and bias
W = [0, np.zeros((18,9))]


# In[10]:


# we train here!
def training(eta,ite):
    RMSE = []
    loss_prev = 100
    for i in range(ite):
        loss_now = loss(train_sets, W)
        #print(loss_now)
        if(loss_now > loss_prev): break
        loss_prev = loss_now
        RMSE.append(loss_now)
        grad = gd_grad(train_sets, W)
        for j in range(2):
            W[j] -= grad[j] * eta
    return RMSE


# In[11]:


# Matplotlib is not included in python standard library.
pd.Series(training(1E-6 * 7.6, 100000))#.plot() # takes a whole day!
#plt.xlabel('Iteration')
#plt.ylabel('RMSE')


# In[12]:


# recover from feature scaling
W[0] -= (train_x_min/(train_x_max-train_x_min)).sum()
W[1] /= (train_x_max-train_x_min)


# In[13]:


print('validation loss: '+str(loss(valid_sets, W)))
for i in range(len(W)):
    if(i==0): W[0] = W[0] * np.ones((18,9)) # padding
    else: W[i] = W[i].tolist()
np.save('model.npy', W)
