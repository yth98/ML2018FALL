#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Generative Model
import pandas as pd
import numpy as np


# In[2]:


# read files
data_x = pd.read_csv('./train_x.csv').astype('int64')
data_y = pd.read_csv('./train_y.csv').astype('int64')


# In[3]:


d = pd.concat([data_x,data_y], axis='columns')
d_0 = d.loc[d['Y']==0].drop(['Y'], axis='columns')
d_1 = d.loc[d['Y']==1].drop(['Y'], axis='columns')


# In[4]:


# calculate mu
mu_0 = d_0.mean().values
mu_1 = d_1.mean().values
# calculate sigma
l = len(mu_0)
y0 = len(d_0.index)#data_y['Y'].value_counts()[0]
y1 = len(d_1.index)#data_y['Y'].value_counts()[1]
sigma_0 = np.zeros((l,l))
for i in range(y0):
    s = np.matrix(d_0.iloc[i,:].values - mu_0)
    s = np.matmul(s.T, s)
    sigma_0 += s
sigma_1 = np.zeros((l,l))
for i in range(y1):
    s = np.matrix(d_1.iloc[i,:].values - mu_1)
    s = np.matmul(s.T, s)
    sigma_1 += s
sigma = np.matrix((sigma_0 + sigma_1) / (y0 + y1))
mu_0 = np.matrix(mu_0).T
mu_1 = np.matrix(mu_1).T


# In[5]:


sigma_inv = np.linalg.inv(sigma)
# calculate w and b
w = (mu_1-mu_0).T * sigma_inv
b = (mu_0.T*sigma_inv*mu_0 - mu_1.T*sigma_inv*mu_1) / 2 + np.log(y1/y0)


# In[6]:


np.savez('model_gm.npz', w=w, b=b)


# In[ ]:




