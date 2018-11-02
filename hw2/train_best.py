#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'notebook')
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt


# In[2]:


# https://stackoverflow.com/questions/3985619/
# performance concern
from scipy.special import expit


# In[3]:


# read files
# https://stackoverflow.com/a/41540037/2137255
data_x = pd.read_csv('./train_x.csv').astype('str').drop(['SEX','EDUCATION'], axis=1) # get_dummies doesn't apply on ints
data_y = pd.read_csv('./train_y.csv').astype('int64')


# In[4]:


# do one-hot encoding
ohe_x = pd.get_dummies(data_x[['MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']], prefix=['MAR','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'])
data_x = pd.concat([data_x.drop(['MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'], axis='columns'),ohe_x], axis='columns')
data_x = data_x.astype('int64')


# In[5]:


# init w and b
w = np.zeros((1,21+3+57))
b = 0


# In[6]:


data_x = data_x.values
data_y = data_y.T.values[0]


# In[7]:


def accuracy(p,x,y):
    w, b = p
    e = np.sign(w @ x.T + b) # do matmul on array: https://stackoverflow.com/a/40350379
    e[e == -1] = 0
    e = np.equal(e, y)
    return np.average(e)


# In[8]:


# cross entropy
# x, y as array
def loss(p,x,y):
    loss = 0
    w, b = p
    z = (w @ x.T + b).tolist()[0]
    f = expit(z)
    f = np.clip(f, 1E-320, 1-1E-16)
    for i in range(len(y)):
        loss -= y[i] * np.log(f[i]) + (1-y[i]) * np.log(1-f[i])
    return loss/len(y)


# In[9]:


# gradient
def gd(p,x,y):
    w, b = p
    diff = y - expit((w @ x.T + b).tolist()[0])
    return -diff @ x, -diff.sum()


# In[10]:


def log_reg(alpha, beta1, beta2, ite):
    global w, b
    # implement Adam https://arxiv.org/pdf/1412.6980.pdf
    # ref: https://seba-1511.github.io/dist_blog/
    mw = vw = np.zeros(w.shape)
    mb = vb = 0
    lss = []
    bk = False
    for t in range(1, ite+1): 
        # batch : 100 * 200
        batch = 100*(t%200)
        gw, gb = gd((w,b),data_x[batch:batch+100],data_y[batch:batch+100])
        mw, mb = mw*beta1 + gw*(1-beta1), mb*beta1 + gb*(1-beta1)
        vw, vb = vw*beta2+gw**2*(1-beta2), vb*beta2+gb**2*(1-beta2)
        a = alpha * (1-beta2**t)**0.5/(1-beta1**t)
        w -= a * mw / (vw**0.5+1E-8)
        b -= a * mb / (vb**0.5+1E-8)
        if(not (t%40)):
            ls = loss((w,b),data_x,data_y)
            lss += [ls]
            #print(ls, accuracy((w,b),data_x,data_y))
            if(accuracy((w,b),data_x,data_y))>0.822: bk = True
        if(bk): break
    return pd.Series(lss)


# In[11]:


def feature_scaling(x):
    # https://stackoverflow.com/questions/39277638/
    xmin = np.minimum.reduce(x)
    xmax = np.maximum.reduce(x)
    x = (x-[xmin]*len(x)) / ([xmax-xmin]*len(x))
    return x, xmin, xmax
def un_scaling(w, b, xmin, xmax):
    w /= (xmax-xmin)
    b -= (xmin*w).sum()
    return w, b


# In[12]:


data_x, x_min, x_max = feature_scaling(data_x)
log_reg(1E-3, 1-1E-1, 1-1E-3, 80*200)
w, b = un_scaling(w, b, x_min, x_max)
np.savez('model_best.npz', w=w, b=b)
print(w,b)


# In[ ]:




