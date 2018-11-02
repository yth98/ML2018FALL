#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import numpy as np


# In[2]:


# this SHOULD BE commented out further!
###sys.argv = [0] * 5
###sys.argv[3] = 'test_x.csv'
###sys.argv[4] = 'out_best.csv'


# In[3]:


t = pd.read_csv(sys.argv[3]).astype('str').drop(['SEX','EDUCATION'], axis=1)
ohe_t = pd.get_dummies(t[['MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']], prefix=['MAR','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'])
t = pd.concat([t.drop(['MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'], axis='columns'),ohe_t], axis='columns')
t = t.astype('int64')
t.insert(t.columns.get_loc('PAY_5_7')+1,'PAY_5_8',np.zeros(len(t.index)))
t.insert(t.columns.get_loc('PAY_6_7')+1,'PAY_6_8',np.zeros(len(t.index)))
f = np.load('./model_best.npz')
w = f['w'][0]
b = f['b']
a = []
for i in range(len(t.index)):
    z = (w * t.iloc[i,:].values).sum() + b
    if(np.sign(z) <= 0): v = 0
    else: v = 1
    a += [['id_'+str(i),v]]


# In[4]:


ans = pd.DataFrame(a, columns=['id','value'])
#print(ans)
ans.to_csv(sys.argv[4],index=False)


# In[ ]:




