#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import numpy as np


# In[2]:


# this SHOULD BE commented out further!
###sys.argv[1] = 'test.csv'
###sys.argv[2] = 'out.csv'


# In[3]:


def testing(ps,pid):
    wb = np.load('./model.npy')
    X = ps.loc[ps[0] == pid].iloc[:,2:11].reset_index(drop=True).astype('float64')
    X.columns = range(9) # reset columns
    S = pd.DataFrame(np.zeros((len(X.index),len(X.columns))))
    for i in range(len(wb)):
        if(i==0): continue
        else:
            W = pd.DataFrame(wb[i])
            S += W * (X**i)
    return wb[0][0][0] + S.sum().values.sum()


# In[4]:


qs = pd.read_csv(sys.argv[1],header=None).replace('NR',0)
todo = qs[0].unique()


# In[5]:


ans = pd.DataFrame(data={0:['id'],1:['value']})
for qid in todo:
    value = testing(qs,qid)
    #print(qs[0])
    ans = ans.append(pd.Series([qid,value]),ignore_index=True)


# In[6]:


#print(ans)


# In[7]:


ans.to_csv(sys.argv[2],header=False,index=False)

