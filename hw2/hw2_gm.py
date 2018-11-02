#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import numpy as np


# In[2]:


# this SHOULD BE commented out further!
###sys.argv[3] = 'test_x.csv'
###sys.argv[4] = 'out_gm.csv'


# In[3]:


t = pd.read_csv(sys.argv[3])
f = np.load('./model_gm.npz')
w = f['w'][0]
b = f['b'][0][0]
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




