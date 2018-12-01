#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
from keras.models import Sequential, load_model


# In[4]:


model = [load_model('./model_3.h5'), load_model('./model_5.h5'), load_model('./model_6.h5')]


# In[6]:


test = pd.read_csv(sys.argv[1])
test_x = test['feature'].values
test_t = test['id'].values
x = []
for i in range(len(test_x)):
    x.append(np.reshape(np.fromstring(test_x[i], dtype=int, sep=' '),(48,48,1)))
x = np.array(x)
y = []
for m in model:
    y.append(m.predict(x))
y = np.array(y).mean(axis=0).argmax(axis=1)
print(y)
a = np.array([test_t,y]).T


# In[7]:


ans = pd.DataFrame(a, columns=['id','label'])
print(ans)
ans.to_csv(sys.argv[2],index=False)
