#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
np.random.seed(114514)
from tensorflow import set_random_seed
set_random_seed(28825252)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization


# In[4]:


train_x = pd.read_csv(sys.argv[1])['feature'].values
train_y = pd.get_dummies(pd.read_csv(sys.argv[1])['label']).values
x = []
x_vali = []
y = []
y_vali = []
for i in range(len(train_x)):
    a = np.fromstring(train_x[i], dtype=int, sep=' ')
    if(i>=1500):
        x.append(np.reshape(a,(48,48,1)))
        y.append(train_y[i])
    else:
        x_vali.append(np.reshape(a,(48,48,1)))
        y_vali.append(train_y[i])
x = np.array(x)
y = np.array(y)
x_vali = np.array(x_vali)
y_vali = np.array(y_vali)


# In[6]:


# I forgot to add activation layer to Conv2D, making my hw3 progress stuck for 2 weeks! (early baseline missed)
model = Sequential()
model.add(Convolution2D(120,3,activation='relu',input_shape=(48,48,1)))
model.add(Convolution2D(120,3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2)) # 80*22*22
model.add(Dropout(0.1))
model.add(Convolution2D(240,2,activation='relu'))
model.add(Convolution2D(240,2,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2)) # 160*10*10
model.add(Dropout(0.1))
model.add(Convolution2D(480,3,activation='relu'))
model.add(Convolution2D(480,3,activation='relu'))
model.add(Convolution2D(480,3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2)) # 240*3*3
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(720, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(720, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()


# In[7]:


history = model.fit(x, y, epochs=80, batch_size=100)


# In[9]:


model.save("model_3.h5")
