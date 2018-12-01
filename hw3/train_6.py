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
    if(i % 18 != 0):
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


model = Sequential()
model.add(Convolution2D(180,3,activation='relu',padding='same',input_shape=(48,48,1)))
model.add(Convolution2D(180,3,activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2)) # 180*24*24
model.add(Dropout(0.2))
model.add(Convolution2D(360,2,activation='relu',padding='same'))
model.add(Convolution2D(360,2,activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2)) # 360*12*12
model.add(Dropout(0.2))
model.add(Convolution2D(540,2,activation='relu',padding='same'))
model.add(Convolution2D(540,2,activation='relu',padding='same'))
model.add(Convolution2D(540,2,activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2)) # 540*6*6
model.add(Dropout(0.2))
model.add(Convolution2D(720,2,activation='relu',padding='same'))
model.add(Convolution2D(720,2,activation='relu',padding='same'))
model.add(Convolution2D(720,2,activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2)) # 720*3*3
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(2560, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(960, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()


# In[7]:


history = model.fit(x, y, epochs=100, batch_size=140, validation_data=(x_vali,y_vali))


# In[10]:


model.save("model_6.h5")
