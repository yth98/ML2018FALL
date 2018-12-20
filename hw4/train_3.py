#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
np.random.seed(114514)
from tensorflow import set_random_seed
set_random_seed(114514)
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.preprocessing.sequence import pad_sequences


# In[3]:


import jieba
jieba.set_dictionary(sys.argv[4])
from gensim.models import Word2Vec
import emoji


# In[4]:


train_x = [line.rstrip("\n").split(",",maxsplit=1)[1] for line in open(sys.argv[1])][1:]
train_y = pd.read_csv(sys.argv[2])
print("Offensive: "+str(train_y[train_y['label']==1].size)+"/"+str(train_y.size))
train_y = pd.get_dummies(train_y['label']).values


# In[5]:


# Word Segmentation
x = []
x_vali = []
y = []
y_vali = []
for i in range(len(train_x)):
    a = list(jieba.cut(emoji.demojize(train_x[i]), cut_all=False))
    if(i % 30 != 29):
        x.append(a)
        y.append(train_y[i])
    else:
        x_vali.append(a)
        y_vali.append(train_y[i])
x = np.array(x)
y = np.array(y)
x_vali = np.array(x_vali)
y_vali = np.array(y_vali)
test_x = [list(jieba.cut(emoji.demojize(line.rstrip("\n").split(",",maxsplit=1)[1]), cut_all=False)) for line in open(sys.argv[3])][1:]


# In[6]:


# Word to Vector
w_model = Word2Vec(np.concatenate((x,x_vali,test_x)), size=250, sg=1, iter=7, max_final_vocab=35000)
w_model.save("word2vec_3.model")
x_vec = []
x_vali_vec = []
for sent in x:
    aa = []
    for word in sent:
        if(word==':'): pass
        try: aa.append(w_model.wv[word])
        except KeyError: pass
    x_vec.append(aa)
for sent in x_vali:
    aa = []
    for word in sent:
        if(word==':'): pass
        try: aa.append(w_model.wv[word])
        except KeyError: pass
    x_vali_vec.append(aa)


# In[7]:


ll=0
for i in x_vec:
    ll+=len(i)
print(ll/len(x_vec))
ll=0
for i in x_vec:
    ll=max([ll,len(i)])
print(ll)


# In[8]:


# Padding
x_vec      = pad_sequences(x_vec,maxlen=60,padding='post',dtype='float64')
x_vali_vec = pad_sequences(x_vali_vec,maxlen=60,padding='post',dtype='float64')


# In[10]:


model = Sequential()
model.add(LSTM(units=5, activation='tanh',input_shape=(60,250)))
model.add(Dropout(0.1))
model.add(Dense(400, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[11]:


history = model.fit(x_vec, y, epochs=15, batch_size=100, validation_data=(x_vali_vec,y_vali))


# In[14]:


model.save("model_3.h5")


# In[ ]:




