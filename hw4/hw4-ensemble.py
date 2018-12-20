#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


# In[3]:


import emoji
import jieba
jieba.set_dictionary(sys.argv[2])
from gensim.models import Word2Vec


# In[4]:


test_x = [list(jieba.cut(emoji.demojize(line.rstrip("\n").split(",",maxsplit=1)[1]), cut_all=False)) for line in open(sys.argv[1])][1:]
test_t = [line.rstrip("\n").split(",",maxsplit=1)[0] for line in open(sys.argv[1])][1:]
y = []
for m_path, w_path in zip(['model_2.h5','model_3.h5','model_4.h5'],['word2vec_2.model','word2vec_3.model','word2vec_4.model']):
    model = load_model(m_path)
    w_model = Word2Vec.load(w_path)
    test_x_vec = []
    for sent in test_x:
        aa = []
        for word in sent:
            try: aa.append(w_model.wv[word])
            except KeyError: pass
        test_x_vec.append(aa)
    test_x_vec = pad_sequences(test_x_vec,maxlen=60,padding='post',dtype='float64')
    y.append(model.predict(test_x_vec))


# In[7]:


a = np.array([test_t,np.array(y).mean(axis=0).argmax(axis=1)]).T
ans = pd.DataFrame(a, columns=['id','label'])
#print(ans)
ans.to_csv(sys.argv[3],index=False)


# In[ ]:




