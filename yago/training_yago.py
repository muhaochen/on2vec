
# coding: utf-8

# In[1]:



# In[2]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[3]:

import sys
if '../../src' not in sys.path:
    sys.path.append('../../src')
    
import numpy as np
import os
import tensorflow as tf

import data  # we don't import individual things in a model. This is to make auto reloading in Notebook happy
import model
import trainer
from trainer import Trainer


# In[4]:
model_path = 'yago-model.ckpt'
data_path = 'yago-data.bin'
filename = "../../../onto2vec-dataset/Yago/yagoFactsFiltCDTTrain.tsv"
more_filt = ['../../../onto2vec-dataset/Yago/yagoFactsFiltCDTTest.tsv']

refine = ['hasChild','isLeaderOf']
merge = ['isLocatedIn','hasGender']

this_data = data.Data()
this_data.load_data(filename=filename, 
                    rel_tran=['isLocatedIn', 'isConnectedTo'], 
                    rel_ref=refine, 
                    rel_coe=merge,
                    rel_inv=['isMarriedTo', 'hasNeighbor', 'isConnectedTo', 'dealsWith'])
for f in more_filt:
    this_data.record_more_data(f)


# In[ ]:

m_train = Trainer()
m_train.build(this_data, dim=100, batch_size=500, save_path = model_path, data_save_path = data_path, L1=False)


# In[ ]:

m_train.train(epochs=500, save_every_epoch=50, lr=0.001, a1=0.75, a2=2.0, m1=1., m2=1., balance_alpha1=False, feed_pre=False)


# In[ ]:



