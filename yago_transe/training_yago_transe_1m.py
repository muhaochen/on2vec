
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
import transe
import trainer_transe
from trainer_transe import Trainer


# In[4]:
model_path = 'yago-transe-1m.ckpt'
data_path = 'yago-data-1m.bin'
filename = "../../../onto2vec-dataset/Yago/yagoFactsFiltCD100Train.tsv"
more_filt = ['../../../onto2vec-dataset/Yago/yagoFactsFiltCD100Test.tsv']

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
m_train.build(this_data, dim=100, batch_size=1024, save_path = model_path, data_save_path = data_path, L1=False)


# In[ ]:

m_train.train(epochs=500, save_every_epoch=50, lr=0.001, a1=0.0, a2=1., m1=0.5, m2=0.5, balance_alpha1=True, feed_pre=False)


# In[ ]:



