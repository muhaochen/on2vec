import sys
if '../../src' not in sys.path:
    sys.path.append('../../src')

import numpy as np
from numpy import linalg as LA
import tensorflow as tf
import time

import data  
import model
import trainer
from tester_basic import Tester

model_file = '../dbpedia/db-model.ckpt'
data_file = '../dbpedia/db-data.bin'
test_data = '../../../onto2vec-dataset/dbpedia/db_onto.tsv'


# In[13]:

tester = Tester()
tester.build(save_path = model_file, data_save_path = data_file, pre_pool=True)
tester.load_test_data(test_data, splitter = '\t', line_end = '\n')


# In[16]:
upper_bound = 15000000
ht_map = tester.ht_map
print len(tester.this_data.cons)


# In[17]:

batch = []
count = 0
for i in range(len(tester.this_data.cons)):
    for j in range(len(tester.this_data.cons)):
        if i != j and (tester.ht_map.get(i)==None or tester.ht_map[i].get(j)==None):
            batch.append((i, j))
            count += 1
            if count > upper_bound:
                break
    if count > upper_bound:
        break
batch = np.array(batch)
print len(batch)


# In[29]:

import multiprocessing
from multiprocessing import Process, Value, Lock, Manager

cpu_count = multiprocessing.cpu_count()

manager = Manager()
rst_list = manager.list()

index = Value('i', 0, lock=True) #index
thres = 2.1

t0 = time.time()
# In[30]:

def test2(tester, index, batch, rst_list, thres):
    while index.value < len(batch):
        id = index.value
        index.value += 1
        if id % 1000000 == 0:
            print("Tested %d in %d seconds." % (id+1, time.time()-t0))
        h, t = batch[id]
        rel_pool = tester.vec_r
        target = np.array([0, 0, 0, 10000.])
        for rel in range(len(rel_pool)):
            head_vec = tester.projection(h, rel, h_or_t='h')
            tail_vec = tester.projection(t, rel, h_or_t='t')
            rel_vec = tester.rel_index2vec(rel)
            dis_score = LA.norm(head_vec+rel_vec-tail_vec)
            if dis_score < target[3]:
                target[0] = h
                target[1] = rel
                target[2] = t
                target[3] = dis_score
        # test_rel 
        # [[category, rank]*]
        if target[3] < thres:
            rst_list.append(target)


# In[27]:

processes = [Process(target=test2, args=(tester, index, batch, rst_list, thres)) for x in range(cpu_count / 3 - 1)]
for p in processes:
    p.start()
for p in processes:
    p.join()


# In[28]:

print len(rst_list)
hd = np.array(rst_list)


# In[ ]:

hd = hd[hd[:, 3].argsort()]
fp = open('predict_new.txt','w')
for i in range(min(200, len(hd))):
    line = hd[i]
    fp.write(tester.this_data.con_index2str(int(line[0])) + '\t' + tester.this_data.rel_index2str(int(line[1])) + '\t' + tester.this_data.con_index2str(int(line[2])) + '\t' + str(line[3]) + '\n')

