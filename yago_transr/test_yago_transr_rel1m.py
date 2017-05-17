from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
if '../../src' not in sys.path:
    sys.path.append('../../src')

import os
if not os.path.exists('../../results/yago100_v2'):
    os.makedirs('../../results/yago100_v2')

import numpy as np
from numpy import linalg as LA
import tensorflow as tf
import time

import data  
import transr
import trainer_transr
from tester_transr import Tester

model_file = 'yago-transr-1m.ckpt'
data_file = 'yago-data-1m.bin'
test_data = '../../../onto2vec-dataset/Yago/yagoFactsFiltCD100Test.tsv'
result_file2 = '../../results/yago100_v2/test_yago_rel_transr.txt'

trsym = ['isLocatedIn', 'isConnectedTo','isMarriedTo', 'hasNeighbor', 'isConnectedTo', 'dealsWith']
hier = ['hasChild', 'hasAcademicAdvisor', 'isLocatedIn', 'isLeaderOf', 'hasGender']
r_trsym = set([])
r_hier = set([])

tester = Tester()
tester.build(save_path = model_file, data_save_path = data_file, pre_pool=True)
tester.load_test_data(test_data, splitter = '\t', line_end = '\n')

for r in trsym:
    r_trsym.add(tester.this_data.rel_str2index(r))
for r in hier:
    r_hier.add(tester.this_data.rel_str2index(r))

import multiprocessing
from multiprocessing import Process, Value, Lock, Manager

cpu_count = multiprocessing.cpu_count()

manager = Manager()
manager2 = Manager()

index = Value('i', 0, lock=True) #index
score_rel = manager.list() #scores for each case (and its rel)
score_trsym = manager.list()
score_hier = manager.list()

t0 = time.time()
        

def test2(tester, index, score_rel, score_trsym, score_hier):
    while index.value < len(tester.test_triples):
        id = index.value
        index.value += 1
        if id % 100 == 0:
            print("Tested %d in %d seconds." % (id+1, time.time()-t0))
        h, r, t = tester.test_triples[id]
        rel_pool = tester.vec_r
        target = np.array([-1, 10000.])
        for rel in range(len(rel_pool)):
            head_vec = tester.projection(h, rel, h_or_t='h')
            tail_vec = tester.projection(t, rel, h_or_t='t')
            rel_vec = tester.rel_index2vec(rel)
            dis_score = LA.norm(head_vec+rel_vec-tail_vec)
            if dis_score < target[1]:
                target[0] = rel
                target[1] = dis_score
        # test_rel 
        # [[category, rank]*]
        hit = 0.
        if target[0] == r:
            hit = 1.
        score_rel.append(hit)     
        if r in r_trsym:
            score_trsym.append(hit)
        if r in r_hier:
            score_hier.append(hit)

processes = [Process(target=test2, args=(tester, index, score_rel, score_trsym, score_hier)) for x in range(cpu_count - 1)]
for p in processes:
    p.start()
for p in processes:
    p.join()

with open(result_file2, 'w') as fp:
    fp.write('Rel\n')
    score = 0.
    for s in score_rel:
        score += s
    score /= len(score_rel)
    fp.write(str(score) + '\n')
    fp.write('trsym\n')
    score = 0.
    for s in score_trsym:
        score += s
    score /= len(score_trsym)
    fp.write(str(score) + '\n')
    fp.write('hier\n')
    score = 0.
    for s in score_hier:
        score += s
    score /= len(score_hier)
    fp.write(str(score) + '\n')
    