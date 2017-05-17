from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
if '../../src' not in sys.path:
    sys.path.append('../../src')

import os
if not os.path.exists('../../results/yago100/prc'):
    os.makedirs('../../results/yago100/prc')

import numpy as np
from numpy import linalg as LA
import tensorflow as tf
import time

import data  
import transh
import trainer
from tester_transh import Tester

transh_file = 'yago-transh-1m.ckpt'
data_file = 'yago-data-1m.bin'
test_data = '../../../onto2vec-dataset/Yago/yagoFactsFiltCD100Test.tsv'
result_file = '../../results/yago100/prc/prc_transh_tail.txt'
result_file2 = '../../results/yago100/prc/prc_transh_head.txt'
result_file3 = '../../results/yago100/prc/prc_transh_rel.txt'

trsym = ['isLocatedIn', 'isConnectedTo','isMarriedTo', 'hasNeighbor', 'isConnectedTo', 'dealsWith']
hier = ['hasChild', 'hasAcademicAdvisor', 'isLocatedIn', 'isLeaderOf', 'hasGender']
r_trsym = set([])
r_hier = set([])

tester = Tester()
tester.build(save_path = transh_file, data_save_path = data_file, pre_pool=True)
tester.load_test_data(test_data, splitter = '\t', line_end = '\n')

TopK = len(tester.this_data.rels)

import multiprocessing
from multiprocessing import Process, Value, Lock, Manager

cpu_count = multiprocessing.cpu_count()

manager = Manager()

index = Value('i', 0, lock=True) #index

num_rel = Value('i', 0, lock=True)

score_rel = manager.list() #scores for each case (and its rel)


t0 = time.time()

def test2(tester, index, score_rel, num_rel):
    while index.value < len(tester.test_triples):
        id = index.value
        index.value += 1
        if id % 100 == 0:
            print("Tested %d in %d seconds." % (id+1, time.time()-t0))
        h, r, t = tester.test_triples[id]
        rel_pool = tester.vec_r
        # test_rel 
        # [[category, rank]*]
        this_score_rel = []
        hit = False
        for rel_pred in range(len(tester.this_data.rels)):
            #head_vec = tester.projection(h, rel_pred, h_or_t='h')
            #tail_vec = tester.projection(t, rel_pred, h_or_t='t')
            #rel_vec = tester.rel_index2vec(rel_pred)
            this_pred = np.array([0., 0.])
            if rel_pred in tester.ht_map[h][t]:
                this_pred[0] = 1.
                num_rel.value += 1
                hit = True
            this_pred[1] = tester.dissimilarity(h,r, t)
            this_score_rel.append(this_pred)
            if hit:
                break
        for line in this_score_rel:
            score_rel.append(line)
        '''
        this_rank_rel = tester.rank_index_from(target_vec, rel_pool, r, self_id = r)
        if this_rank_rel < TopK:
            this_score_rel = np.zeros((TopK, 2))
            pred = tester.kNN(target_vec, rel_pool, topk = TopK, self_id=None)
            for i in range(len(pred)):
                if pred[i][0] in tester.ht_map[h][t]:
                    this_score_rel[i][0] = 1.
                    num_rel.value += 1
                this_score_rel[i][1] = pred[i][1]
        score_rel.append(this_score_rel)
        '''


# tester.rel_num_cases
index = Value('i', 0, lock=True) #index
print('Average norm=%d' % np.mean([LA.norm(vec, ord=(1 if tester.this_data.L1 else 2)) for vec in tester.vec_c]))
processes = [Process(target=test2, args=(tester, index, score_rel, num_rel)) for x in range(cpu_count - 1)]
for p in processes:
    p.start()
for p in processes:
    p.join()

with open(result_file3, 'w') as fp:
    hd = np.array(score_rel).reshape((-1, 2))
    print(score_rel[1])
    print(score_rel[3])
    hd = hd[hd[:, 1].argsort()]
    #hd.view('i8,i8').sort(order=['f1'], axis=0)
    count = 0
    index = 0
    while count < num_rel.value and index < len(hd):
        fp.write('\t'.join([str(x) for x in hd[index]]) + '\n')
        if hd[index][0] > 0.:
            count += 1
        index += 1
print("PRC for rel..")