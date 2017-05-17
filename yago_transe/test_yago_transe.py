from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
if '../../src' not in sys.path:
    sys.path.append('../../src')

import os
if not os.path.exists('../../results/dbpedia'):
    os.makedirs('../../results/dbpedia')

import numpy as np
from numpy import linalg as LA
import tensorflow as tf
import time

import data  
import transe
import trainer_transe
from tester_transe import Tester

model_file = 'yago-transe.ckpt'
data_file = 'yago-data.bin'
test_data = '../../../onto2vec-dataset/Yago/yagoFactsFiltCDTTest.tsv'
result_file = '../../results/yago/test_yago_transe.txt'
result_file2 = '../../results/yago/test_yago_rel_transe.txt'

trsym = ['isLocatedIn', 'isConnectedTo','isMarriedTo', 'hasNeighbor', 'isConnectedTo', 'dealsWith']
hier = ['hasChild', 'hasAcademicAdvisor', 'isLocatedIn', 'isLeaderOf', 'hasGender']
r_trsym = set([])
r_hier = set([])

TopK = 10

tester = Tester()
tester.build(save_path = model_file, data_save_path = data_file)
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
score_head = manager.list() #scores for each case (and its rel)
score_rel = manager.list() #scores for each case (and its rel)
score_tail = manager.list() #scores for each case (and its rel)
rank_head = manager2.list() #rank, and its rel
rank_rel = manager2.list() #rank, and its rel
rank_tail = manager2.list() #rank, and its rel
score_trsym = manager.list()
score_hier = manager.list()
rank_trsym = manager2.list()
rank_hier = manager2.list()

t0 = time.time()

def test(tester, index, score_head, score_tail, score_trsym, score_hier, rank_head, rank_tail, rank_trsym, rank_hier):
    while index.value < len(tester.test_triples):
        id = index.value
        index.value += 1
        if id % 100 == 0:
            print("Tested %d in %d seconds." % (id+1, time.time()-t0))
        h, r, t = tester.test_triples[id]
        head_pool = tester.vec_c
        tail_pool = tester.vec_c
        head_vec = tester.vec_c[h]
        tail_vec = tester.vec_c[t]
        rel_vec = tester.rel_index2vec(r)
        # test_tail 
        # [[category, rank]*]
        rel_cat = tester.rel_cat_id(r)
        target_vec = head_vec + rel_vec
        this_rank_tail = np.array([rel_cat, tester.rank_index_from(target_vec, tail_pool, t, self_id = t)])
        this_score_tail = np.zeros(TopK + 1)
        this_score_tail[0] = rel_cat
        hit = False
        pred = tester.kNN(target_vec, tail_pool, topk = TopK, self_id=None)
        for i in range(len(pred)):
            if not hit and pred[i][0] in tester.hr_map[h][r]:
                hit = True
            if hit:
                this_score_tail[i + 1] = 1.
        rank_tail.append(this_rank_tail)
        score_tail.append(this_score_tail)
        if r in r_trsym:
            rank_trsym.append(this_rank_tail)
            score_trsym.append(this_score_tail)
        if r in r_hier:
            rank_hier.append(this_rank_tail)
            score_hier.append(this_score_tail)
        # test_head
        target_vec = tail_vec - rel_vec
        this_rank_head = np.array([rel_cat, tester.rank_index_from(target_vec, head_pool, h, self_id = h)])
        this_score_head = np.zeros(TopK + 1)
        this_score_head[0] = rel_cat
        hit = False
        pred = tester.kNN(target_vec, head_pool, topk = TopK, self_id=None)
        for i in range(len(pred)):
            if not hit and pred[i][0] in tester.tr_map[t][r]:
                hit = True
            if hit:
                this_score_head[i + 1] = 1.
        rank_head.append(this_rank_head)
        score_head.append(this_score_head)
        if r in r_trsym:
            rank_trsym.append(this_rank_head)
            score_trsym.append(this_score_head)
        if r in r_hier:
            rank_hier.append(this_rank_head)
            score_hier.append(this_score_head)
        
        

def test2(tester, index, score_rel, score_trsym, score_hier, rank_rel, rank_trsym, rank_hier):
    while index.value < len(tester.test_triples):
        id = index.value
        index.value += 1
        if id % 100 == 0:
            print("Tested %d in %d seconds." % (id+1, time.time()-t0))
        h, r, t = tester.test_triples[id]
        rel_pool = tester.vec_r
        head_vec = tester.vec_c[h]
        tail_vec = tester.vec_c[t]
        rel_vec = tester.rel_index2vec(r)
        # test_rel 
        # [[category, rank]*]
        rel_cat = tester.rel_cat_id(r)
        target_vec = tail_vec - head_vec
        this_rank_rel = np.array([rel_cat, tester.rank_index_from(target_vec, rel_pool, r, self_id = r)])
        this_score_rel = np.zeros(TopK + 1)
        this_score_rel[0] = rel_cat
        hit = False
        pred = tester.kNN(target_vec, rel_pool, topk = TopK, self_id=None)
        for i in range(len(pred)):
            if not hit and pred[i][0] in tester.ht_map[h][t]:
                hit = True
            if hit:
                this_score_rel[i + 1] = 1.
        rank_rel.append(this_rank_rel)
        score_rel.append(this_score_rel)
        if r in r_trsym:
            rank_trsym.append(this_rank_rel)
            score_trsym.append(this_score_rel)
        if r in r_hier:
            rank_hier.append(this_rank_rel)
            score_hier.append(this_score_rel)

# tester.rel_num_cases
print('Average norm=%d' % np.mean([LA.norm(vec, ord=(1 if tester.this_data.L1 else 2)) for vec in tester.vec_c]))
processes = [Process(target=test, args=(tester, index, score_head, score_tail, score_trsym, score_hier, rank_head, rank_tail, rank_trsym, rank_hier)) for x in range(cpu_count - 1)]
for p in processes:
    p.start()
for p in processes:
    p.join()

with open(result_file, 'w') as fp:
    fp.write('Tail\n')
    #tail
    mean_rank = np.zeros(4)
    hits = np.zeros((4, TopK))
    rank = 0.0
    hit = np.zeros(TopK)
    for s in rank_tail:
        mean_rank[int(s[0])] += s[1]
    for i in range(len(mean_rank)):
        if tester.rel_num_cases[i] == 0:
            mean_rank[i] = 0.0
        else:
            mean_rank[i] /= tester.rel_num_cases[i]
    fp.write('rank='+'\t'.join(str(x) for x in mean_rank) + '\n')
    for i in range(len(mean_rank)):
        rank += mean_rank[i] * tester.rel_num_cases[i]
    rank /= np.sum(tester.rel_num_cases)
    fp.write('over all='+str(rank) + '\n')
    for s in score_tail:
        hits[int(s[0])] += s[1:]
    for i in range(len(hits)):
        if tester.rel_num_cases[i] == 0:
            continue
        else:
            hits[i] /= tester.rel_num_cases[i]
    fp.write('hits11='+'\t'.join(str(x) for x in hits[0]) + '\n')
    fp.write('hits1n='+'\t'.join(str(x) for x in hits[1]) + '\n')
    fp.write('hitsn1='+'\t'.join(str(x) for x in hits[2]) + '\n')
    fp.write('hitsnn='+'\t'.join(str(x) for x in hits[3]) + '\n')
    for i in range(len(hits)):
        hit += hits[i] * tester.rel_num_cases[i]
    hit /= np.sum(tester.rel_num_cases)
    fp.write('overall='+'\t'.join(str(x) for x in hit) + '\n')
    #head
    fp.write('Head\n')
    mean_rank = np.zeros(4)
    hits = np.zeros((4, TopK))
    rank = 0.0
    hit = np.zeros(TopK)
    for s in rank_head:
        mean_rank[int(s[0])] += s[1]
    for i in range(len(mean_rank)):
        if tester.rel_num_cases[i] == 0:
            mean_rank[i] = 0.0
        else:
            mean_rank[i] /= tester.rel_num_cases[i]
    fp.write('rank='+'\t'.join(str(x) for x in mean_rank) + '\n')
    for i in range(len(mean_rank)):
        rank += mean_rank[i] * tester.rel_num_cases[i]
    rank /= np.sum(tester.rel_num_cases)
    fp.write('over all='+str(rank) + '\n')
    for s in score_head:
        hits[int(s[0])] += s[1:]
    for i in range(len(hits)):
        if tester.rel_num_cases[i] == 0:
            continue
        else:
            hits[i] /= tester.rel_num_cases[i]
    fp.write('hits11='+'\t'.join(str(x) for x in hits[0]) + '\n')
    fp.write('hits1n='+'\t'.join(str(x) for x in hits[1]) + '\n')
    fp.write('hitsn1='+'\t'.join(str(x) for x in hits[2]) + '\n')
    fp.write('hitsnn='+'\t'.join(str(x) for x in hits[3]) + '\n')
    for i in range(len(hits)):
        hit += hits[i] * tester.rel_num_cases[i]
    hit /= np.sum(tester.rel_num_cases)
    fp.write('overall='+'\t'.join(str(x) for x in hit) + '\n')
    #trsym
    fp.write('tr sym\n')
    rank = 0.0
    hit = np.zeros(TopK)
    for s in rank_trsym:
        rank += s
    rank /= len(rank_trsym)
    for s in score_trsym:
        hit += s[1:]
    hit /= len(score_trsym)
    fp.write('rank=' + str(rank) + '\n')
    fp.write('hits='+'\t'.join(str(x) for x in hit) + '\n')
    #hier
    fp.write('hier\n')
    rank = 0.0
    hit = np.zeros(TopK)
    for s in rank_hier:
        rank += s
    rank /= len(rank_hier)
    for s in score_hier:
        hit += s[1:]
    hit /= len(score_hier)
    fp.write('rank=' + str(rank) + '\n')
    fp.write('hits='+'\t'.join(str(x) for x in hit) + '\n')
    for s in rank_hier:
        rank += s
    rank /= len(rank_hier)


# tester.rel_num_cases
index = Value('i', 0, lock=True) #index
print('Average norm=%d' % np.mean([LA.norm(vec, ord=(1 if tester.this_data.L1 else 2)) for vec in tester.vec_c]))
score_trsym = manager.list()
score_hier = manager.list()
rank_trsym = manager2.list()
rank_hier = manager2.list()
processes = [Process(target=test2, args=(tester, index, score_rel, score_trsym, score_hier, rank_rel, rank_trsym, rank_hier)) for x in range(cpu_count - 1)]
for p in processes:
    p.start()
for p in processes:
    p.join()

with open(result_file2, 'w') as fp:
    fp.write('Rel\n')
    #tail
    mean_rank = np.zeros(4)
    hits = np.zeros((4, TopK))
    rank = 0.0
    hit = np.zeros(TopK)
    for s in rank_rel:
        mean_rank[int(s[0])] += s[1]
    for i in range(len(mean_rank)):
        if tester.rel_num_cases[i] == 0:
            mean_rank[i] = 0.0
        else:
            mean_rank[i] /= tester.rel_num_cases[i]
    fp.write('rank='+'\t'.join(str(x) for x in mean_rank) + '\n')
    for i in range(len(mean_rank)):
        rank += mean_rank[i] * tester.rel_num_cases[i]
    rank /= np.sum(tester.rel_num_cases)
    fp.write('over all='+str(rank) + '\n')
    for s in score_rel:
        hits[int(s[0])] += s[1:]
    for i in range(len(hits)):
        if tester.rel_num_cases[i] == 0:
            continue
        else:
            hits[i] /= tester.rel_num_cases[i]
    fp.write('rank11='+'\t'.join(str(x) for x in hits[0]) + '\n')
    fp.write('rank1n='+'\t'.join(str(x) for x in hits[1]) + '\n')
    fp.write('rankn1='+'\t'.join(str(x) for x in hits[2]) + '\n')
    fp.write('ranknn='+'\t'.join(str(x) for x in hits[3]) + '\n')
    for i in range(len(hits)):
        hit += hits[i] * tester.rel_num_cases[i]
    hit /= np.sum(tester.rel_num_cases)
    fp.write('overall='+'\t'.join(str(x) for x in hit) + '\n')
    fp.write('tr sym\n')
    rank = 0.0
    hit = np.zeros(TopK)
    for s in rank_trsym:
        rank += s
    rank /= len(rank_trsym)
    for s in score_trsym:
        hit += s[1:]
    hit /= len(score_trsym)
    fp.write('rank=' + str(rank) + '\n')
    fp.write('hits='+'\t'.join(str(x) for x in hit) + '\n')
    #hier
    fp.write('hier\n')
    rank = 0.0
    hit = np.zeros(TopK)
    for s in rank_hier:
        rank += s
    rank /= len(rank_hier)
    for s in score_hier:
        hit += s[1:]
    hit /= len(score_hier)
    fp.write('rank=' + str(rank) + '\n')
    fp.write('hits='+'\t'.join(str(x) for x in hit) + '\n')
    for s in rank_hier:
        rank += s
    rank /= len(rank_hier)
