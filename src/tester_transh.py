''' Module for held-out test.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from numpy import linalg as LA
import heapq as HP

import data  
import transh
import trainer_transh
from sklearn.neighbors import BallTree
from sklearn.neighbors import LSHForest
from sklearn.neighbors import KDTree
import multiprocessing
from multiprocessing import Manager

# This class is used to load and combine a TF_Parts and a Data object, and provides some useful methods for training
class Tester(object):
    def __init__(self):
        self.tf_parts = None
        self.this_data = None
        self.vec_c = np.array([0])
        self.vec_r = np.array([0])
        self.mat_h = np.array([0])
        self.mat_t = np.array([0])
        # below for test data
        self.test_triples = np.array([0])
        self.test_triples_group = {}
        # hr2t
        self.hr_map = {}
        # tr2h
        self.tr_map = {}
        self.ht_map = {}
        # relation categories
        self.rel11 = set([])
        self.rel1n = set([])
        self.reln1 = set([])
        self.relnn = set([])
        self.pre_pool = False
        # store num of test cases for each category
        self.rel_num_cases = np.zeros(4)
        self.manager = Manager()
    
    def build(self, save_path = 'this-model.ckpt', data_save_path = 'this-data.bin', pre_pool=False, use_balltree=False, use_lsh=False):
        self.this_data = data.Data()
        self.this_data.load(data_save_path)
        self.tf_parts = transh.TFParts(num_rels=self.this_data.num_rels(),
                         num_cons=self.this_data.num_cons(),
                         dim=self.this_data.dim,
                         batch_size=self.this_data.batch_size, L1=self.this_data.L1)
        sess = tf.Session()
        self.tf_parts._saver.restore(sess, save_path)  # load it
        value_ht, value_r, value_Mh = sess.run([self.tf_parts._ht, self.tf_parts._r, self.tf_parts._Mh])  # extract values.
        sess.close()
        self.vec_c = np.array(value_ht)
        self.vec_r = np.array(value_r)
        self.mat_t = self.mat_h = np.reshape(np.array(value_Mh), (-1, 1, self.this_data.dim))
        self.pj_pool = {}
        self.balltree = self.manager.dict()
        self.kdtree = self.manager.dict()
        #self.lsh = self.manager.dict()
        

        if pre_pool:
            self.pre_pool=True
            for r in range(len(self.vec_r)):
                self.pj_pool[r] = self.projection_pool(self.vec_c, r)
        '''
        self.balltree=None
        self.lsh=None
        if use_balltree:
            self.balltree={}
            for r in range(len(self.vec_r)):
                self.balltree[r] = BallTree(self.pj_pool[r], leaf_size=40)
            print("Built balltrees.")
        if use_lsh:
            self.lsh={}
            for r in range(len(self.vec_r)):
                self.lsh[r] = LSHForest(n_estimators=10, min_hash_match=3000, n_candidates=1500, n_neighbors=10, radius=1.0, radius_cutoff_ratio=0.9)
                self.lsh[r].fit(self.pj_pool[r])
            print("Built LSH.")
        '''
        # when a model doesn't have Mt, suppose it should pass Mh
    '''
    def rebuild_lsh(self, m_est, m_min_hash, m_cand, m_nn, m_radius, m_cutoff):
        if self.pre_pool == False:
            print("Precomputed pool is not allowed.")
        for r in range(len(self.vec_r)):
            self.lsh[r] = LSHForest(n_estimators=m_est, min_hash_match=m_min_hash, n_candidates=m_cand, n_neighbors=m_nn, radius=m_radius, radius_cutoff_ratio=m_cutoff)
            self.lsh[r].fit(self.pj_pool[r])
    '''
    def load_test_data(self, filename, splitter = '\t', line_end = '\n'):
        num_lines = 0
        triples = []
        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            if len(line) < 3:
                continue
            num_lines += 1
            h = self.this_data.con_str2index(line[0])
            r = self.this_data.rel_str2index(line[1])
            t = self.this_data.con_str2index(line[2])
            if h == None or r == None or t == None:
                continue
            triples.append([h, r, t])
            if self.hr_map.get(h) == None:
                self.hr_map[h] = {}
            if self.hr_map[h].get(r) == None:
                self.hr_map[h][r] = set([t])
            else:
                self.hr_map[h][r].add(t)
            if self.tr_map.get(t) == None:
                self.tr_map[t] = {}
            if self.tr_map[t].get(r) == None:
                self.tr_map[t][r] = set([h])
            else:
                self.tr_map[t][r].add(h)
            if self.ht_map.get(h) == None:
                self.ht_map[h] = {}
            if self.ht_map[h].get(t) == None:
                self.ht_map[h][t] = set([r])
            else:
                self.ht_map[h][t].add(r)
            # add to group
            if self.test_triples_group.get(r) == None:
                self.test_triples_group[r] = [(h, r, t)]
            else:
                self.test_triples_group[r].append((h, r, t))
        self.test_triples = np.array(triples)
        has_1n = np.zeros(self.this_data.num_rels())
        has_n1 = np.zeros(self.this_data.num_rels())
        for h, r, t in self.test_triples:
            if len(self.hr_map[h][r]) > 1:
                has_1n[r] = 1
            if len(self.tr_map[t][r]) > 1:
                has_n1[r] = 1
        for r in range(len(has_1n)):
            if has_1n[r] == 1 and has_n1[r] == 1:
                self.relnn.add(r)
            elif has_1n[r] == 1:
                self.rel1n.add(r)
            elif has_n1[r] == 1:
                self.reln1.add(r)
            else:
                self.rel11.add(r)
        for h, r, t in self.test_triples:
            self.rel_num_cases[self.rel_cat_id(r)] += 1
        self.test_triples = np.array(sorted(self.test_triples, key=lambda x: x[1]))
        print("Loaded test data from %s, %d out of %d." % (filename, len(triples), num_lines))
        print("Rel each cat:", self.rel_num_cases)
                
        
        
    
    # by default, return head_mat
    def get_mat(self, r, h_or_t='h'):
        if h_or_t == 't':
            return self.mat_t[r]
        else:
            return self.mat_h[r]
    
    def con_index2vec(self, c):
        return self.vec_c[c]

    def rel_index2vec(self, r):
        return self.vec_r[r]

    def con_str2vec(self, str):
        return self.vec_c[self.this_data.con_str2index(str)]
    
    def rel_str2vec(self, str):
        return self.vec_r[self.this_data.rel_str2index(str)]
    
    class index_dist:
        def __init__(self, index, dist):
            self.dist = dist
            self.index = index
            return
        def __lt__(self, other):
            return self.dist > other.dist
                
    def con_index2str(self, str):
        this_index = self.this_data.con_str2index(str)
        if this_index == None:
            return None
        return self.vec_c[this_index]
    
    def rel_index2str(self, str):
        this_index = self.this_data.rel_str2index(str)
        if this_index == None:
            return None
        return self.vec_r[this_index]
    
    # input must contain a pool of pre- or post-projected vecs. return a list of indices and dist
    def kNN(self, vec, vec_pool, topk=10, self_id=None):
        q = []
        for i in range(len(vec_pool)):
            #skip self
            if i == self_id:
                continue
            dist = LA.norm(vec - vec_pool[i], ord=(1 if self.this_data.L1 else 2))
            if len(q) < topk:
                HP.heappush(q, self.index_dist(i, dist))
            else:
                #indeed it fetches the biggest
                tmp = HP.nsmallest(1, q)[0]
                if tmp.dist > dist:
                    HP.heapreplace(q, self.index_dist(i, dist) )
        rst = []
        while len(q) > 0:
            item = HP.heappop(q)
            rst.insert(0, (item.index, item.dist))
        return rst
    
    #don't use when self.pre_pool is False
    def kNN_simple(self, vec, r, topk=10, self_id=None):
        q = []
        vec_pool=self.pj_pool[r]
        for i in range(len(vec_pool)):
            #skip self
            if i == self_id:
                continue
            dist = LA.norm(vec - vec_pool[i], ord=(1 if self.this_data.L1 else 2))
            if len(q) < topk:
                HP.heappush(q, self.index_dist(i, dist))
            else:
                #indeed it fetches the biggest
                tmp = HP.nsmallest(1, q)[0]
                if tmp.dist > dist:
                    HP.heapreplace(q, self.index_dist(i, dist) )
        rst = []
        while len(q) > 0:
            item = HP.heappop(q)
            rst.insert(0, item.index)
        return rst

    def build_pool(self, r):
        self.pj_pool[r] = np.dot(self.vec_c, self.mat_t[r])

    def build_balltree(self, r):
        if self.pj_pool.get(r) == None:
            self.build_pool(r)
        self.balltree[r] = BallTree(self.pj_pool[r], leaf_size=40)

    def build_kdtree(self, r):
        if self.pj_pool.get(r) == None:
            self.build_pool(r)
        self.kdtree[r] = KDTree(self.pj_pool[r], leaf_size=40)
    
    def del_pool(self, r):
        if self.pj_pool.get(r) != None:
            del self.pj_pool[r]

    def del_balltree(self, r):
        if self.balltree.get(r) != None:
            del self.balltree[r]

    def del_kdtree(self, r):
        if self.kdtree.get(r) != None:
            del self.kdtree[r]
        
    #don't use when self.pre_pool is False
    def kNN_balltree(self, vec, r, topk=10):
        if self.balltree.get(r) == None:
            self.build_balltree(r)
        return self.balltree[r].query([vec], k=topk, return_distance=False)[0]
    
    #don't use when self.pre_pool is False
    def kNN_kdtree(self, vec, r, topk=10):
        if self.kdtree.get(r) == None:
            self.build_kdtree(r)
        return self.kdtree[r].query([vec], k=topk, return_distance=False)[0]
    
    #don't use when self.pre_pool is False
    '''
    def kNN_lsh(self, vec, r, topk=10):
        return self.lsh[r].kneighbors([vec], n_neighbors=topk, return_distance=False)[0]
    '''
        
    # input must contain a pool of pre- or post-projected vecs. return a list of indices and dist. rank an index in a vec_pool from 
    def rank_index_from(self, vec, vec_pool, index, self_id = None):
        dist = LA.norm(vec - vec_pool[index], ord=(1 if self.this_data.L1 else 2))
        rank = 1
        for i in range(len(vec_pool)):
            if i == index or i == self_id:
                continue
            if dist > LA.norm(vec - vec_pool[i], ord=(1 if self.this_data.L1 else 2)):
                rank += 1
        return rank
    
    # 0~3
    def rel_cat_id(self, r):
        if r in self.relnn:
            return 3
        elif r in self.rel1n:
            return 1
        elif r in self.reln1:
            return 2
        else:
            return 0
    
    # We assume there's an operator in each model, which is defined using a concept, a relation (id),  and a marker for head-or-tail. Later, if we change it to a simpler model such as TransR, we keep the interface of this operator, but change its behavior. This is useful for evaluation regardless of the model definition.
    def projection(self, c, r, h_or_t='h'):
        if self.pre_pool:
            return self.pj_pool[r][c]
        mat = None
        vec_c = self.vec_c[c]
        if h_or_t == 't':
            mat = self.mat_t[r]
        else:
            mat = self.mat_h[r]
        return vec_c - np.dot(mat, np.dot(vec_c.reshape((self.this_data.dim, 1)), mat.reshape((1, self.this_data.dim))))
    '''
    def projection_pool(self, ht, r, h_or_t='h'):
        mat = None
        if h_or_t == 't':
            mat = self.mat_t[r]
        else:
            mat = self.mat_h[r]
        return np.dot(ht, mat)
    '''
    def projection_pool(self, ht, r, h_or_t='h'):
        if self.pre_pool and self.pj_pool.get(r) != None:
            return self.pj_pool[r]
        rst = np.zeros(ht.shape)
        for i in range(len(rst)):
            vec_c = ht[i]
            mat = self.mat_h[r]
            rst[i] = vec_c - np.dot(mat, np.dot(vec_c.reshape((self.this_data.dim, 1)), mat.reshape((1, self.this_data.dim))))
        return rst

    def dissimilarity(self, h, r, t):
        h_vec = None
        t_vec = None
        if self.pre_pool:
            h_vec = self.pj_pool[r][h]
            t_vec = self.pj_pool[r][t]
        else:
            h_vec = self.projection(h, r)
            t_vec = self.projection(t, r)
        return LA.norm(h_vec + self.vec_r[r] - t_vec, ord=(1 if self.this_data.L1 else 2))

    def dissimilarity_r(self, h, rel, t, r):
        return LA.norm(self.projection(h, r, 'h') + self.vec_r[rel] - self.projection(t, r, 't'), ord=(1 if self.this_data.L1 else 2))