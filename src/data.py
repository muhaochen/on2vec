"""Processing of data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pickle
import time

class Data(object):
    '''The abustrct class that defines interfaces for holding all data.
    '''

    def __init__(self):
        # concept vocab
        self.cons = []
        # rel vocab
        self.rels = []
        # transitive rels vocab
        self.rels_tran = []
        # refinement rels vocab
        self.rels_ref = []
        # coercion rels vocab
        self.rels_coe = []
        # invertible rels vocab (for reference use)
        self.rels_inv = []
        self.index_cons = {}
        self.index_rels = {}
        # end-point concept indices for refinement and coercion relations
        self.index_ref_con = np.array([0])
        self.index_coe_con = np.array([0])
        # save triples as array of indices
        self.triples = np.array([0])
        self.triples_record = set([])
        # map for sigma
        self.index_sigma_ref = {}
        self.index_sigma_coe = {}
        # map for tau
        self.index_tau = {}
        self.index_tau_revert = {}
        self.max_tran_sets = {}
        self.max_tran_sets_rev = {}
        # head per tail and tail per head (for each relation). used for bernoulli negative sampling
        self.hpt = np.array([0])
        self.tph = np.array([0])
        # recorded for tf_parts
        self.dim = 64
        self.batch_size = 1024
        self.L1=False

    def load_data(self, filename, rel_tran=[], rel_ref=[], rel_coe=[], rel_inv=[], splitter = '\t', line_end = '\n', identify_trans=False):
        '''Load the dataset.'''
        triples = []
        tran_triples = []
        last_c = -1
        last_r = -1
        self.rels_tran = set(rel_tran)
        self.rels_ref = set(rel_ref)
        self.rels_coe = set(rel_coe)
        self.index_ref_con = set([])
        self.index_coe_con = set([])
        hr_map = {}
        tr_map = {}
        for line in open(filename):
            line = line.rstrip(line_end).split('\t')
            if self.index_cons.get(line[0]) == None:
                self.cons.append(line[0])
                last_c += 1
                self.index_cons[line[0]] = last_c
            if self.index_cons.get(line[2]) == None:
                self.cons.append(line[2])
                last_c += 1
                self.index_cons[line[2]] = last_c
            if self.index_rels.get(line[1]) == None:
                self.rels.append(line[1])
                last_r += 1
                self.index_rels[line[1]] = last_r
            h = self.index_cons[line[0]]
            r = self.index_rels[line[1]]
            t = self.index_cons[line[2]]
            triples.append([h, r, t])
            self.triples_record.add((h, r, t))
            '''
            if line[1] in self.rels_tran:
                #tran_triples.append([h, r, t])
                if self.index_tau.get(h) == None:
                    self.index_tau[h] = {}
                if self.index_tau[h].get(r) == None:
                    self.index_tau[h][r] = set([t])
                else:
                    self.index_tau[h][r].add(t)
            '''
            if identify_trans:
                if hr_map.get(h) == None:
                    hr_map = {}
                if hr_map[h].get(r) == None:
                    hr_map[h][r] = set([t])
                else:
                    hr_map[h][r].add(t)
                if tr_map.get(t) == None:
                    tr_map = {}
                if tr_map[t].get(r) == None:
                    tr_map[t][r] = set([h])
                else:
                    tr_map[t][r].add(h)
            if line[1] in self.rels_ref:
                self.index_ref_con.add(h)
                if self.index_sigma_ref.get(h) == None:
                    self.index_sigma_ref[h] = {}
                if self.index_sigma_ref[h].get(r) == None:
                    self.index_sigma_ref[h][r] = set([t])
                else:
                    self.index_sigma_ref[h][r].add(t)
            if line[1] in self.rels_coe:
                self.index_coe_con.add(t)
                if self.index_sigma_coe.get(t) == None:
                    self.index_sigma_coe[t] = {}
                if self.index_sigma_coe[t].get(r) == None:
                    self.index_sigma_coe[t][r] = set([h])
                else:
                    self.index_sigma_coe[t][r].add(h)
        self.triples = np.array(triples)
        # calculate tph and hpt
        tph_array = np.zeros((len(self.rels), len(self.cons)))
        hpt_array = np.zeros((len(self.rels), len(self.cons)))
        if identify_trans:
            self.rels_tran = set([])
        for h,r,t in self.triples:
            tph_array[r][h] += 1.
            hpt_array[r][t] += 1.
            if identify_trans:
                if hr_map.get(t) != None and hr_map[t].get(r) != None and hr_map[h][r] >= hr_map[t][r]:
                    self.rels_tran.add(r)
                    if self.index_tau.get(h) == None:
                        self.index_tau[h] = {}
                    temp_set = self.index_tau[h] = hr_map[h]
                    if self.max_tran_sets.get(r) == None:
                        self.max_tran_sets[r] = [temp_set]
                    else:
                        within = False
                        for s in self.max_tran_sets[r]:
                            if temp_set >= s:
                                self.max_tran_sets[r].remove(s)
                            elif temp_set <= s:
                                within = True
                                break
                        if within == False:
                            self.max_tran_sets[r].append(temp_set)
                if tr_map.get(h) != None and tr_map[h].get(r) != None and tr_map[t][r] >= tr_map[h][r]:
                    if self.index_tau_revert.get(t) == None:
                        self.index_tau_revert[t] = {}
                    temp_set = self.index_tau_revert[t] = tr_map[t]
                    if self.max_tran_sets_rev.get[r] == None:
                        self.max_tran_sets[r] = [temp_set]
                    else:
                        within = False
                        for s in self.max_tran_sets_rev[r]:
                            if temp_set >= s:
                                self.max_tran_sets_rev[r].remove(s)
                            elif temp_set <= s:
                                within = True
                                break
                            if within == False:
                                self.max_tran_sets_rev[r].append(temp_set)
        self.tph = np.mean(tph_array, axis = 1)
        self.hpt = np.mean(hpt_array, axis = 1)
        self.index_rel_ref = np.array([self.index_rels[r] for r in self.rels_ref])
        self.index_rel_coe = np.array([self.index_rels[r] for r in self.rels_coe])
        self.index_rel_tran = np.array([self.index_rels[r] for r in self.rels_tran])
        self.index_ref_con = np.array([r for r in self.index_ref_con])
        self.index_coe_con = np.array([r for r in self.index_coe_con])
    
    # add more triples to self.triples_record to 'filt' negative sampling
    def record_more_data(self, filename, splitter = '\t', line_end = '\n'):
        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            if len(line) < 3:
                continue
            h = self.con_str2index(line[0])
            r = self.rel_str2index(line[1])
            t = self.con_str2index(line[2])
            if h != None and r != None and t != None:
                self.triples_record.add((h, r, t))
        print("Loaded %s to triples_record." % (filename))
    
    def num_cons(self):
        '''Returns number of ontologies. 

        This means all ontologies have index that 0 <= index < num_onto().
        '''
        return len(self.cons)

    def num_rels(self):
        '''Returns number of relations.

        This means all relations have index that 0 <= index < num_rels().
        Note that we consider *ALL* relations, e.g. $R_O$, $R_h$ and $R_{tr}$.
        '''
        return len(self.rels)

    def rel_str2index(self, rel_str):
        '''For relation `rel_str` in string, returns its index.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.index_rels.get(rel_str)

    def rel_index2str(self, rel_index):
        '''For relation `rel_index` in int, returns its string.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.rels[rel_index]

    def con_str2index(self, con_str):
        '''For ontology `con_str` in string, returns its index.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.index_cons.get(con_str)

    def con_index2str(self, con_index):
        '''For ontology `con_index` in int, returns its string.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.cons[con_index]

    def con_ref(self):
        return self.index_ref_con

    def con_coe(self):
        return self.index_coe_con

    def rel(self):
        return np.array(range(self.num_rels()))

    def rel_ref(self):
        '''Returns a set of indices of relations that is in $R_h$.'''
        return self.index_rel_ref

    def rel_coe(self):
        '''Returns a set of indices of relations that is in $R_h$.'''
        return self.index_rel_coe

    def rel_tr(self):
        '''Returns a set of indices of relations that is in $R_{tr}$.'''
        return self.index_rel_tran
    
    def rel_inv(self):
        '''Returns a set of indices of relations that is invertible.'''
        return np.array([self.index_rels[r] for r in self.rels_inv])

    #transitive hopping ends
    def tau(self, c_index, r_index, start='h'):
        '''Returns set of indices for function $\tau$, given relation and end-point concept. Return [] if meets definition, but function range empty. Return None if violates definition.'''
        assert(start in ['h', 't'])
        if r_index in self.rel_tr():
            if start == 'h':
                dict = self.index_tau.get(c_index)
                if dict != None:
                    res = dict.get(r_index)
                    if res == None:
                        res = []
                    return res
            else:
                dict = self.index_tau_revert.get(c_index)
                if dict != None:
                    res = dict.get(r_index)
                    if res == None:
                        res = []
                    return res
        return None

    def tau_str(self, c, r):
        return self.tau(self.con_str2index(c), self.rel_str2index(r))
        
    #Offsprings/Preds
    def sigma(self, c_index, r_index, many='h'):
        '''Returns set of indices for function $\sigma$, given relation and end-point concept. Will self-test whether r is refinement or coercion. Return [] if meets definition, but function range empty. Return None if violates definition.'''
        assert(many in ['h', 't'])
        if many=='t' and r_index in self.rel_ref():
            dict = self.index_sigma_ref.get(c_index)
            if dict != None:
                res = dict.get(r_index)
                if res == None:
                    res = []
                return res
        elif many=='h' and r_index in self.rel_coe():
            dict = self.index_sigma_coe.get(c_index)
            if dict != None:
                res = dict.get(r_index)
                if res == None:
                    res = []
                return res
        return None

    def sigma_str(self, c, r, many='r'):
        return self.sigma(self.con_str2index(c), self.rel_str2index(r), many)
    
    def corrupt_pos(self, t, pos):
        hit = True
        res = None
        while hit:
            res = np.copy(t)
            samp = np.random.randint(self.num_cons())
            while samp == t[pos]:
                samp = np.random.randint(self.num_cons())
            res[pos] = samp
            if tuple(res) not in self.triples_record:
                hit = False
        return res
            
        
    #bernoulli negative sampling
    def corrupt(self, t, tar = None):
        if tar == 't':
            return self.corrupt_pos(t, 2)
        elif tar == 'h':
            return self.corrupt_pos(t, 0)
        else:
            this_tph = self.tph[t[1]]
            this_hpt = self.hpt[t[1]]
            assert(this_tph > 0 and this_hpt > 0)
            np.random.seed(int(time.time()))
            if np.random.uniform(high=this_tph + this_hpt, low=0.) < this_hpt:
                return self.corrupt_pos(t, 2)
            else:
                return self.corrupt_pos(t, 0)
    
    #bernoulli negative sampling on a batch
    def corrupt_batch(self, t_batch, tar = None):
        return np.array([self.corrupt(t, tar) for t in t_batch])
        

    def save(self, filename):
        f = open(filename,'wb')
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print("Save data object as", filename)
    def load(self, filename):
        f = open(filename,'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print("Loaded data object from", filename)