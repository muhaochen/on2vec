''' Module for training TF parts.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time

import data  
import model


class Trainer(object):
    def __init__(self):
        self.batch_size=128
        self.dim=64
        self.this_data = None
        self.tf_parts = None
        self.save_path = 'this-model.ckpt'
        self.data_save_path = 'this-data.bin'
        self.L1=False

    def build(self, data, dim=64, batch_size=128, save_path = 'this-model.ckpt', data_save_path = 'this-data.bin', L1=False):
        self.this_data = data
        self.dim = self.this_data.dim = dim
        self.batch_size = self.this_data.batch_size = batch_size
        self.data_save_path = data_save_path
        self.save_path = save_path
        self.L1 = self.this_data.L1 = L1
        self.tf_parts = model.TFParts(num_rels=self.this_data.num_rels(),
                                 num_cons=self.this_data.num_cons(),
                                 dim=dim,
                                 batch_size=self.batch_size,
                                 L1=self.L1)

    def gen_A_batch(self, forever=False, shuffle=True):
        l = self.this_data.triples.shape[0]
        while True:
            triples = self.this_data.triples
            if shuffle:
                np.random.seed(int(time.time()))
                np.random.shuffle(triples)
            for i in range(0, l, self.batch_size):
                batch = triples[i: i+self.batch_size, :]
                if batch.shape[0] < self.batch_size:
                    batch = np.concatenate((batch, self.this_data.triples[:self.batch_size - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == self.batch_size
                neg_batch = self.this_data.corrupt_batch(batch)
                h_batch, r_batch, t_batch = batch[:, 0], batch[:, 1], batch[:, 2]
                neg_h_batch, neg_t_batch = neg_batch[:, 0], neg_batch[:, 2]
                yield h_batch.astype(np.int64), r_batch.astype(np.int64), t_batch.astype(np.int64), neg_h_batch.astype(np.int64), neg_t_batch.astype(np.int64)
            if not forever:
                break
        
        
    def gen_B_single(self, variant='t', forever=False, shuffle=True):
        assert variant in ('t', 'h')
        if variant == 't':
            c_func, r_func = self.this_data.con_ref, self.this_data.rel_ref
        elif variant == 'h':
            c_func, r_func = self.this_data.con_coe, self.this_data.rel_coe
        while True:
            c_set = c_func()
            if shuffle:
                np.random.seed(int(time.time()))
                np.random.shuffle(c_set)
            for c in c_set:
                for r in r_func():
                    index_range = self.this_data.sigma(c, r, many=variant)
                    for i in index_range:
                        if variant == 't':
                            neg_samp = self.this_data.corrupt([c, r, i], tar=variant)
                            yield c, i, r, neg_samp[0], neg_samp[2]
                        else:
                            neg_samp = self.this_data.corrupt([i, r, c], tar=variant)
                            yield i, c, r, neg_samp[0], neg_samp[2]
            if not forever:
                break

    def gen_B_batch(self, variant, forever=False, shuffle=True):
        o1_batch, o2_batch, r_batch, n1_batch, n2_batch = [], [], [], [], []
        for o1, o2, r, n1, n2  in self.gen_B_single(variant, forever, shuffle):
            o1_batch.append(o1)
            o2_batch.append(o2)
            r_batch.append(r)
            n1_batch.append(n1)
            n2_batch.append(n2)
            if len(o1_batch) == self.batch_size:
                yield np.array(o1_batch, dtype=np.int64), np.array(o2_batch, dtype=np.int64), np.array(r_batch, dtype=np.int64), np.array(n1_batch, dtype=np.int64), np.array(n2_batch, dtype=np.int64)
                o1_batch, o2_batch, r_batch, n1_batch, n2_batch = [], [], [], [], []

    '''
    def gen_C_batch(self.this_data, self.batch_size, forever=False):
        num_cons=self.this_data.num_cons()
        start = 0
        while True:
            yield np.array([_ % num_cons for _ in range(start, start + self.batch_size)], dtype=np.int64)
            start += self.batch_size
            if not forever and start >= num_cons:
                break
    '''

    def feed_ht(self, sess, cvec="entity2vec.bern", cmap="entity2id.txt", splitter='\t', line_end='\n'):
        new_ht = np.zeros([self.this_data.num_cons(), self.dim])
        id2con = {}
        for line in open(cmap):
            line = line.rstrip(line_end).split(splitter)
            if len(line) != 2:
                continue
            id2con[int(line[1])] = line[0]
        id = 0
        for line in open(cvec):
            line = line.rstrip(line_end).rstrip(splitter).split(splitter)
            if id == 0:
                assert(len(line) == self.dim)
            if len(line) != self.dim:
                continue
            line = np.array([float(x) for x in line])
            new_ht[self.this_data.con_str2index(id2con[id])] = line
            id += 1
        sess.run(self.tf_parts._assign_ht_op, feed_dict={self.tf_parts._ht_assign: new_ht})
        print("Overloaded ht of size: %d" % len(new_ht))

    def feed_r(self, sess, rvec="relation2vec.bern", rmap="relation2id.txt", splitter='\t', line_end='\n'):
        new_r = np.zeros([self.this_data.num_rels(), self.dim])
        id2rel = {}
        for line in open(rmap):
            line = line.rstrip(line_end).split(splitter)
            if len(line) != 2:
                continue
            id2rel[int(line[1])] = line[0]
        id = 0
        for line in open(rvec):
            line = line.rstrip(line_end).rstrip(splitter).split(splitter)
            if id == 0:
                assert(len(line) == self.dim)
            if len(line) != self.dim:
                continue
            line = np.array([float(x) for x in line])
            new_r[self.this_data.rel_str2index(id2rel[id])] = line
            id += 1
        sess.run(self.tf_parts._assign_r_op, feed_dict={self.tf_parts._r_assign: new_r})
        print("Overloaded r of size: %d" % len(new_r))

    def feed_m(self, sess, mvec="A.bern", rmap="relation2id.txt", splitter='\t', line_end='\n'):
        new_m = np.zeros([self.this_data.num_rels(), self.dim * self.dim])
        id2rel = {}
        for line in open(rmap):
            line = line.rstrip(line_end).split(splitter)
            if len(line) != 2:
                continue
            id2rel[int(line[1])] = line[0]
        mats = []
        for line in open(mvec):
            line = line.rstrip(line_end).rstrip(splitter).split(splitter)
            if len(line) != self.dim:
                continue
            line = np.array([float(x) for x in line])
            mats.append(line)
        mats = np.array(mats).reshape([-1, self.dim * self.dim])
        assert(len(mats) == self.this_data.num_rels())
        id = 0
        for m in mats:
            new_m[self.this_data.rel_str2index(id2rel[id])] = m
            id += 1
        sess.run(self.tf_parts._assign_m_op, feed_dict={self.tf_parts._m_assign: new_m})
        print("Overloaded mt, mh of size: %d" % len(new_m))

    def train(self, epochs=20, save_every_epoch=10, lr=0.001, a1=0.1, a2=0.05, m1=0.5, m2=0.5, balance_alpha1=False, 
              feed_pre=False, cvec="entity2vec.bern", cmap="entity2id.txt", rvec="relation2vec.bern", rmap="relation2id.txt", mvec="A.bern", splitter='\t', line_end='\n'):
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        if feed_pre:
            self.feed_ht(sess, cvec, cmap, splitter, line_end)
            self.feed_r(sess, rvec, rmap, splitter, line_end)
            self.feed_m(sess, mvec, rmap, splitter, line_end)
        
        num_A_batch = len(list(self.gen_A_batch()))
        num_B_t_batch = len(list(self.gen_B_batch('t')))
        num_B_h_batch = len(list(self.gen_B_batch('h')))
        #num_C_batch = len(list(gen_C_batch(self.this_data, self.batch_size)))
        total_batches = max(num_A_batch, num_B_t_batch, num_B_h_batch)
        print('self.batch_size=', self.batch_size)
        print('num_A_batch =', num_A_batch)
        print('num_B_t_batch =', num_B_t_batch)
        print('num_B_h_batch =', num_B_h_batch)
        #print('num_C_batch =', num_C_batch)
        print('total_batches =', total_batches)
        
        # margins
        self.tf_parts._m1 = m1
        self.tf_parts._m2 = m2    
        t0 = time.time()
        for epoch in range(epochs):
            epoch_loss = self.train1epoch(sess, num_A_batch, num_B_t_batch, num_B_h_batch, total_batches, lr, a1, a2, m1, m2, balance_alpha1, epoch + 1)
            print("Time use: %d" % (time.time() - t0))
            if np.isnan(epoch_loss):
                print("Training collapsed.")
                return
            if (epoch + 1) % save_every_epoch == 0:
                this_save_path = self.tf_parts._saver.save(sess, self.save_path)
                self.this_data.save(self.data_save_path)
                print("Model saved in file: %s. Data saved in file: %s" % (this_save_path, self.data_save_path))
        this_save_path = self.tf_parts._saver.save(sess, self.save_path)
        self.this_data.save(self.data_save_path)
        print("Model saved in file: %s. Data saved in file: %s" % (this_save_path, self.data_save_path))
        sess.close()
        print("Done")

    def train1epoch(self, sess, num_A_batch, num_B_t_batch, num_B_h_batch, total_batches, lr, a1, a2, m1, m2, balance_alpha1, epoch):
        '''build and train a model.

        Args:
            self.batch_size: size of batch
            num_epoch: number of epoch. A epoch means a turn when all A/B_t/B_h/C are passed at least once.
            dim: dimension of embedding
            lr: learning rate
            self.this_data: a Data object holding data.
            save_every_epoch: save every this number of epochs.
            save_path: filepath to save the tensorflow model.
        '''

        this_gen_A_batch = self.gen_A_batch(forever=True)
        this_gen_B_t_batch = self.gen_B_batch(variant='t', forever=True)
        this_gen_B_h_batch = self.gen_B_batch(variant='h', forever=True)
        
        this_loss = []
        
        loss_A = loss_B1 = loss_B2 = loss_C_A = loss_C_B1 = loss_C_B2 = 0
        
        a11 = a1
        a12 = a1
        if balance_alpha1:
            a11 = a1 * 2 * num_B_h_batch / (num_B_t_batch + num_B_h_batch)
            a12 = a1 * 2 * num_B_t_batch / (num_B_t_batch + num_B_h_batch)

        for batch_id in range(total_batches):
            # Optimize loss A
            # Optimize loss A
            if batch_id < num_A_batch:
                A_h_index, A_r_index, A_t_index, A_hn_index, A_tn_index  = next(this_gen_A_batch)
                _, loss_A = sess.run([self.tf_parts._train_op_A, self.tf_parts._A_loss],
                        feed_dict={self.tf_parts._A_h_index: A_h_index, 
                                   self.tf_parts._A_r_index: A_r_index,
                                   self.tf_parts._A_t_index: A_t_index,
                                   self.tf_parts._A_hn_index: A_hn_index, 
                                   self.tf_parts._A_tn_index: A_tn_index,
                                   self.tf_parts._lr: lr})
                _, loss_C_A = sess.run([self.tf_parts._train_op_C_A, self.tf_parts._C_loss_A],
                        feed_dict={self.tf_parts._A_h_index: A_h_index, 
                                   self.tf_parts._A_r_index: A_r_index,
                                   self.tf_parts._A_t_index: A_t_index,
                                   self.tf_parts._A_hn_index: A_hn_index, 
                                   self.tf_parts._A_tn_index: A_tn_index,
                                   #self.tf_parts._B_h_index: B_h_index, 
                                   #self.tf_parts._B_t_index: B_t_index,
                                   #self.tf_parts._B_hn_index: B_hn_index, 
                                   #self.tf_parts._B_tn_index: B_tn_index,
                                   #self.tf_parts._B_r_index: B_r_index,
                                   self.tf_parts._lr: lr * a2})
            
            # Optimize loss B_t
            if batch_id < num_B_t_batch:
                B_h_index, B_t_index, B_r_index, B_hn_index, B_tn_index  = next(this_gen_B_t_batch)
                _, loss_B1 = sess.run([self.tf_parts._train_op_B_t, self.tf_parts._B_t_loss],
                        feed_dict={self.tf_parts._B_h_index: B_h_index, 
                                   self.tf_parts._B_t_index: B_t_index,
                                   self.tf_parts._B_r_index: B_r_index,
                                   self.tf_parts._B_hn_index: B_hn_index, 
                                   self.tf_parts._B_tn_index: B_tn_index,
                                   self.tf_parts._lr: lr * a11})
                _, loss_C_B1 = sess.run([self.tf_parts._train_op_C_B1, self.tf_parts._C_loss_B1],
                        feed_dict={self.tf_parts._B_h_index: B_h_index, 
                                   self.tf_parts._B_t_index: B_t_index,
                                   self.tf_parts._B_hn_index: B_hn_index, 
                                   self.tf_parts._B_tn_index: B_tn_index,
                                   self.tf_parts._B_r_index: B_r_index,
                                   self.tf_parts._lr: lr * a2})

            # Optimize loss B_h
            if batch_id < num_B_h_batch:
                B_h_index, B_t_index, B_r_index, B_hn_index, B_tn_index = next(this_gen_B_h_batch)
                _, loss_B2 = sess.run([self.tf_parts._train_op_B_h, self.tf_parts._B_h_loss],
                        feed_dict={self.tf_parts._B_h_index: B_h_index, 
                                   self.tf_parts._B_t_index: B_t_index,
                                   self.tf_parts._B_r_index: B_r_index,
                                   self.tf_parts._B_hn_index: B_hn_index, 
                                   self.tf_parts._B_tn_index: B_tn_index,
                                   self.tf_parts._lr: lr * a12})
                _, loss_C_B2 = sess.run([self.tf_parts._train_op_C_B2, self.tf_parts._C_loss_B2],
                        feed_dict={self.tf_parts._B_h_index: B_h_index, 
                                   self.tf_parts._B_t_index: B_t_index,
                                   self.tf_parts._B_hn_index: B_hn_index, 
                                   self.tf_parts._B_tn_index: B_tn_index,
                                   self.tf_parts._B_r_index: B_r_index,
                                   self.tf_parts._lr: lr * a2})
            
            # Optimize loss C
            #   generate fake data
            #C_h_index = next(this_gen_C_h_batch)
            '''
            if batch_id < num_A_batch:
                _, loss_C = sess.run([self.tf_parts._train_op_C, self.tf_parts._C_loss],
                        feed_dict={self.tf_parts._A_h_index: A_h_index, 
                                   self.tf_parts._A_r_index: A_r_index,
                                   self.tf_parts._A_t_index: A_t_index,
                                   self.tf_parts._A_hn_index: A_hn_index, 
                                   self.tf_parts._A_tn_index: A_tn_index,
                                   #self.tf_parts._B_h_index: B_h_index, 
                                   #self.tf_parts._B_t_index: B_t_index,
                                   #self.tf_parts._B_hn_index: B_hn_index, 
                                   #self.tf_parts._B_tn_index: B_tn_index,
                                   #self.tf_parts._B_r_index: B_r_index,
                                   self.tf_parts._lr: lr * a2})
            '''
            
            # Observe total loss
            batch_loss = [loss_A, loss_B1, loss_B2, loss_C_A, loss_C_B1, loss_C_B2]
            if len(this_loss) == 0:
                this_loss = np.array(batch_loss)
            else:
                this_loss += np.array(batch_loss)
            
            if ((batch_id + 1) % 100 == 0) or batch_id == total_batches - 1:
                print('process: %d / %d. Epoch %d' % (batch_id+1, total_batches, epoch))
        '''
        value_ht, value_r, value_Mh, value_Mt = sess.run([tf_parts._ht, tf_parts._r, tf_parts._Mh, tf_parts._Mt])  # extract values.
        print('sample of value_ht:', value_ht)
        print('sample of value_r:', value_r)
        print('sample of value_Mh:', value_Mh)
        print('sample of value_Mt:', value_Mt)
        '''
        this_total_loss = np.sum(this_loss)
        print("Loss of epoch %d = %s" % (epoch, np.sum(this_total_loss)))
        print([l for l in this_loss])
        return this_total_loss

# A safer loading is available in Tester, with parameters like batch_size and dim recorded in the corresponding Data component
def load_tfparts(batch_size = 128,
                dim = 64,
                this_data=None,
                save_path = 'this-model.ckpt',
                L1=False):
    tf_parts = model.TFParts(num_rels=this_data.num_rels(),
                             num_cons=this_data.num_cons(),
                             dim=dim,
                             batch_size=self.batch_size,
                            L1=L1)
    with tf.Session() as sess:
        tf_parts._saver.restore(sess, save_path)