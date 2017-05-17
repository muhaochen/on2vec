'''Model for holding TF parts. etc.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import data as pymod_data
from data import Data
import pickle

# Orthogonal Initializer from
# https://github.com/OlavHN/bnlstm
def orthogonal(shape):
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  return q.reshape(shape)

def orthogonal_initializer(scale=1.0, dtype=tf.float32):
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    return tf.constant(orthogonal(shape) * scale, dtype)
  return _initializer

class TFParts(object):
    '''TensorFlow-related things. 
    
    This is to keep TensorFlow-related components in a neat shell.
    '''

    def __init__(self, num_rels, num_cons, dim, batch_size):
        self._num_rels = num_rels
        self._num_cons = num_cons
        self._dim = dim  # dimension of both relation and ontology. 
        self._batch_size = batch_size
        self._epoch_loss = 0
        self.build()

    @property
    def num_cons(self):
        return self._num_cons

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):
        return self._batch_size    

    def build(self):
        tf.reset_default_graph()

        with tf.variable_scope("graph", initializer=orthogonal_initializer()):
            # Variables (matrix of embeddings/transformations)

            self._ht = ht = tf.get_variable(
                name='ht',  # for t AND h
                shape=[self.num_cons, self.dim],
                dtype=tf.float32)
            self._r = r = tf.get_variable(
                name='r',
                shape=[self.num_rels, self.dim],
                dtype=tf.float32)
            self._Mh = Mh = tf.get_variable(
                name='Mh', 
                shape=[self.num_rels, self.dim * self.dim],
                dtype=tf.float32,)
            self._Mt = Mt = tf.get_variable(
                name='Mt', 
                shape=[self.num_rels, self.dim * self.dim],
                dtype=tf.float32)

            # Type A loss : || M_h h + r - M_t t ||_2
            self._A_h_index = A_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='A_h_index')
            self._A_r_index = A_r_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='A_r_index')
            self._A_t_index = A_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='A_t_index')
            '''
            A_loss_matrix = tf.sub(
                tf.add(
                    tf.batch_matmul(tf.nn.embedding_lookup(ht, A_h_index), tf.reshape(tf.nn.embedding_lookup(Mh, A_r_index), [-1, self.dim, self.dim])),
                    tf.nn.embedding_lookup(r, A_r_index)),
                tf.batch_matmul(tf.nn.embedding_lookup(ht, A_t_index), tf.reshape(tf.nn.embedding_lookup(Mt, A_r_index), [-1, self.dim, self.dim]))
            )'''
            
            A_h_batch_mul = tf.squeeze(tf.batch_matmul(tf.expand_dims(tf.nn.embedding_lookup(ht, A_h_index), 1), tf.reshape(tf.nn.embedding_lookup(Mh, A_r_index), [-1, self.dim, self.dim])), [1])
            A_t_batch_mul = tf.squeeze(tf.batch_matmul(tf.expand_dims(tf.nn.embedding_lookup(ht, A_t_index), 1), tf.reshape(tf.nn.embedding_lookup(Mt, A_r_index), [-1, self.dim, self.dim])), [1])
            
            A_loss_matrix = tf.sub(
                tf.add(A_h_batch_mul, tf.nn.embedding_lookup(r, A_r_index)),
                A_t_batch_mul
            )
            # L-2 norm
            self._A_loss = A_loss = tf.reduce_sum(
                tf.sqrt(tf.reduce_sum(tf.square(A_loss_matrix), 1))) 

            A_vec_restraint = tf.concat(0, [tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(tf.nn.embedding_lookup(ht, A_h_index)), 1)), 1.), 0.), tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(tf.nn.embedding_lookup(ht, A_t_index)), 1)), 1.), 0.)])
            A_proj_restraint = tf.concat(0, [tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(A_h_batch_mul), 1)), 1.), 0.), tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(A_t_batch_mul), 1)), 1.), 0.)])
            A_rel_restraint = tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(tf.nn.embedding_lookup(ht, A_r_index)), 1)), 1.), 0.)

            # Type B loss : 
            # 2 losses: t-related <- omega(M_t o1, M_t o2) and h-related <- omega(M_h o1, M_h o2)
            # Let's use || a M_hr + r - b M_tr ||_2 as omega(a,b)
            # They share the same input place holders
            self._B_o1_index = B_o1_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='B_o1_index')
            self._B_r_index = B_r_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='B_r_index')
            self._B_o2_index = B_o2_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='B_o2_index')
            B_t_batch_mul_o1 = tf.squeeze(tf.batch_matmul(tf.expand_dims(tf.nn.embedding_lookup(ht, B_o1_index), 1), tf.reshape(tf.nn.embedding_lookup(Mt, B_r_index), [-1, self.dim, self.dim])), [1])
            B_t_batch_mul_o2 = tf.squeeze(tf.batch_matmul(tf.expand_dims(tf.nn.embedding_lookup(ht, B_o2_index), 1), tf.reshape(tf.nn.embedding_lookup(Mt, B_r_index), [-1, self.dim, self.dim])), [1])
            B_t_loss_matrix = tf.sub(tf.add(B_t_batch_mul_o1, tf.nn.embedding_lookup(r, B_r_index)), B_t_batch_mul_o2)
            
            self._B_t_loss = B_t_loss = tf.reduce_sum(
                tf.sqrt(tf.reduce_sum(tf.square(B_t_loss_matrix), 1)))
            
            B_h_batch_mul_o1 = tf.squeeze(tf.batch_matmul(tf.expand_dims(tf.nn.embedding_lookup(ht, B_o1_index), 1), tf.reshape(tf.nn.embedding_lookup(Mh, B_r_index), [-1, self.dim, self.dim])), [1])
            B_h_batch_mul_o2 = tf.squeeze(tf.batch_matmul(tf.expand_dims(tf.nn.embedding_lookup(ht, B_o2_index), 1), tf.reshape(tf.nn.embedding_lookup(Mh, B_r_index), [-1, self.dim, self.dim])), [1])
            B_h_loss_matrix = tf.sub(tf.add(B_h_batch_mul_o1, tf.nn.embedding_lookup(r, B_r_index)), B_h_batch_mul_o2)

            self._B_h_loss = B_h_loss = tf.reduce_sum(
                tf.sqrt(tf.reduce_sum(tf.square(B_h_loss_matrix), 1)))
            
            # penalize on pre- and post-projected vectors whose norm exceeds 1
            B_vec_restraint = tf.concat(0, [tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(tf.nn.embedding_lookup(ht, B_o1_index)), 1)), 1.), 0.), tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(tf.nn.embedding_lookup(ht, B_o2_index)), 1)), 1.), 0.)])
            B_t_proj_restraint = tf.concat(0, [tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(B_t_batch_mul_o1), 1)), 1.), 0.), tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(B_t_batch_mul_o2), 1)), 1.), 0.)])
            B_h_proj_restraint = tf.concat(0, [tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(B_h_batch_mul_o1), 1)), 1.), 0.), tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(B_h_batch_mul_o2), 1)), 1.), 0.)])
            B_rel_restraint = tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(tf.nn.embedding_lookup(ht, B_r_index)), 1)), 1.), 0.)

            # Type C loss : penalize all pre- and post-projected vec that norm exceeds 1
            self._C_loss = C_loss = tf.reduce_sum(tf.concat(0, [A_vec_restraint, B_vec_restraint, A_proj_restraint, B_t_proj_restraint, B_h_proj_restraint, A_rel_restraint, B_rel_restraint]))

            # Total loss:
            self._total_loss = total_loss = tf.reduce_sum(tf.pack([A_loss, B_t_loss, B_h_loss, C_loss]))
            
            # Optimizer
            self._lr = lr = tf.placeholder(tf.float32)
            self._opt = opt = tf.train.GradientDescentOptimizer(lr)
            self._train_op_A = train_op_A = opt.minimize(A_loss)
            self._train_op_B_t = train_op_B_t = opt.minimize(B_t_loss)
            self._train_op_B_h = train_op_B_h = opt.minimize(B_h_loss)
            self._train_op_C = train_op_C = opt.minimize(C_loss)

            # Saver
            self._saver = tf.train.Saver()