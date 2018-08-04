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

    def __init__(self, num_rels, num_cons, dim, batch_size, L1=False):
        self._num_rels = num_rels
        self._num_cons = num_cons
        self._dim = dim  # dimension of both relation and ontology. 
        self._batch_size = batch_size
        self._epoch_loss = 0
        # margins
        self._m1 = 0.25
        self._m2 = 0.25
        self.L1 = L1
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
            # Mh has |r| number of matrices, each dedicated to a relation
            self._Mh = Mh = tf.get_variable(
                name='Mh', 
                shape=[self.num_rels, self.dim * self.dim],
                dtype=tf.float32,)
            # Mt has |r| number of matrices, each dedicated to a relation
            self._Mt = Mt = tf.get_variable(
                name='Mt', 
                shape=[self.num_rels, self.dim * self.dim],
                dtype=tf.float32)

            self._ht_assign = ht_assign = tf.placeholder(
                name='ht_assign',
                shape=[self.num_cons, self.dim],
                dtype=tf.float32)
            self._r_assign = r_assign = tf.placeholder(
                name='r_assign',
                shape=[self.num_rels, self.dim],
                dtype=tf.float32)
            self._m_assign = m_assign = tf.placeholder(
                name='r_assign',
                shape=[self.num_rels, self.dim * self.dim],
                dtype=tf.float32)

            # Type A loss : [|| M_hr h + r - M_tr t ||_2 + m1 - || M_hr h' + r - M_tr t' ||_2]+    here [.]+ means max (. , 0)
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
            self._A_hn_index = A_hn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='A_hn_index')
            self._A_tn_index = A_tn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='A_tn_index')
            '''
            A_loss_matrix = tf.sub(
                tf.add(
                    tf.matmul(A_h_con_batch, tf.reshape(A_mat_h_batch, [-1, self.dim, self.dim])),
                    A_rel_batch),
                tf.matmul(A_t_con_batch, tf.reshape(A_mat_t_batch, [-1, self.dim, self.dim]))
            )'''
            
            # a batch of vectors multiply a batch of matrices.
            A_h_con_batch = tf.nn.embedding_lookup(ht,A_h_index)
            A_t_con_batch = tf.nn.embedding_lookup(ht,A_t_index)
            A_rel_batch = tf.nn.embedding_lookup(r,A_r_index)
            A_mat_h_batch = tf.nn.embedding_lookup(Mh,A_r_index)
            A_mat_t_batch = tf.nn.embedding_lookup(Mt,A_r_index)
            A_hn_con_batch = tf.nn.embedding_lookup(ht,A_hn_index)
            A_tn_con_batch = tf.nn.embedding_lookup(ht,A_tn_index)
            # This is a batch of h * M_hr given a batch of (h, r, t)
            A_h_batch_mul = tf.squeeze(tf.matmul(tf.expand_dims(A_h_con_batch, 1), tf.reshape(A_mat_h_batch, [-1, self.dim, self.dim])), [1])
            # This is a batch of t * M_hr given a batch of (h, r, t)
            A_t_batch_mul = tf.squeeze(tf.matmul(tf.expand_dims(A_t_con_batch, 1), tf.reshape(A_mat_t_batch, [-1, self.dim, self.dim])), [1])
            # negative sampled h and t
            A_hn_batch_mul = tf.squeeze(tf.matmul(tf.expand_dims(A_hn_con_batch, 1), tf.reshape(A_mat_h_batch, [-1, self.dim, self.dim])), [1])
            A_tn_batch_mul = tf.squeeze(tf.matmul(tf.expand_dims(A_tn_con_batch, 1), tf.reshape(A_mat_t_batch, [-1, self.dim, self.dim])), [1])
            
            # This stores h M_hr + r - t M_tr
            A_loss_matrix = tf.sub(
                tf.add(A_h_batch_mul, A_rel_batch),
                A_t_batch_mul
            )
            # This stores h' M_hr + r - t' M_tr for negative samples
            A_neg_matrix = tf.sub(
                tf.add(A_hn_batch_mul, A_rel_batch),
                A_tn_batch_mul
            )
            # L-2 norm
            # [||h M_hr + r - t M_tr|| + m1 - ||h' M_hr + r - t' M_tr||)]+     here [.]+ means max (. , 0)
            # L1 norm
            if self.L1:
                self._A_loss = A_loss = tf.reduce_sum(
                    tf.maximum(
                    tf.sub(tf.add(tf.reduce_sum(tf.abs(A_loss_matrix), 1), self._m1),
                    tf.reduce_sum(tf.abs(A_neg_matrix), 1)), 
                    0.)
                ) 
            else:
                self._A_loss = A_loss = tf.reduce_sum(
                    tf.maximum(
                    tf.sub(tf.add(tf.sqrt(tf.reduce_sum(tf.square(A_loss_matrix), 1)), self._m1),
                    tf.sqrt(tf.reduce_sum(tf.square(A_neg_matrix), 1))), 
                    0.)
                ) 

            # soft-constraint on vector norms for both positive and negative sampled h and t
            # [||h|| - 1]+  +  [||t|| - 1]+  +  [||h'|| - 1]+  +  [||t'|| - 1]+
            A_vec_restraint = tf.concat(0, [tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(A_h_con_batch), 1)), 1.), 0.), tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(A_t_con_batch), 1)), 1.), 0.), 
            tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(A_hn_con_batch), 1)), 1.), 0.), tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(A_tn_con_batch), 1)), 1.), 0.)])
            # soft-constraint on projected vectors for both positive and negative sampled h and t
            A_proj_restraint = tf.concat(0, [tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(A_h_batch_mul), 1)), 1.), 0.), tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(A_t_batch_mul), 1)), 1.), 0.), 
            tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(A_hn_batch_mul), 1)), 1.), 0.), tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(A_tn_batch_mul), 1)), 1.), 0.)])
            A_rel_restraint = tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(A_rel_batch), 1)), 2.), 0.)

            # Type B loss : 
            # 2 losses: t-related <- omega(M_t o1, M_t o2) and h-related <- omega(M_h o1, M_h o2)
            # Let's use || a M_hr + r - b M_tr ||_2 as omega(a,b)
            # They share the same input place holders
            # Negative sampling samples only the "many" end

            self._B_h_index = B_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='B_h_index')
            self._B_r_index = B_r_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='B_r_index')
            self._B_t_index = B_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='B_t_index')
            # negative sampled h and t
            self._B_hn_index = B_hn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='B_hn_index')
            self._B_tn_index = B_tn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self.batch_size],
                name='B_tn_index')


            B_con_h_batch = tf.nn.embedding_lookup(ht,B_h_index)
            B_con_t_batch = tf.nn.embedding_lookup(ht,B_t_index)
            B_mat_h_batch = tf.nn.embedding_lookup(Mh,B_r_index)
            B_mat_t_batch = tf.nn.embedding_lookup(Mt,B_r_index)
            B_rel_batch = tf.nn.embedding_lookup(r,B_r_index)
            B_con_hn_batch = tf.nn.embedding_lookup(ht,B_hn_index)
            B_con_tn_batch = tf.nn.embedding_lookup(ht,B_tn_index)
            # multiplication of a batch of vectors and a batch of matrices
            B_t_batch_mul_head = tf.squeeze(tf.matmul(tf.expand_dims(B_con_h_batch, 1), tf.reshape(B_mat_h_batch, [-1, self.dim, self.dim])), [1])
            B_t_batch_mul_tail = tf.squeeze(tf.matmul(tf.expand_dims(B_con_t_batch, 1), tf.reshape(B_mat_t_batch, [-1, self.dim, self.dim])), [1])
            # multiplication of a batch of vectors and a batch of matrices for negative samples
            B_tn_batch_mul_head = tf.squeeze(tf.matmul(tf.expand_dims(B_con_hn_batch, 1), tf.reshape(B_mat_h_batch, [-1, self.dim, self.dim])), [1])
            B_tn_batch_mul_tail = tf.squeeze(tf.matmul(tf.expand_dims(B_con_tn_batch, 1), tf.reshape(B_mat_t_batch, [-1, self.dim, self.dim])), [1])
            # t*M_hr + r ~ t*M_tr
            # This stores h M_hr + r - t M_tr for more t's of the singular h's. Below it is the one for negative samples
            B_t_loss_matrix = tf.sub(tf.add(B_t_batch_mul_head, B_rel_batch), B_t_batch_mul_tail)
            B_tn_loss_matrix = tf.sub(tf.add(B_tn_batch_mul_head, B_rel_batch), B_tn_batch_mul_tail)
            
            # [||h M_hr + r - t M_tr|| + m1 - ||h M_hr + r - t' M_tr||]+   Actually only t is corrupted for B_t related batches
            if self.L1:
                self._B_t_loss = B_t_loss = tf.reduce_sum(
                    tf.maximum(
                    tf.sub(
                    tf.add(tf.reduce_sum(tf.abs(B_t_loss_matrix), 1), self._m2), tf.reduce_sum(tf.abs(B_tn_loss_matrix), 1)
                    ), 0.)
                )
            else:
                self._B_t_loss = B_t_loss = tf.reduce_sum(
                    tf.maximum(
                    tf.sub(
                    tf.add(tf.sqrt(tf.reduce_sum(tf.square(B_t_loss_matrix), 1)), self._m2), tf.sqrt(tf.reduce_sum(tf.square(B_tn_loss_matrix), 1))
                    ), 0.)
                )
            
            # multiplication of a batch of vectors and a batch of matrices
            B_h_batch_mul_head = tf.squeeze(tf.matmul(tf.expand_dims(B_con_h_batch, 1), tf.reshape(B_mat_h_batch, [-1, self.dim, self.dim])), [1])
            B_h_batch_mul_tail = tf.squeeze(tf.matmul(tf.expand_dims(B_con_t_batch, 1), tf.reshape(B_mat_t_batch, [-1, self.dim, self.dim])), [1])
            # multiplication of a batch of vectors and a batch of matrices for negative samples
            B_hn_batch_mul_head = tf.squeeze(tf.matmul(tf.expand_dims(B_con_hn_batch, 1), tf.reshape(B_mat_h_batch, [-1, self.dim, self.dim])), [1])
            B_hn_batch_mul_tail = tf.squeeze(tf.matmul(tf.expand_dims(B_con_tn_batch, 1), tf.reshape(B_mat_t_batch, [-1, self.dim, self.dim])), [1])
            # t*M_tr - r ~ h*M_hr
            # This stores h M_hr + r - t M_tr for more h's of the singular t's. Below it is the one for negative samples
            B_h_loss_matrix = tf.sub(tf.sub(B_h_batch_mul_tail, B_rel_batch), B_h_batch_mul_head)
            B_hn_loss_matrix = tf.sub(tf.sub(B_hn_batch_mul_tail, B_rel_batch), B_hn_batch_mul_head)

            #  [||t M_tr - r - h M_hr|| + m2 - ||t M_tr - r - h M_hr|| ]+      Actually only h is corrupted for B_h related batches
            if self.L1:
                self._B_h_loss = B_h_loss = tf.reduce_sum(
                    tf.maximum(
                    tf.sub(
                    tf.add(tf.reduce_sum(tf.abs(B_h_loss_matrix), 1), self._m2), tf.reduce_sum(tf.abs(B_hn_loss_matrix), 1)
                    ), 0.)
                )
            else:
                self._B_h_loss = B_h_loss = tf.reduce_sum(
                    tf.maximum(
                    tf.sub(
                    tf.add(tf.sqrt(tf.reduce_sum(tf.square(B_h_loss_matrix), 1)), self._m2), tf.sqrt(tf.reduce_sum(tf.square(B_hn_loss_matrix), 1))
                    ), 0.)
                )
            
            # penalize on pre- and post-projected vectors whose norm exceeds 1

            B_vec_restraint = tf.concat(0, [tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(B_con_h_batch), 1)), 1.), 0.), tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(B_con_t_batch), 1)), 1.), 0.), 
            tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(B_con_hn_batch), 1)), 1.), 0.), tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(B_con_tn_batch), 1)), 1.), 0.)])
            B_t_proj_restraint = tf.concat(0, [tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(B_t_batch_mul_head), 1)), 1.), 0.), tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(B_t_batch_mul_tail), 1)), 1.), 0.), 
            tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(B_tn_batch_mul_head), 1)), 1.), 0.), tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(B_tn_batch_mul_tail), 1)), 1.), 0.)])
            B_h_proj_restraint = tf.concat(0, [tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(B_h_batch_mul_head), 1)), 1.), 0.), tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(B_h_batch_mul_tail), 1)), 1.), 0.), 
            tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(B_hn_batch_mul_head), 1)), 1.), 0.), tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(B_hn_batch_mul_tail), 1)), 1.), 0.)])
            B_rel_restraint = tf.maximum(tf.sub(tf.sqrt(tf.reduce_sum(tf.square(B_rel_batch), 1)), 2.), 0.)


            # Type C loss : Soft-constraint on vector norms
            #self._C_loss = C_loss = tf.reduce_sum(tf.concat(0, [A_vec_restraint, B_vec_restraint, A_proj_restraint, B_t_proj_restraint, B_h_proj_restraint, A_rel_restraint, B_rel_restraint]))
            #self._C_loss = C_loss = tf.reduce_sum(tf.concat(0, [A_vec_restraint, B_vec_restraint, A_proj_restraint, B_t_proj_restraint, B_h_proj_restraint]))
            #self._C_loss = C_loss = tf.reduce_sum(tf.concat(0, [A_vec_restraint, A_proj_restraint, A_rel_restraint]))
            self._C_loss_A = C_loss_A = tf.reduce_sum(tf.concat(0, [A_vec_restraint, A_proj_restraint, A_rel_restraint]))
            self._C_loss_B1 = C_loss_B1 = tf.reduce_sum(tf.concat(0, [B_vec_restraint, B_t_proj_restraint, B_rel_restraint]))
            self._C_loss_B2 = C_loss_B2 = tf.reduce_sum(tf.concat(0, [B_vec_restraint, B_h_proj_restraint, B_rel_restraint]))
            
            # Optimizer
            self._lr = lr = tf.placeholder(tf.float32)
            self._opt = opt = tf.train.GradientDescentOptimizer(lr)
            self._train_op_A = train_op_A = opt.minimize(A_loss)
            self._train_op_B_t = train_op_B_t = opt.minimize(B_t_loss)
            self._train_op_B_h = train_op_B_h = opt.minimize(B_h_loss)
            #self._train_op_C = train_op_C = opt.minimize(C_loss)
            self._train_op_C_A = train_op_C_A = opt.minimize(C_loss_A)
            self._train_op_C_B1 = train_op_C_B1 = opt.minimize(C_loss_B1)
            self._train_op_C_B2 = train_op_C_B2 = opt.minimize(C_loss_B2)

            self._assign_ht_op = assign_ht_op = self._ht.assign(ht_assign)
            self._assign_r_op = assign_r_op = self._r.assign(r_assign)
            self._assign_m_op = assign_m_op = [self._Mt.assign(m_assign), self._Mh.assign(m_assign)]
            

            # Saver
            self._saver = tf.train.Saver()