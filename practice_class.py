#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:41:10 2017

@author: vinnam
"""
import tensorflow as tf
import numpy as np

class A:
    #To build the graph when instantiated
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
             self.const = tf.constant([1.0])
    # To launch the graph
    def launchG(self):
        with tf.Session(graph=self.graph) as sess:
             print sess.run(self.const)
             
class B(A):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
             self.const = tf.constant([2.0])

A = A()
A.launchG()
B = B()
B.launchG()



length = tf.constant(0.1)
noise = tf.constant(0.1)
Z = tf.constant(np.array(np.random.normal(size = [10, 3]), dtype = 'float32'))
mu = tf.constant(np.array(np.zeros(shape = [10, 1])), dtype = 'float32')
test = mvn_likelihood_sqkern(Z, mu, length, noise)

sess = tf.Session()
print sess.run(test)

reset_graph()

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()
    return



def mvn_likelihood_sqkern(Z, mu, length, noise, JITTER_VALUE = 1e-5, FLOATING_TYPE = 'float32'):
    r = tf.reshape(tf.reduce_sum(tf.square(Z), 1), [-1, 1]) # M x 1
    K_uu = tf.exp(-0.5 * (r - 2 * tf.matmul(Z, tf.transpose(Z)) + tf.transpose(r))) # Check
    L = tf.cholesky(tf.reciprocal(tf.square(length)) * K_uu + (JITTER_VALUE + tf.square(noise)) * tf.diag(tf.squeeze(tf.ones_like(r))))
    
    d = Z - mu
    alpha = tf.matrix_triangular_solve(L, d, lower=True)
    num_col = 1 if tf.rank(Z) == 1 else tf.shape(Z)[1]
    num_col = tf.cast(num_col, FLOATING_TYPE)
    num_dims = tf.cast(tf.shape(Z)[0], FLOATING_TYPE)
    ret = - 0.5 * num_dims * num_col * np.log(2 * np.pi)
    ret += - num_col * tf.reduce_sum(tf.log(tf.diag_part(L)))
    ret += - 0.5 * tf.reduce_sum(tf.square(alpha))
    return ret
