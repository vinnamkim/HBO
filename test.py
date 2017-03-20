#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 16:41:35 2017

@author: vinnam
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
import math

FLOATING_TYPE = 'float32'
TERMINATION_COND = math.pow(10, -6)


def Lower_bound(N, K, D, M, X, y, Z, mu, Sigma, sigma_f, sigma):
    # N, K, D, M are constants
    # X is a N x D placeholder
    # y is a N x 1 placeholder
    # Z is a M x K variable
    # mu is a K x D variable 
    # Sigma is a K x D variable
    # sigma is variable
    # sigma_f is variable
    
    # K_{u,u} is M x M matrix
    r = tf.reshape(tf.reduce_sum(tf.square(Z), 1), [-1, 1])
    K_uu = tf.exp(-0.5 * (r - 2 * tf.matmul(Z, tf.transpose(Z)) + tf.transpose(r)))
    K_uu_chol = tf.cholesky(K_uu)

    # Psi1 is a M x M matrix
    div = tf.transpose(tf.expand_dims(tf.matmul(X, tf.transpose(Sigma)), 0)) # K x N x 1
    div_plusone = div + 1 # K x N x 1
    mu_x = tf.transpose(tf.expand_dims(tf.matmul(X, tf.transpose(mu)), 0)) # K x N x 1
    ZZ = tf.transpose(tf.expand_dims(Z, 0), [2, 0, 1]) # K x 1 x M
    
    Psi1 = sigma_f * tf.reduce_prod(tf.rsqrt(div_plusone) * tf.exp(-tf.square(mu_x - ZZ) / div_plusone), axis = 0) # K x N x M
    
    # Psi2 is a M x M matrix
    ZZ_t = tf.transpose(ZZ, [0, 2, 1]) # K x M x 1
    twodiv_plusone = tf.transpose(tf.expand_dims(2 * div + 1, -1), [1, 0, 2, 3]) # N x K x 1 x 1
    
    Psi21 = tf.square(sigma_f) * tf.exp(-0.25 * tf.reduce_sum(tf.square(ZZ - ZZ_t), axis = 0))
    
    mu_xx = tf.transpose(tf.expand_dims(mu_x, -1), [1, 0, 2, 3]) # N x K x 1 x 1
    ZZZ = tf.expand_dims(0.5 * (ZZ + ZZ_t), 0) # 1 x K x M x M
    Psi22 = tf.reduce_sum(tf.reduce_prod(tf.rsqrt(twodiv_plusone) * tf.exp(-tf.square(mu_xx - ZZZ) / twodiv_plusone), axis = 1), axis = 0) # M x M
    Psi2 = Psi21 * Psi22
    
    sigma_sq = tf.square(sigma)
    A = tf.cholesky(tf.square(sigma) * K_uu + Psi2)
    C = 0.5 / sigma_sq
    N_ = tf.constant(N, dtype = FLOATING_TYPE)
    M_ = tf.constant(M, dtype = FLOATING_TYPE)
    pi_ = tf.constant(np.pi, dtype = FLOATING_TYPE)
    
    F1 = tf.reduce_sum(- N_ * 0.5 * tf.log(2 * pi_) - \
    (N_ - M_) * tf.log(sigma) + \
    0.5 * tf.reduce_sum(tf.log(tf.diag_part(K_uu_chol))) - \
    0.5 * tf.reduce_sum(tf.log(tf.diag_part(A))) - \
    C * tf.matmul(tf.transpose(y), y) + \
    C * tf.matmul(tf.matmul(tf.transpose(y), Psi1), tf.matmul(tf.cholesky_solve(A, tf.transpose(Psi1)), y)) - \
    C * N * tf.square(sigma_f) + \
    C * tf.trace(tf.cholesky_solve(K_uu, Psi2)))
    
    D_ = tf.constant(D, dtype = FLOATING_TYPE)
    
    KL = 0.5 * tf.reduce_sum(tf.log(tf.reduce_sum(tf.square(Sigma), 1)) - D_ * tf.log(tf.reduce_sum(tf.square(mu) + tf.square(Sigma), 1)) + D_ * tf.log(D_))
        
    l_square = tf.reduce_sum(tf.square(mu) + tf.square(Sigma), 1) / D_
    
    return {'F1' : F1, 'KL' : KL, 'l_square' : l_square}

def GP_train(N, K, D, M, X, y, Z, mu, Sigma, sigma_f, sigma, barrier_coeff):
    LB = Lower_bound(N, K, D, M, X, y, Z, mu, Sigma, sigma_f, sigma)    
    log_barrier = - barrier_coeff * tf.reduce_sum(tf.log(Sigma)) - barrier_coeff * tf.log(sigma_f) - barrier_coeff * tf.log(sigma)
    OBJ_train = - (LB['F1'] - LB['KL']) + log_barrier
    opt = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    
    train_op = opt.minimize(OBJ_train)
    
    return {'train_op' : train_op, 'obj' : OBJ_train, 'l_square' : LB['l_square']}

def Toy_sim(N, D, d, sigma_f, sigma):
    X = np.random.uniform(size = [N, D])
    W = np.random.normal(size = [d, D])
    X_proj = np.transpose(np.matmul(W, np.transpose(X)))
    r = np.reshape(np.sum(np.square(X_proj), axis = 1), [-1, 1])
    K = np.square(sigma_f) * np.exp(-0.5 * (r - 2 * np.matmul(X_proj, np.transpose(X_proj)) + np.transpose(r)))
    K = K + np.square(sigma) * np.eye(N)
    chol = np.linalg.cholesky(K)
    y = np.matmul(chol, np.random.normal(size = [N, 1]))
    
    return {'y' : y, 'X' : X, 'W' : W}
    
N = 100
D = 25
d = 5
sigma_f = 0.1
sigma = 0.05

sim = Toy_sim(N, D, d, sigma_f, sigma)

M = 50
K = 20 

if 'sess' in globals() and sess:
    sess.close()
tf.reset_default_graph()
    
X_ = tf.placeholder(name = 'X', shape = [None, D], dtype = FLOATING_TYPE)
y_ = tf.placeholder(name = 'Y', shape = [None, 1], dtype = FLOATING_TYPE)
barrier_coeff_ = tf.placeholder(name = 'barrier_coeff', shape = [], dtype = FLOATING_TYPE)
Z_ = tf.get_variable('Z', initializer = tf.random_uniform_initializer(), shape = [M, K], dtype = FLOATING_TYPE)
mu_ = tf.get_variable('mu', initializer = tf.random_normal_initializer(), shape = [K, D], dtype = FLOATING_TYPE)
Sigma_ = tf.get_variable('Sigma', initializer = tf.random_uniform_initializer(), shape = [K, D], dtype = FLOATING_TYPE)
sigma_f_ = tf.get_variable('sigma_f', initializer = tf.random_uniform_initializer(), shape = [], dtype = FLOATING_TYPE)
sigma_ = tf.get_variable('sigma', initializer = tf.random_uniform_initializer(), shape = [], dtype = FLOATING_TYPE)

LB = Lower_bound(N, K, D, M, X_, y_, Z_, mu_, Sigma_, sigma_f_, sigma_)
         
train_step = GP_train(N, K, D, M, X_, y_, Z_, mu_, Sigma_, sigma_f_, sigma_, barrier_coeff_)

feed_dict = {X_ : sim['X'], y_ : sim['y'], barrier_coeff_ : np.power(10., -6)}
sess = tf.Session()
sess.run(tf.global_variables_initializer())

- sess.run(LB['F1'], feed_dict) + sess.run(LB['KL'], feed_dict)
obj_value = sess.run(train_step['obj'], feed_dict)
test = sess.run(sigma_, feed_dict)
sess.run(mu_, feed_dict)

for i in range(10):
    _, obj_value_new = sess.run([train_step['train_op'], train_step['obj']], feed_dict)
    print(obj_value_new)
    if(i & 100 is 0):
        print(i, obj_value_new)
    if(np.abs(obj_value_new - obj_value) < np.power(10., -8)):
        break
    obj_value = obj_value_new


sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
sess.run(Z_).shape




#####################

X = tf.placeholder(name = 'X', shape = [None, D], dtype = FLOATING_TYPE)
y = tf.placeholder(name = 'Y', shape = [None, 1], dtype = FLOATING_TYPE)
Z = tf.get_variable('Z', initializer = tf.random_uniform_initializer(), shape = [M, K], dtype = FLOATING_TYPE)
mu = tf.get_variable('mu', initializer = tf.random_normal_initializer(), shape = [K, D], dtype = FLOATING_TYPE)
Sigma = tf.get_variable('Sigma', initializer = tf.random_uniform_initializer(), shape = [K, D], dtype = FLOATING_TYPE)
sigma_f = tf.get_variable('sigma_f', initializer = tf.random_uniform_initializer(), shape = [], dtype = FLOATING_TYPE)
sigma = tf.get_variable('sigma', initializer = tf.random_uniform_initializer(), shape = [], dtype = FLOATING_TYPE)



# N, K, D, M are constants
# X is a N x D placeholder
# y is a N x 1 placeholder
# Z is a M x K variable
# mu is a K x D variable 
# Sigma is a K x D variable
# sigma is variable
# sigma_f is variable

# K_{u,u} is M x M matrix
r = tf.reshape(tf.reduce_sum(tf.square(Z), 1), [-1, 1])
K_uu = tf.exp(-0.5 * (r - 2 * tf.matmul(Z, tf.transpose(Z)) + tf.transpose(r)))
K_uu_chol = tf.cholesky(K_uu)

# Psi1 is a M x M matrix
div = tf.transpose(tf.expand_dims(tf.matmul(X, tf.transpose(Sigma)), 0)) # K x N x 1
div_plusone = div + 1 # K x N x 1
mu_x = tf.transpose(tf.expand_dims(tf.matmul(X, tf.transpose(mu)), 0)) # K x N x 1
ZZ = tf.transpose(tf.expand_dims(Z, 0), [2, 0, 1]) # K x 1 x M

Psi1 = sigma_f * tf.reduce_prod(tf.rsqrt(div_plusone) * tf.exp(-tf.square(mu_x - ZZ) / div_plusone), axis = 0) # K x N x M

# Psi2 is a M x M matrix
ZZ_t = tf.transpose(ZZ, [0, 2, 1]) # K x M x 1
twodiv_plusone = tf.transpose(tf.expand_dims(2 * div + 1, -1), [1, 0, 2, 3]) # N x K x 1 x 1

Psi21 = tf.square(sigma_f) * tf.exp(-0.25 * tf.reduce_sum(tf.square(ZZ - ZZ_t), axis = 0))

mu_xx = tf.transpose(tf.expand_dims(mu_x, -1), [1, 0, 2, 3]) # N x K x 1 x 1
ZZZ = tf.expand_dims(0.5 * (ZZ + ZZ_t), 0) # 1 x K x M x M
Psi22 = tf.reduce_sum(tf.reduce_prod(tf.rsqrt(twodiv_plusone) * tf.exp(-tf.square(mu_xx - ZZZ) / twodiv_plusone), axis = 1), axis = 0) # M x M
Psi2 = Psi21 * Psi22

sigma_sq = tf.square(sigma)
A = tf.cholesky(tf.square(sigma) * K_uu + Psi2)
C = 0.5 / sigma_sq
N_ = tf.constant(N, dtype = FLOATING_TYPE)
M_ = tf.constant(M, dtype = FLOATING_TYPE)
pi_ = tf.constant(np.pi, dtype = FLOATING_TYPE)

F1 = tf.reduce_sum(- N_ * 0.5 * tf.log(2 * pi_) - \
(N_ - M_) * tf.log(sigma) + \
0.5 * tf.reduce_sum(tf.log(tf.diag_part(K_uu_chol))) - \
0.5 * tf.reduce_sum(tf.log(tf.diag_part(A))) - \
C * tf.matmul(tf.transpose(y), y) + \
C * tf.matmul(tf.matmul(tf.transpose(y), Psi1), tf.matmul(tf.cholesky_solve(A, tf.transpose(Psi1)), y)) - \
C * N * tf.square(sigma_f) + \
C * tf.trace(tf.cholesky_solve(K_uu, Psi2)))

D_ = tf.constant(D, dtype = FLOATING_TYPE)

KL = 0.5 * tf.reduce_sum(tf.log(tf.reduce_sum(tf.square(Sigma), 1)) - D_ * tf.log(tf.reduce_sum(tf.square(mu) + tf.square(Sigma), 1)) + D_ * tf.log(D_))
    
l_square = tf.reduce_sum(tf.square(mu) + tf.square(Sigma), 1) / D_

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

feed_dict = {X : sim['X'], y : sim['y']}

sess.run(KL, feed_dict)


###################





while(True):
    opt.compute_gradients(f)
    term, _ = sess.run([grad_norm, train_op])
    
    if(term < cond):
        break
        




    sess.run([x, y])
    sess.run(grad_norm)
    
def fun2(x):
    x1 = x[0]
    x2 = x[1]
    
    c1 = 3/2 - x1 - 2 * x2 - 1/2 * np.sin(2 * np.pi * (np.square(x1) - 2 * x2))
    c2 = np.square(x1) + np.square(x2) - 3/2
    
    return c1 + c2

def GPfit(x, y, noise_, sigma_, length_):
    # x : D by N
    # y : 1 by N
    GPsigma_chol

def GPsigma_chol(X, noise_, sigma_, length_):
    r = tf.reshape(tf.reduce_sum(tf.square(X), 1), [-1, 1])
    C = r - 2 * tf.matmul(X, tf.transpose(X)) + tf.transpose(r)
    C = tf.div(C, tf.square(length_))
    C = tf.multiply(tf.constant(-0.5, dtype = FLOATING_TYPE), C)
    C = tf.exp(C)
    C = tf.square(sigma_) * C
    C = tf.matrix_set_diag(C, (tf.square(noise_) + tf.square(sigma_)) * tf.ones_like(tf.diag_part(C)))
    
    return tf.cholesky(C)


sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32)
c 
f = fun1(x)
feed_dict = {x : [np.linspace(0, 1, num = 100), np.repeat(0, 100), np.linspace(0, 100, num = 100)]}
plt.plot(sess.run(f, feed_dict))


plt.plot(fun2(feed_dict[x]))




b = tf.reshape(a, [2, 3, 1])

c = tf.expand_dims(x, 0)

N = 4
K = 2
D = 3
M = 3
X = tf.constant([[2., 2., 2.], [1., 1., 1.], [3., 4., 5.], [1., 2., 3.]])
mu = tf.constant([[ 1.,  3.,  5.], [ 2.,  4.,  6.]])
Sigma = tf.constant([[ 1.,  3.,  5.], [ 2.,  4.,  6.]])
Z = tf.constant([[1., 1.], [2., 3.], [3., 2.]])
sigma_f = tf.constant(0.1)
e = []

def Psi1(i, j, N, K, D, M, X, y, Z, mu, Sigma, sigma_f, sigma):
    Z_sp = tf.split(Z, K, axis = 1)
    mu_sp = tf.split(mu, K, axis = 0)
    Sigma_sp = tf.split(Sigma, K, axis = 0)
    
    e = []
    for Sigma_k, mu_k, Z_k in zip(Sigma_sp, mu_sp, Z_sp):
        div = tf.reshape(tf.reduce_sum(tf.multiply(tf.square(X), Sigma_k), axis = 1) + 1, [-1, 1])
        e.append(tf.rsqrt(div) * tf.exp(-tf.square(tf.matmul(X, tf.transpose(mu_k)) - tf.transpose(Z_k)) / (2 * div)))
    
    return sigma_f * tf.reduce_prod(tf.stack(e), axis = 0)
    
sess = tf.InteractiveSession()

a = tf.tile(X, [K, 1])




sess.run()


def Psi2(i, j, N, K, D, M, X, y, Z, mu, Sigma, sigma_f, sigma):
    X_sp = tf.split(X, N, axis = 0)
    Z_sp = tf.split(Z, K, axis = 1)
    mu_sp = tf.split(mu, K, axis = 0)
    Sigma_sp = tf.split(Sigma, K, axis = 0)
    
    e1 = []
    for Z_k in Z_sp:
        e1.append(tf.square(Z_k - tf.transpose(Z_k)))
        
    e2 = tf.square(sigma_f) * tf.exp(- 0.25 * tf.reduce_sum(tf.stack(e1), axis = 0))

    e3 = []
    for X_i in X_sp:
        e = []
        for Z_k, mu_k, Sigma_k in zip(Z_sp, mu_sp, Sigma_sp):
            div = 2 * tf.reduce_sum(tf.multiply(tf.square(X_i), Sigma_k)) + 1
            e.append(tf.rsqrt(div) * tf.exp(-tf.square(tf.reduce_sum(tf.multiply(X_i, mu_k)) - 0.5 * (Z_k + tf.transpose(Z_k))) / div))
        e3.append(tf.reduce_prod(tf.stack(e), axis = 0))
    
    Psi2 = tf.multiply(e2, tf.reduce_sum(tf.stack(e3), axis = 0))
    
    return Psi2
    
for a, b in zip(range(0, 5), range(1, 6)):
    print("%d %d" % (a, b))

for d_ in d:
    d_
    
f = tf.stack(e)

tf.matmul(c, f)

sess.run(tf.tile(x, [2, 1]))

tf.diag(d)
tf.split()
a
sess.run(tf.diag(b))
div = tf.matmul(c)
sess.run(a)
sess.run(b[0, :, :])


