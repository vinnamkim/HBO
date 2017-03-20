#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 16:41:35 2017

@author: vinnam
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sinc_2d
from sklearn import preprocessing

FLOATING_TYPE = 'float32'
JITTER_VALUE = 1e-5

def Lower_bound(N, K, D, M, X, y, Z, mu, log_Sigma, log_sigma_f, log_tau):
    # N, K, D, M are constants
    # X is a N x D placeholder
    # y is a N x 1 placeholder
    # Z is a M x K variable
    # mu is a K x D variable 
    # Sigma is a K x D variable
    # tau is a variable
    # sigma_f is a variable
    
    N_ = tf.constant(N, dtype = FLOATING_TYPE)
    M_ = tf.constant(M, dtype = FLOATING_TYPE)
    D_ = tf.constant(D, dtype = FLOATING_TYPE)
    K_ = tf.constant(K, dtype = FLOATING_TYPE)
    pi_ = tf.constant(np.pi, dtype = FLOATING_TYPE)
    
    sigma_f_sq = tf.exp(2 * log_sigma_f)
    tau_sq = tf.exp(2 * log_tau)
    Sigma_sq = tf.exp(2 * log_Sigma)
    
    sigma_f_sq_div_tau_sq = tf.exp(2 * (log_sigma_f - log_tau))
    
    # K_{u,u} is M x M matrix
    r = tf.reshape(tf.reduce_sum(tf.square(Z), 1), [-1, 1])
    K_uu = tf.exp(-0.5 * (r - 2 * tf.matmul(Z, tf.transpose(Z)) + tf.transpose(r))) # Check
    Lm = tf.cholesky(K_uu + JITTER_VALUE * tf.eye(M))
    
    # Psi1 is a M x M matrix
    div = tf.transpose(tf.expand_dims(tf.matmul(tf.square(X), tf.transpose(Sigma_sq)), 0)) # K x N x 1
    div_plusone = div + 1 # K x N x 1 # Check
    mu_x = tf.transpose(tf.expand_dims(tf.matmul(X, tf.transpose(mu)), 0)) # K x N x 1
    ZZ = tf.transpose(tf.expand_dims(Z, 0), [2, 0, 1]) # K x 1 x M
    
    Psi1 = tf.reduce_prod(tf.rsqrt(div_plusone) * tf.exp(- tf.square(mu_x - ZZ) / (2 * div_plusone)), axis = 0) # K x N x M
    
    # Psi2 is a M x M matrix
    ZZ_t = tf.transpose(ZZ, [0, 2, 1]) # K x M x 1
    twodiv_plusone = tf.transpose(tf.expand_dims(2 * div + 1, -1), [1, 0, 2, 3]) # N x K x 1 x 1
    Psi21 = tf.exp(-0.25 * tf.reduce_sum(tf.square(ZZ - ZZ_t), axis = 0)) # Check
    
    # Psi21 Check
    mu_xx = tf.transpose(tf.expand_dims(mu_x, -1), [1, 0, 2, 3]) # N x K x 1 x 1
    ZZZ = tf.expand_dims(0.5 * (ZZ + ZZ_t), 0) # 1 x K x M x M
    Psi22 = tf.reduce_sum(tf.reduce_prod(tf.rsqrt(twodiv_plusone) * tf.exp(-tf.square(mu_xx - ZZZ) / twodiv_plusone), axis = 1), axis = 0) # M x M
    
    Psi2 = Psi21 * Psi22 # M x M
    
    ## CHECK
    
    
    Psi1InvLm = tf.transpose(tf.matrix_triangular_solve(Lm, tf.transpose(Psi1))) # N x M
    
    C = tf.matrix_triangular_solve(Lm, tf.transpose(tf.matrix_triangular_solve(Lm, Psi2))) # M x M
    
    A = tau_sq * tf.eye(M) + sigma_f_sq * C # M x M
    
    La = tf.cholesky(A) # M x M
    
    YPsi1InvLm = tf.matmul(tf.transpose(y), Psi1InvLm) # 1 x M
    
    YPsi1InvLmInvLa = tf.transpose(tf.matrix_triangular_solve(La, tf.transpose(YPsi1InvLm))) # 1 x M
    
    YYall = tf.reduce_sum(tf.square(y))
    
    F012 = - (N_ - M_) * log_tau - 0.5 * N_ * tf.log(2 * pi_) - \
    (0.5 / tau_sq) * YYall - tf.reduce_sum(tf.log(tf.diag_part(La)))
    
    F3 = (0.5 * sigma_f_sq_div_tau_sq) * tf.reduce_sum(tf.square(YPsi1InvLmInvLa))
    
    TrK = - (0.5 * sigma_f_sq_div_tau_sq) * (N_ - tf.reduce_sum(tf.diag_part(C)))
    
    F = F012 + F3 + TrK
    
    mu_Sigma_sq = tf.reduce_sum(tf.square(mu) + Sigma_sq, 1) # D 
    
    KL = - tf.reduce_sum(log_Sigma) + 0.5 * D_ * tf.reduce_sum(tf.log(mu_Sigma_sq)) # Check
        
    #KL = - (D_ * 0.5 + alpha) * tf.reduce_sum(tf.log(2 * beta + tf.reduce_sum(tf.square(mu) + Sigma_sq, 1))) + 0.5 * tf.reduce_sum(tf.log(Sigma_sq))
    
    l_square = mu_Sigma_sq / D_
    
    return {'F1' : F, 'KL' : KL, 'l_square' : l_square, 'F012' : F012, 'F3' : F3, 'TrK' : TrK, 'La' : La, 'YPsi1InvLm' : YPsi1InvLm}

def GP_train(N, K, D, M, X, y, Z, mu, log_Sigma, log_sigma_f, log_tau, learning_rate):
    LB = Lower_bound(N, K, D, M, X, y, Z, mu, log_Sigma, log_sigma_f, log_tau)    
    
    OBJ_train = - (LB['F1'] - LB['KL'])
    
    opt = tf.train.AdamOptimizer(learning_rate = learning_rate)
    
    train_op = opt.minimize(OBJ_train)
    
    return {'train_op' : train_op, 'obj' : OBJ_train, 'l_square' : LB['l_square'], 'F1' : LB['F1'], 'KL' : LB['KL']}

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()
    return

N = 200
D = 10
M = 40
K = 4
Iter = 1000

sim = sinc_2d.sinc_2d(N, D, 5, 0.01)
data = sim.data()

reset_graph()

X = tf.placeholder(name = 'X', shape = [None, D], dtype = FLOATING_TYPE)
y = tf.placeholder(name = 'Y', shape = [None, 1], dtype = FLOATING_TYPE)
Z = tf.get_variable('Z', initializer = tf.random_uniform_initializer(), shape = [M, K], dtype = FLOATING_TYPE)
mu = tf.get_variable('mu', initializer = tf.random_normal_initializer(), shape = [K, D], dtype = FLOATING_TYPE)
log_Sigma = tf.get_variable('Sigma', initializer = tf.random_uniform_initializer(), shape = [K, D], dtype = FLOATING_TYPE)
log_sigma_f = tf.get_variable('sigma_f', initializer = tf.random_uniform_initializer(), shape = [], dtype = FLOATING_TYPE)
log_tau = tf.get_variable('sigma', initializer = tf.random_uniform_initializer(), shape = [], dtype = FLOATING_TYPE)

train_step = GP_train(N, K, D, M, X, y, Z, mu, log_Sigma, log_sigma_f, log_tau, learning_rate = 1e-2)

feed_dict = {X : data['X'], y : data['Y']}

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(Iter):
    _, new_obj = sess.run([train_step['train_op'], train_step['obj']], feed_dict)
    print(new_obj)

W_hat = sess.run(mu)

lsq_hat = sess.run(train_step['l_square'])

idx_list = np.argsort(lsq_hat)[::-1]

X_p_hat = np.matmul(data['X'], np.transpose(W_hat[idx_list[0:2], :]))

min_max_scaler = preprocessing.MinMaxScaler()

scaled_X_p_hat = min_max_scaler.fit_transform(-X_p_hat)
scaled_X_p = min_max_scaler.fit_transform(data['X_p'])

plt.scatter(scaled_X_p_hat[:, 0], data['Y'])
plt.scatter(scaled_X_p[:, 1], data['Y'])

plt.scatter(scaled_X_p_hat[:, 1], data['Y'])
plt.scatter(scaled_X_p[:, 0], data['Y'])

plt.scatter(X_p_hat[:, 0], np.sinc(X_p_hat[:, 0]))

sim.show_fun()
sim.show_Y()