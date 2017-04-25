#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:41:37 2017

@author: vinnam
"""

import numpy as np
np.seterr(all='raise')
import tensorflow as tf
import settings
from sampling import hit_and_run
from sampling import sample_enclosingbox

def mvn_likelihood_sqkern(X, y, mu, length_sq, sigma_sq, noise_sq, JITTER_VALUE, FLOATING_TYPE):
    r = tf.reshape(tf.reduce_sum(tf.square(X), 1), [-1, 1]) # M x 1
    K_uu = tf.exp(-(0.5 / length_sq) * (r - 2 * tf.matmul(X, tf.transpose(X)) + tf.transpose(r))) # Check
    L = tf.cholesky(sigma_sq * K_uu + (JITTER_VALUE + noise_sq) * tf.diag(tf.squeeze(tf.ones_like(r, dtype = FLOATING_TYPE))))
    
    d = y - mu
    alpha = tf.matrix_triangular_solve(L, d, lower=True)
    num_col = 1 if tf.rank(y) == 1 else tf.shape(y)[1]
    num_col = tf.cast(num_col, FLOATING_TYPE)
    num_dims = tf.cast(tf.shape(y)[0], FLOATING_TYPE)
    ret = - 0.5 * num_dims * num_col * np.log(2 * np.pi)
    ret += - num_col * tf.reduce_sum(tf.log(tf.diag_part(L)))
    ret += - 0.5 * tf.reduce_sum(tf.square(alpha))

    return ret, L

def f_star(x_star, X, y, length_sq, sigma_sq, noise_sq, L, FLOATING_TYPE):
    # x_star : L x D
    # l : N x D
    # X : N x D
    # L : N x N
    
    xx_star = tf.expand_dims(x_star, 1) # xx_star : L x 1 x D
    
    l = tf.transpose(sigma_sq * tf.exp((-0.5 / length_sq) * tf.reduce_sum(tf.square(X - xx_star), axis = -1))) # N x L
    
    L_inv_l = tf.matrix_triangular_solve(L, l, lower = True) # N x L
    L_inv_y = tf.matrix_triangular_solve(L, y, lower = True) # N x 1
    
    mu = tf.squeeze(tf.matmul(tf.transpose(L_inv_l), L_inv_y)) # L x 1
    var = tf.transpose(sigma_sq - tf.reduce_sum(tf.square(L_inv_l), axis = 0)) # L x 1
#    var = tf.clip_by_value(var, clip_value_min = tf.constant(MIN_VAR), clip_value_max = tf.constant(np.finfo(FLOATING_TYPE).max, dtype = FLOATING_TYPE))
    
    return mu, var
        
def acq_fun(mu_star, var_star, max_fun, beta, method):
    if method is 'EI':
        std_star = tf.sqrt(var_star)
        dist = tf.contrib.distributions.Normal(mu = mu_star, sigma = std_star)
        diff = (mu_star - max_fun)
        Z = diff / tf.sqrt(var_star)
        EI = diff * dist.cdf(Z) + std_star * dist.pdf(Z)
        return EI # L x 1
    elif method is 'UCB':
        return mu_star + tf.sqrt(beta) * tf.sqrt(var_star)
        
def log_barrier(x_star, A, b):
    # x_star : L x D 
    # A : DD x D
    # b : DD x 1
    Ax = tf.matmul(A, tf.transpose(x_star)) # DD x L
    
    return tf.reshape(tf.reduce_sum(tf.log(b - Ax) + tf.log(Ax + b), axis = 0), [-1, 1]) # FOR MAXIMIZATION L x 1
    
class GP:
    def __init__(self, D, K, LEARNING_RATE = 1e-1, ACQ_FUN = 'EI'):
        self.FLOATING_TYPE = settings.dtype
        self.JITTER_VALUE = settings.jitter
        self.ACQ_FUN = ACQ_FUN
        
        FLOATING_TYPE = self.FLOATING_TYPE
        JITTER_VALUE = self.JITTER_VALUE
        
        self.D = D
        
        self.graph = tf.Graph()
        
        self.x_init = np.zeros([1, D], dtype = self.FLOATING_TYPE)
        
        with self.graph.as_default():
            ####### VARIABLES #######
            
            X = tf.placeholder(name = 'X', shape = [None, D], dtype = FLOATING_TYPE)
            y = tf.placeholder(name = 'y', shape = [None, 1], dtype = FLOATING_TYPE)
            
            self.inputs = {'X' : X, 'y' : y}
            
            log_length = tf.placeholder(name = 'log_length', shape = [], dtype = FLOATING_TYPE)
            log_sigma = tf.placeholder(name = 'log_sigma', shape = [], dtype = FLOATING_TYPE)
            log_noise = tf.placeholder(name = 'log_noise', shape = [], dtype = FLOATING_TYPE)
            
            self.params = {'log_length' : log_length, 'log_sigma' : log_sigma, 'log_noise' : log_noise}
            
            z_star = tf.placeholder(name = 'z_star', shape = [None, K], dtype = FLOATING_TYPE)
            A = tf.placeholder(name = 'A', shape = [D, K], dtype = FLOATING_TYPE)
            max_fun = tf.placeholder(name = 'max_fun', shape = [], dtype = FLOATING_TYPE)
            beta = tf.placeholder(name = 'beta', shape = [], dtype = FLOATING_TYPE)
            
            self.acq_inputs = {'z_star' : z_star, 'A' : A, 'max_fun' : max_fun, 'beta' : beta}
            
            x_star = tf.matmul(z_star, tf.transpose(A))
            
            ####### TRANSFORMED VARIABLES #######
            
            length_sq = tf.exp(2 * log_length)
            sigma_sq = tf.exp(2 * log_sigma)
            noise_sq = tf.exp(2 * log_noise)
            mu = tf.zeros_like(y)
            
            ####### GP Likelihood #######
            
            F, L = mvn_likelihood_sqkern(X, y, mu, length_sq, sigma_sq, noise_sq, JITTER_VALUE, FLOATING_TYPE)
            
            ####### x_star dist #######
            
            mu_star, var_star = f_star(x_star, X, y, length_sq, sigma_sq, noise_sq, L, FLOATING_TYPE)
            
            ####### Acqusition function #######
            
            if ACQ_FUN is 'EI':
                F_acq = acq_fun(mu_star, var_star, max_fun, beta, method = ACQ_FUN)
            elif ACQ_FUN is 'UCB':
                F_acq = acq_fun(mu_star, var_star, max_fun, beta, method = ACQ_FUN)
            
            ####### FITTING TRAIN STEP #######
            
            ####### OUTPUTS #######
            
            self.train_f = -F
            self.train_g = tf.stack(tf.gradients(self.train_f, [log_length, log_sigma, log_noise]), 0)
            self.acq_f = F_acq
            self.acq_g = tf.gradients(F_acq, z_star)
            self.mu_star = mu_star
            self.var_star = var_star
            
#            self.outputs = {'OBJ' : OBJ_fit, 'train_fit' : train_fit, 'F_acq' : F_acq, 'mu_star' : mu_star, 'var_star' : var_star}
            
            ####### FITTING TRAIN STEP #######
#            if SEARCH_METHOD is 'grad':
#                phi = log_barrier(x_star, A)
#                
#                OBJ_next = -(t * F_acq + phi)
#                
#                opt_next = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
#                
#                train_next = opt_next.minimize(OBJ_next, var_list = [x_star])
#                
#                self.outputs['train_next'] = train_next