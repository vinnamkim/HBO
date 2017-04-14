#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:41:37 2017

@author: vinnam
"""

import numpy as np
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
    def __init__(self, D, LEARNING_RATE = 1e-1, ACQ_FUN = 'EI', SEARCH_METHOD = 'random'):
        self.FLOATING_TYPE = settings.dtype
        self.JITTER_VALUE = settings.jitter
        self.SEARCH_METHOD = SEARCH_METHOD
        self.ACQ_FUN = ACQ_FUN
        
        FLOATING_TYPE = self.FLOATING_TYPE
        JITTER_VALUE = self.JITTER_VALUE
        
        self.fitted_params = {'log_length' : None, 'log_sigma' : None, 'log_noise' : None}
        
        self.D = D
        
        self.graph = tf.Graph()
        
        self.x_init = np.zeros([1, D], dtype = self.FLOATING_TYPE)
        
        with self.graph.as_default():
            ####### VARIABLES #######
            
            X = tf.placeholder(name = 'X', shape = [None, D], dtype = FLOATING_TYPE)
            y = tf.placeholder(name = 'y', shape = [None, 1], dtype = FLOATING_TYPE)
            log_length = tf.get_variable(name = 'log_length', initializer = tf.random_normal_initializer(), shape = [], dtype = FLOATING_TYPE)
            log_sigma = tf.get_variable(name = 'log_sigma', initializer = tf.random_normal_initializer(), shape = [], dtype = FLOATING_TYPE)
            log_noise = tf.get_variable(name = 'log_noise', initializer = tf.random_normal_initializer(), shape = [], dtype = FLOATING_TYPE)
            
            if SEARCH_METHOD is 'grad':
                x_star = tf.get_variable(name = 'x_star', initializer = tf.random_normal_initializer(), shape = [None, D], dtype = FLOATING_TYPE)
            elif SEARCH_METHOD is 'random':
                x_star = tf.placeholder(name = 'x_star', shape = [None, D], dtype = FLOATING_TYPE)
            
            A = tf.placeholder(name = 'A', shape = [None, D], dtype = FLOATING_TYPE)
            b = tf.placeholder(name = 'b', shape = [None, 1], dtype = FLOATING_TYPE)
            t = tf.placeholder(name = 't', shape = [], dtype = FLOATING_TYPE)
            max_fun = tf.placeholder(name = 'max_fun', shape = [], dtype = FLOATING_TYPE)
            beta = tf.placeholder(name = 'beta', shape = [], dtype = FLOATING_TYPE)
            self.inputs = {'X' : X, 'y' : y}
            self.params = {'log_length' : log_length, 'log_sigma' : log_sigma, 'log_noise' : log_noise}
            self.acq_inputs = {'x_star' : x_star, 'A' : A, 'b' : b, 't' : t, 'max_fun' : max_fun, 'beta' : beta}
                
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
            
            opt_fit = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
            
            OBJ_fit = -F
            
            train_fit = opt_fit.minimize(OBJ_fit)
            
            ####### OUTPUTS #######
            
            self.outputs = {'F' : F, 'chol' : L, 'OBJ' : OBJ_fit , 'train_fit' : train_fit, 'F_acq' : F_acq, 'mu_star' : mu_star, 'var_star' : var_star}
            
            ####### FITTING TRAIN STEP #######
            if SEARCH_METHOD is 'grad':
                phi = log_barrier(x_star, A)
                
                OBJ_next = -(t * F_acq + phi)
                
                opt_next = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
                
                train_next = opt_next.minimize(OBJ_next, var_list = [x_star])
                
                self.outputs['train_next'] = train_next
            
    def fitting(self, data, types, Iter = 500, init_method = 'fix'):
        ####### INIT VALUES OF PARAMETERS #######        
        X = data[types[0]]
        y = data[types[1]]
        
        var_y = np.var(y)
        
        init_value = {'log_sigma' : 0.5 * np.log(var_y), 'log_noise' : 0.5 * np.log(var_y / 100)}
        
        for key in init_value.keys():
            self.fitted_params[key] = init_value[key]
        
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            
            if init_method is 'fix':
                for key in init_value.keys():
                    sess.run(self.params[key].assign(init_value[key]))
            
            feed_dict = {self.inputs['X'] : X,
                         self.inputs['y'] : y}
            
            obj = sess.run(self.outputs['OBJ'], feed_dict)
            
            ####### TRAIN_STEP #######
            
            train_dict = {'train_fit' : self.outputs['train_fit'],
                          'OBJ' : self.outputs['OBJ']}
            
            train_dict.update(self.params)
            
            print('TRAIN_STEP')
            
            for i in xrange(Iter):
                try:
                    train_step = sess.run(train_dict, feed_dict)
                    
                    if train_step['OBJ'] < obj:
                        obj = train_step['OBJ']
                        for key in self.fitted_params.keys():
                            self.fitted_params[key] = train_step[key]
                    
                    if i % 100 is 0:
                        print i, train_step['OBJ']
                        
                except Exception as inst:
                    print inst
                    break
        
    def test(self, data, types, xx):
        X = data[types[0]]
        y = data[types[1]]
        A = data['A']
        max_fun = data[types[2]]
        beta = data['beta']
        x_star = np.reshape(xx, [-1, self.D])
        
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            
            feed_dict = {self.inputs['X'] : X, self.inputs['y'] : y, self.acq_inputs['A'] : A, self.acq_inputs['max_fun'] : max_fun, self.acq_inputs['beta'] : beta}
            
            for key in self.fitted_params.keys():
                sess.run(self.params[key].assign(self.fitted_params[key]))
            
            if self.SEARCH_METHOD is 'grad':
                sess.run(self.acq_inputs['x_star'].assign(x_star))
            elif self.SEARCH_METHOD is 'random':
                feed_dict[self.acq_inputs['x_star']] = x_star
            
            mu, var, F_acq = sess.run([self.outputs['mu_star'], self.outputs['var_star'], self.outputs['F_acq']], feed_dict)
        
        return [mu, var, F_acq]
    
    def finding_next(self, data, types, t0 = 10., gamma = 2., Iter_search = 10, Iter_outer = 10, Iter_inner = 100, Iter_random = 1000, N_burnin = 100):
        X = data[types[0]]
        y = data[types[1]]
        A = data['A']
        max_fun = data[types[2]]
        beta = data['beta']
        D = self.D
        b = data['b']
        
        if self.SEARCH_METHOD is 'grad':
            with tf.Session(graph=self.graph) as sess:
                feed_dict = {self.inputs['X'] : X,
                             self.inputs['y'] : y,
                             self.acq_inputs['A'] : A,
                             self.acq_inputs['b'] : b,
                             self.acq_inputs['t'] : t0,
                             self.acq_inputs['max_fun'] : max_fun,
                             self.acq_inputs['beta'] : beta}
                
                x_init = self.x_init
                x_list = hit_and_run(x_init, A, b, N_burnin, Iter_random)
                self.x_init = np.reshape(x_list[-1], [1, D])
                
                np.random.shuffle(x_list)
                
                ## x_star INIT
                x_star = x_list[Iter_search]
                
                ## OUTER t INIT
                t = t0
                feed_dict[self.inputs['t']] = t
                
                ###### INTERIOR POINT METHOD ######
                
                ## OUTER LOOP
                for n_outer in xrange(Iter_outer):
                    ## INITIALIZE
                    sess.run(tf.global_variables_initializer())
                
                    ## FITTED PARAMS INIT
                    for key in self.fitted_params.keys():
                        sess.run(self.params[key].assign(self.fitted_params[key]))
                        
                    ## x_star assign
                    sess.run(self.acq_inputs['x_star'].assign(x_star))
                    
                    ## INNER SEARCH
                    #obj = sess.run(self.outputs['F_acq'], feed_dict = feed_dict)
                    
                    for n_inner in xrange(Iter_inner):
                        sess.run(self.outputs['train_next'], feed_dict)
                        #_, newobj = sess.run([self.outputs['train_next'], self.outputs['F_acq']], feed_dict)
                        
                        #if newobj > obj:
                        #    obj = newobj
                        #    x_star = sess.run(self.inputs['x_star'])
                    
                    x_star = sess.run(self.acq_inputs['x_star'])
                    
                    ## INCREASE t
                    feed_dict[self.acq_inputs['t']] = t * gamma
                    
            obj = sess.run(self.outputs['F_acq'], feed_dict = feed_dict)
            
            return np.reshape(x_star[np.argmax(obj)], [1, D])
        
        elif self.SEARCH_METHOD is 'random':
#            x_init = self.x_init
#            x_star = hit_and_run(x_init, A, b, N_burnin, Iter_random)
            x_star = sample_enclosingbox(A, b, Iter_random)

            feed_dict = {self.inputs['X'] : X,
                         self.inputs['y'] : y,
                         self.acq_inputs['x_star'] : x_star,
                         self.acq_inputs['max_fun'] : max_fun,
                         self.acq_inputs['beta'] : beta}
            
            with tf.Session(graph=self.graph) as sess:
                ## INITIALIZE
                sess.run(tf.global_variables_initializer())
                
                ## FITTED PARAMS INIT
                for key in self.fitted_params.keys():
                    sess.run(self.params[key].assign(self.fitted_params[key]))
                try:    
                    obj = sess.run(self.outputs['F_acq'], feed_dict = feed_dict)
                except Exception as inst:
                    print inst
                    
#            self.x_init = np.reshape(x_star[-1], [-1, D])
            
            next_x = x_star[np.argmax(obj)]
                    
            return np.reshape(next_x, [1, D])


