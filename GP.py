#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:41:37 2017

@author: vinnam
"""

import numpy as np
import tensorflow as tf

def mvn_likelihood_sqkern(Z, mu, length_sq, sigma_sq, noise_sq, JITTER_VALUE, FLOATING_TYPE):
    r = tf.reshape(tf.reduce_sum(tf.square(Z), 1), [-1, 1]) # M x 1
    K_uu = tf.exp(-(0.5 / length_sq) * (r - 2 * tf.matmul(Z, tf.transpose(Z)) + tf.transpose(r))) # Check
    L = tf.cholesky(sigma_sq * K_uu + (JITTER_VALUE + noise_sq) * tf.diag(tf.squeeze(tf.ones_like(r))))
    
    d = Z - mu
    alpha = tf.matrix_triangular_solve(L, d, lower=True)
    num_col = 1 if tf.rank(Z) == 1 else tf.shape(Z)[1]
    num_col = tf.cast(num_col, FLOATING_TYPE)
    num_dims = tf.cast(tf.shape(Z)[0], FLOATING_TYPE)
    ret = - 0.5 * num_dims * num_col * np.log(2 * np.pi)
    ret += - num_col * tf.reduce_sum(tf.log(tf.diag_part(L)))
    ret += - 0.5 * tf.reduce_sum(tf.square(alpha))

    return ret, L

def f_star(x_star, X, y, length_sq, sigma_sq, noise_sq, L, MIN_VAR, FLOATING_TYPE):
    # x_star : 1 x D
    # l : N x D
        
    l = sigma_sq * tf.exp((-0.5 / length_sq) * tf.reshape(tf.reduce_sum(tf.square(X - x_star), axis = 1), [-1, 1]))
    
    L_inv_l = tf.matrix_triangular_solve(L, l, lower = True)
    L_inv_y = tf.matrix_triangular_solve(L, y, lower = True)
    
    mu = tf.squeeze(tf.matmul(tf.transpose(L_inv_l), L_inv_y))
    var = sigma_sq - tf.reduce_sum(tf.square(L_inv_l))
    var = tf.clip_by_value(var, clip_value_min = tf.constant(MIN_VAR), clip_value_max = tf.constant(np.finfo(FLOATING_TYPE).max))
    
    return mu, var
        
def acq_fun(mu_star, var_star, max_fun, method):
    if method is 'EI':
        std_star = tf.sqrt(var_star)
        dist = tf.contrib.distributions.Normal(mu = mu_star, sigma = std_star)
        diff = (mu_star - max_fun)
        Z = diff / tf.sqrt(var_star)
        EI = diff * dist.cdf(Z) + std_star * dist.pdf(Z)
        return EI
        
def log_barrier(x_star, A):
    Ax = tf.matmul(A, tf.transpose(x_star))
    b = tf.ones_like(Ax)
    
    return tf.reduce_sum(tf.log(b - Ax) + tf.log(Ax + b)) # FOR MAXIMIZATION
    
def hit_and_run(x_star, A, b, Iter = 100):
    D = np.shape(b)[0]
    
    for i in xrange(Iter):
        Ax = np.matmul(A, np.transpose(x_star))
        LEFT = -b - Ax
        RIGHT = b - Ax
        
        alpha = np.random.normal(size = [D, 1])
        A_alpha = np.matmul(A, alpha)
        l = LEFT / A_alpha 
        r = RIGHT / A_alpha
        
        pos = A_alpha > 0
        neg = A_alpha < 0
        
        theta_min = np.max(np.concatenate((l[pos], r[neg])))
        theta_max = np.min(np.concatenate((r[pos], l[neg])))
        
        theta = np.random.uniform(low = theta_min, high = theta_max)
        
        x_star = x_star + theta * np.transpose(alpha)
    return x_star

class GP:
    def __init__(self, D, FLOATING_TYPE = 'float32', JITTER_VALUE = 1e-5, LEARNING_RATE = 1e-1, MIN_VAR = 1e-8, ACQ_FUN = 'EI', SEARCH_METHOD = 'grad'):
        self.FLOATING_TYPE = FLOATING_TYPE
        self.SEARCH_METHOD = SEARCH_METHOD
        
        self.D = D
        
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            ####### VARIABLES #######
            
            X = tf.placeholder(name = 'X', shape = [None, D], dtype = FLOATING_TYPE)
            y = tf.placeholder(name = 'y', shape = [None, 1], dtype = FLOATING_TYPE)
            log_length = tf.get_variable(name = 'log_length', initializer = tf.random_normal_initializer(), shape = [], dtype = FLOATING_TYPE)
            log_sigma = tf.get_variable(name = 'log_sigma', initializer = tf.random_normal_initializer(), shape = [], dtype = FLOATING_TYPE)
            log_noise = tf.get_variable(name = 'log_noise', initializer = tf.random_normal_initializer(), shape = [], dtype = FLOATING_TYPE)
            
            if SEARCH_METHOD is 'grad':
                x_star = tf.get_variable(name = 'x_star', initializer = tf.random_normal_initializer(), shape = [1, D], dtype = FLOATING_TYPE)
            elif SEARCH_METHOD is 'random':
                x_star = tf.placeholder(name = 'x_star', shape = [1, D], dtype = FLOATING_TYPE)
            
            
            max_fun = tf.placeholder(name = 'max_fun', shape = [], dtype = FLOATING_TYPE)
            
            A = tf.placeholder(name = 'A', shape = [None, D], dtype = FLOATING_TYPE)
            t = tf.placeholder(name = 't', shape = [], dtype = FLOATING_TYPE)
            self.inputs = {'X' : X,
                           'y' : y,
                           'log_length' : log_length,
                           'log_sigma' : log_sigma,
                           'log_noise' : log_noise,
                           'x_star' : x_star,
                           'A' : A,
                           't' : t,
                           'max_fun' : max_fun}
            
            ####### TRANSFORMED VARIABLES #######
            
            length_sq = tf.exp(2 * log_length)
            sigma_sq = tf.exp(2 * log_sigma)
            noise_sq = tf.exp(2 * log_noise)
            mu = tf.zeros_like(y)
            
            ####### GP Likelihood #######
            
            F, L = mvn_likelihood_sqkern(y, mu, length_sq, sigma_sq, noise_sq, JITTER_VALUE, FLOATING_TYPE)
            
            ####### x_star dist #######
            
            mu_star, var_star = f_star(x_star, X, y, length_sq, sigma_sq, noise_sq, L, MIN_VAR, FLOATING_TYPE)
            
            ####### Acqusition function #######
            
            F_acq = acq_fun(mu_star, var_star, max_fun, method = ACQ_FUN)
            
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
            
            
    def fitting(self, data, Iter = 500, init_method = 'fix'):
        ####### INIT VALUES OF PARAMETERS #######
        
        X = data['X']
        y = data['y']
        
        var_y = np.var(y)
        
        init_value = {'log_sigma' : 0.5 * np.log(var_y), 'log_noise' : 0.5 * np.log(var_y / 100)}
        
        self.fitted_params = {}
        
        for key in init_value.keys():
            self.fitted_params[key] = init_value[key]
        
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            
            if init_method is 'fix':
                for key in init_value.keys():
                    sess.run(self.inputs[key].assign(init_value[key]))
            
            feed_dict = {self.inputs['X'] : X,
                         self.inputs['y'] : y}
            
            obj = sess.run(self.outputs['OBJ'], feed_dict)
            
            ####### TRAIN_STEP #######
            
            print('TRAIN_STEP')
            
            for i in xrange(Iter):
                try:
                    _, new_obj = sess.run([self.outputs['train_fit'], self.outputs['OBJ']], feed_dict)
                    
                    if new_obj < obj:
                        obj = new_obj
                        for key in self.fitted_params.keys():
                            self.fitted_params[key] = self.inputs[key].eval()
                    
                    if i % 10 is 0:
                        print i, new_obj
                        
                except Exception as inst:
                    print inst
                    break
        
        
    def finding_next(self, data, t0 = 10, mu = 2, Iter_search = 10, Iter_outer = 10, Iter_inner = 100, Iter_random = 10000, Iter_sample = 100):
        if self.SEARCH_METHOD is 'grad':
            
            A = data['A']
            D = self.D
            b = np.ones([D, 1])
            
            with tf.Session(graph=self.graph) as sess:
                feed_dict = {self.inputs['X'] : X,
                             self.inputs['y'] : y,
                             self.inputs['A'] : A,
                             self.inputs['t'] : t0}
                
                x_star = np.zeros([1, D], dtype = self.FLOATING_TYPE)
                
                for n_search in xrange(Iter_search):
                    ## x_star INIT
                    x_star = hit_and_run(x_star, A, b, Iter = Iter_sample)
                    
                    ## OUTER t INIT
                    feed_dict[self.inputs['t']] = t0
                    
                    ###### INTERIOR POINT METHOD ######
                    
                    ## OUTER LOOP
                    for n_outer in xrange(Iter_outer):
                        ## INITIALIZE
                        sess.run(tf.global_variables_initializer())
                    
                        ## FITTED PARAMS INIT
                        for key in self.fitted_params.keys():
                            sess.run(self.inputs[key].assign(self.fitted_params[key]))
                            
                        ## x_star assign
                        sess.run(self.inputs['x_star'].assign(x_star))
                        
                        ## INNER SEARCH
                        obj = sess.run(self.outputs['F_acq'], feed_dict)
                        
                        for n_inner in xrange(Iter_inner):
                            _, newobj = sess.run([self.outputs['train_next'], self.outputs['F_acq']], feed_dict)
                            
                            if newobj > obj:
                                obj = newobj
                                x_star = sess.run(self.inputs['x_star'])
                                
                        ## INCREASE t
                        feed_dict[self.inputs['t']] = self.inputs['t'] * mu

            return x_star
        
        elif self.SEARCH_METHOD is 'random':
            A = data['A']
            D = self.D
            b = np.ones([D, 1])
            
            next_x = np.zeros([1, D], dtype = self.FLOATING_TYPE)
            
            with tf.Session(graph=self.graph) as sess:
                feed_dict = {self.inputs['X'] : X,
                             self.inputs['y'] : y,
                             self.inputs['A'] : A}
            
                x_star = np.zeros([1, D], dtype = self.FLOATING_TYPE)
                
                obj = np.finfo(self.FLOATING_TYPE).min
                
                for n_search in xrange(Iter_random):
                    x_star = hit_and_run(x_star, A, b, Iter = Iter_sample)
                    
                    feed_dict[self.intputs['x_star']] = x_star
                    
                    newobj = sess.run(self.outputs['F_acq'], feed_dict)
                    
                    if newobj > obj:
                        obj = newobj
                        next_x = x_star
                    
            return next_x

from functions import sinc_simple
import matplotlib.pyplot as plt

fun = sinc_simple()

D = fun.dim()
A = np.array([1.0], dtype = 'float32')
N = 2
init_x = np.random.uniform(-1, 1, size = [N])
init_y = fun.evaluate(init_x)

data = {'X' : np.reshape(init_x, [N, D]), 'y' : np.reshape(init_y, [N, 1]), 'D' : D, 'A' : A}

gp = GP(D)
gp.fitting(data)

gp.fitted_params

plt.plot(np.linspace(-1,1), np.squeeze(fun.evaluate(np.linspace(-1,1))))
plt.scatter(data['X'], data['y'])


import numpy as np

N = 500
D = 1
X = np.random.uniform(size = [N, D])
length = 0.5
noise = 0.001
r = np.reshape(np.sum(np.square(X), axis = 1), [-1, 1])
d = r - 2 * np.matmul(X, np.transpose(X)) + np.transpose(r)
L = np.linalg.cholesky(np.exp(-(0.5 / np.square(length)) * d) + np.square(noise) * np.eye(N))
y = np.matmul(L, np.random.normal(size = [N, 1]))
data = {'X' : X, 'y' : y}

gp = GP(D)
gp.fitting(data)

for param in gp.fitted_params:
    print param, np.exp(gp.fitted_params[param])
    
import matplotlib.pyplot as plt
plt.scatter(X, y)


############### HITANDRUN TEST #################

A = np.random.normal(size = [2, 2])
b = np.ones([2, 1])
N = 2
x = np.reshape(np.linspace(-N, N), [-1, 1])
Ax = np.transpose(A[:, 0] * x)

y1 = (b - Ax) / np.reshape(A[:, 1], [-1, 1])
y2 = (-b - Ax) / np.reshape(A[:, 1], [-1, 1])

x_star = np.reshape(np.array([0., 0.]), [1, -1])
result = []
result.append(x_star)

for i in xrange(100):
    x_star = hit_and_run(x_star, A, b, 2)
    result.append(x_star)
result = np.squeeze(result)    

plt.plot(x, y1[0, :])
plt.plot(x, y1[1, :])
plt.plot(x, y2[0, :])
plt.plot(x, y2[1, :])
plt.scatter(result[:, 0], result[:, 1])