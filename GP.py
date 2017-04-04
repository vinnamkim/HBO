#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:41:37 2017

@author: vinnam
"""

import numpy as np
import tensorflow as tf
import settings

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
    
    return mu, var, l, L_inv_l, L_inv_y
        
def acq_fun(mu_star, var_star, max_fun, method):
    if method is 'EI':
        std_star = tf.sqrt(var_star)
        dist = tf.contrib.distributions.Normal(mu = mu_star, sigma = std_star)
        diff = (mu_star - max_fun)
        Z = diff / tf.sqrt(var_star)
        EI = diff * dist.cdf(Z) + std_star * dist.pdf(Z)
        return EI
        
def log_barrier(x_star, A):
    Ax = tf.transpose(tf.matmul(A, tf.transpose(x_star))) # L x D
    b = tf.ones_like(Ax)
    
    return tf.reduce_sum(tf.log(b - Ax) + tf.log(Ax + b), axis = -1) # FOR MAXIMIZATION L x 1
    
def hit_and_run(x_star, A, b, Burnin = 100, Iter = 1000):
    D = np.shape(b)[0]
    
    for i in xrange(Burnin):
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
        
    X = np.zeros(shape = [Iter, D], dtype = x_star.dtype)
    
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
        
        X[i, :] = x_star
        
    return X

class GP:
    def __init__(self, D, FLOATING_TYPE = settings.dtype, JITTER_VALUE = 1e-5, LEARNING_RATE = 1e-1, ACQ_FUN = 'EI', SEARCH_METHOD = 'grad'):
        self.FLOATING_TYPE = FLOATING_TYPE
        self.SEARCH_METHOD = SEARCH_METHOD
        
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
            
            F, L = mvn_likelihood_sqkern(X, y, mu, length_sq, sigma_sq, noise_sq, JITTER_VALUE, FLOATING_TYPE)
            
            ####### x_star dist #######
            
            mu_star, var_star, l, L_inv_l, L_inv_y = f_star(x_star, X, y, length_sq, sigma_sq, noise_sq, L, FLOATING_TYPE)
            
            ####### Acqusition function #######
            
            F_acq = acq_fun(mu_star, var_star, max_fun, method = ACQ_FUN)
            
            ####### FITTING TRAIN STEP #######
            
            opt_fit = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
            
            OBJ_fit = -F
            
            train_fit = opt_fit.minimize(OBJ_fit)
            
            ####### OUTPUTS #######
            
            self.outputs = {'F' : F, 'chol' : L, 'OBJ' : OBJ_fit , 'train_fit' : train_fit, 'F_acq' : F_acq, 'mu_star' : mu_star, 'var_star' : var_star,
                            'l' : l,
                            'L_inv_l' : L_inv_l,
                            'L_inv_y' : L_inv_y}
            
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
        
        self.fitted_params = {'log_length' : None, 'log_sigma' : None, 'log_noise' : None}
        
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
                    
                    if i % 100 is 0:
                        print i, new_obj
                        
                except Exception as inst:
                    print inst
                    break
        
    def test(self, data, xx):
        X = data['X']
        y = data['y']
        A = data['A']
        max_fun = data['max_fun']
        x_star = np.reshape(xx, [-1, self.D])
        
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            
            feed_dict = {self.inputs['X'] : X, self.inputs['y'] : y, self.inputs['A'] : A, self.inputs['max_fun'] : max_fun}
            
            for key in self.fitted_params.keys():
                sess.run(self.inputs[key].assign(self.fitted_params[key]))
            
            if self.SEARCH_METHOD is 'grad':
                sess.run(self.inputs['x_star'].assign(x_star))
            elif self.SEARCH_METHOD is 'random':
                feed_dict[self.inputs['x_star']] = x_star
            
            mu, var, F_acq = sess.run([self.outputs['mu_star'], self.outputs['var_star'], self.outputs['F_acq']], feed_dict)
        
        return [mu, var, F_acq]
    
    def finding_next(self, data, t0 = 10., gamma = 2., Iter_search = 10, Iter_outer = 10, Iter_inner = 100, Iter_random = 1000, N_burnin = 100):
        X = data['X']
        y = data['y']
        A = data['A']
        max_fun = data['max_fun']
        D = self.D
        b = np.ones([D, 1])
        
        if self.SEARCH_METHOD is 'grad':
            with tf.Session(graph=self.graph) as sess:
                feed_dict = {self.inputs['X'] : X,
                             self.inputs['y'] : y,
                             self.inputs['A'] : A,
                             self.inputs['t'] : t0,
                             self.inputs['max_fun'] : max_fun}
                
                x_star = np.zeros([1, D], dtype = self.FLOATING_TYPE)
                x_list = hit_and_run(x_star, A, b, N_burnin, Iter_random)
                np.random.shuffle(x_list)
                
                for n_search in xrange(Iter_search):
                    ## x_star INIT
                    x_star = np.reshape(x_list[n_search], [1, -1])
                    
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
                            sess.run(self.inputs[key].assign(self.fitted_params[key]))
                            
                        ## x_star assign
                        sess.run(self.inputs['x_star'].assign(x_star))
                        
                        ## INNER SEARCH
                        obj = sess.run(self.outputs['F_acq'], feed_dict = feed_dict)
                        
                        for n_inner in xrange(Iter_inner):
                            _, newobj = sess.run([self.outputs['train_next'], self.outputs['F_acq']], feed_dict)
                            
                            if newobj > obj:
                                obj = newobj
                                x_star = sess.run(self.inputs['x_star'])
                                
                        ## INCREASE t
                        feed_dict[self.inputs['t']] = t * gamma

            return x_star
        
        elif self.SEARCH_METHOD is 'random':
            x_init = self.x_init
            
            x_star = hit_and_run(x_init, A, b, N_burnin, Iter_random)
            
            feed_dict = {self.inputs['X'] : X,
                         self.inputs['y'] : y,
                         self.inputs['A'] : A,
                         self.inputs['x_star'] : x_star,
                         self.inputs['max_fun'] : max_fun}
            
            with tf.Session(graph=self.graph) as sess:
                ## INITIALIZE
                sess.run(tf.global_variables_initializer())
                
                ## FITTED PARAMS INIT
                for key in self.fitted_params.keys():
                    sess.run(self.inputs[key].assign(self.fitted_params[key]))
                    
                obj = sess.run(self.outputs['F_acq'], feed_dict = feed_dict)
            
            self.x_init = np.reshape(x_star[-1], [-1, D])
            
            return np.reshape(x_star[np.argmax(obj)], [-1, D])

from functions import sinc_simple
import matplotlib.pyplot as plt

fun = sinc_simple()
N = 10

data = fun.gen_data(N)
A = np.reshape(np.array([1.0]), [1,1])
data['A'] = A

gp = GP(data['D'], SEARCH_METHOD = 'random')

for i in xrange(3):
    gp.fitting(data, Iter = 500)
    
    xx = np.linspace(-1, 1)
    mu, var, EI = gp.test(data, np.reshape(xx, [-1,1]))
    next_x = gp.finding_next(data)
    
    plt.figure()
    fx = np.linspace(-1,1)
    fy = np.squeeze(fun.evaluate(np.linspace(-1,1)))
    plt.plot(fx, fy)
    plt.scatter(data['X'], data['y'])
    plt.plot(xx, (max(fy) - min(fy)) / (max(EI) - min(EI)) * (EI) + min(fy), '-.')
    plt.plot(xx, mu, 'k')
    plt.plot(xx, mu + 2 * np.sqrt(var), 'k:')
    plt.plot(xx, mu - 2 * np.sqrt(var), 'k:')
    plt.scatter(next_x, np.mean(data['y']), marker = 'x')
    plt.title('N is ' + str(len(data['y'])))
    plt.show()
    
    mu, var, EI = gp.test(data, data['X'])
    plt.scatter(data['X'], mu)
    
    fun.update(next_x, data)

############################

from functions import sinc_simple2
import matplotlib.pyplot as plt

fun = sinc_simple2()
N = 1000

data = fun.gen_data(N)
A = np.eye(2)
data['A'] = A

gp = GP(data['D'], SEARCH_METHOD = 'random')

for i in xrange(3):
    gp.fitting(data, Iter = 500)
    
    fx = np.random.uniform(-1,1, [100, 2])
    fy = fun.evaluate(fx)
    pfx = np.matmul(fx, fun.W)
    fxfy = np.concatenate([pfx, fy], axis = 1)
    fxfy = fxfy[fxfy[:, 0].argsort()]
    
    mu, var, EI = gp.test(data, fx)
    
    pfxpp = np.concatenate([pfx, np.reshape(mu, [-1, 1]), np.reshape(var, [-1, 1]), np.reshape(EI, [-1, 1])], axis = 1)
    pfxpp = pfxpp[pfxpp[:, 0].argsort()]
    
    next_x = gp.finding_next(data, Iter_random = 10000)
    
    plt.figure()
    plt.plot(fxfy[:, 0], fxfy[:, 1])
    plt.scatter(np.matmul(data['X'], fun.W), data['y'])
    plt.plot(pfxpp[:, 0], (np.max(fy) - np.min(fy)) * (pfxpp[:, 3]) / (np.max(EI) - np.min(EI)) + min(fy), '-.')
    plt.plot(pfxpp[:, 0], pfxpp[:, 1], 'k')
    plt.plot(pfxpp[:, 0], pfxpp[:, 1] + np.sqrt(pfxpp[:, 2]), 'k:')
    plt.plot(pfxpp[:, 0], pfxpp[:, 1] - np.sqrt(pfxpp[:, 2]), 'k:')
    plt.scatter(np.matmul(next_x, fun.W), -0.25, marker = 'x', color = 'g')
    plt.title('N is ' + str(len(data['y'])))
    plt.show()
        
    fun.update(next_x, data)

    
#t = t.astype('float32')
xx = np.linspace(-1, 1)
#xx = xx.astype('float32')
#xx = data['X']
mu = np.zeros(len(xx))
var = np.zeros(len(xx))
EI = np.zeros(len(xx))
#LL = np.linalg.cholesky(np.exp(2 * gp.fitted_params['log_sigma']) * np.exp(-(0.5 / np.exp(2 * gp.fitted_params['log_length'])) * np.square(t - np.transpose(t))) + (1e-5 + np.exp(2 * gp.fitted_params['log_noise'])) * np.eye(N))
#LL = LL.astype('float32')

for idx in xrange(len(xx)):
    a,b,c = gp.test(data, np.reshape(xx[idx], [1,1]))
    mu[idx] = a
    var[idx] = b
    EI[idx] = c
#    ll = np.exp(2 * gp.fitted_params['log_sigma']) * np.exp(-(0.5 / np.exp(2 * gp.fitted_params['log_length'])) * np.square(t - np.transpose(np.reshape(xx[idx], [1,1]))))
#    mu[idx] = np.squeeze(np.matmul(np.transpose(np.linalg.solve(LL, ll)), np.linalg.solve(LL, data['y'])))
    
#     mu[idx] = np.squeeze(np.matmul(np.transpose(np.linalg.solve(L, ll)), np.linalg.solve(L, data['y'])))
fx = np.linspace(-1,1)
fy = np.squeeze(fun.evaluate(np.linspace(-1,1)))
plt.plot(fx, fy)
plt.scatter(data['X'], data['y'])
plt.plot(xx, (max(fy) - min(fy)) / (max(EI) - min(EI)) * (EI) + min(fy), '-.')
plt.plot(xx, mu, 'k')
plt.plot(xx, mu + 2 * var, 'k:')
plt.plot(xx, mu - 2 * var, 'k:')

next_x = gp.finding_next(data)
plt.plot(xx, EI)
plt.scatter(next_x, 0)



next_x = gp.finding_next(data)
next_x = gp.finding_next(data)
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
gp.test(data, data['X'][0])

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