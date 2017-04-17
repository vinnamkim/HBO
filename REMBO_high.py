#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:56:00 2017

@author: vinnam
"""

from GP import GP
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import preprocessing
import settings
#import functions
#from util_func import X_to_Z, Z_to_Xhat

class REMBO:
    def __init__(self, fun, K, N, ACQ_FUN, SEARCH_METHOD, iter_fit):
        D = fun.D
        
        data = {}
        data['A'] = np.eye(D, dtype = settings.dtype)
        data['b'] = np.sqrt(D) * np.ones(shape = [D, 1], dtype = settings.dtype)
        
        W = np.random.normal(size = [D, K]).astype(dtype = settings.dtype)
        
        data['Z'] = np.random.uniform(low = -np.sqrt(D), high = np.sqrt(D), size = [N, K]).astype(dtype = settings.dtype)
        data['X'], data['y'] = fun.evaluate(np.matmul(data['Z'], np.transpose(W)))        
        
#        data['X'] = np.random.uniform(low = -1.0, high = 1.0, size = [N, D]).astype(dtype = settings.dtype)
#        
#        data['X'], data['y'] = fun.evaluate(data['X'])

        data['max_fun'] = np.max(data['y'])
        
        scaler = preprocessing.MinMaxScaler((-1,1))
        
        data['y_scaled'] = scaler.fit_transform(data['y'])
        data['scaled_max_fun'] = np.array(1.0, dtype = settings.dtype)
        
#        types = ['Z', 'y_scaled', 'scaled_max_fun']
        types = ['X', 'y', 'max_fun']
        
        data['beta'] = fun.beta(data)
        
        gp = GP(D, ACQ_FUN = ACQ_FUN, SEARCH_METHOD = SEARCH_METHOD)
        
        gp.fitting(data, types, iter_fit)
        
        self.gp = gp
        self.D = D
        self.K = K
        self.data = data
        self.fun = fun
        self.W = W
        self.types = types
        self.scaler = scaler
    
    def iterate(self, iter_fit, iter_next):
        data = self.data
        fun = self.fun
        W = self.W
        types = self.types
        scaler = self.scaler
        gp = self.gp
        D = self.D
        K = self.K
        
        z_star = np.random.uniform(-np.sqrt(D), np.sqrt(D), size = [iter_next, K]).astype(dtype = settings.dtype)
        
        x_star = np.matmul(z_star, W.transpose())
        
#        x_star = np.clip(x_star, -1.0, 1.0)
        
        idx, obj = gp.find_next(data, types, x_star)
        
        next_x, next_y = fun.evaluate(x_star[idx].reshape([1, -1]))
        next_z = np.reshape(z_star[idx], [1, -1])
        
        data['Z'] = np.append(data['Z'], next_z, axis = 0)
        data['X'] = np.append(data['X'], next_x, axis = 0)
        data['y'] = np.append(data['y'], next_y, axis = 0)
        
        data['y_scaled'] = scaler.fit_transform(data['y_scaled'])
        
        data['beta'] = fun.beta(data)
        
        gp.fitting(data, types, iter_fit)
        
        return next_x
#        
#fun = functions.sinc_simple10()
#R = REMBO(fun, 1, 100, ACQ_FUN = 'UCB', SEARCH_METHOD = 'random', iter_fit = 500)
#
#for i in xrange(10):
#    data = R.data
#    gp = R.gp
#    W = R.W
#    
#    fx = np.linspace(-np.sqrt(R.D),np.sqrt(R.D), 100).reshape([-1, 1])
#    fx_high = np.matmul(W, fx.transpose()).transpose()
#    fy = fun.evaluate(fx_high)[1]
#
#    mu, var, EI = gp.test(data, R.types, fx_high)
#    
#    EI_scaled = preprocessing.MinMaxScaler((np.min(fy),np.max(fy))).fit_transform(EI.reshape([-1, 1]))
#                                          
#    next_x = R.iterate(500, 10000)
#    
#    plt.figure()
#    plt.plot(fx, fy)
#    plt.scatter(data['Z'], data['y'])    
#    plt.plot(fx, EI_scaled, '-.')
#    plt.plot(fx, mu, 'k')
#    plt.plot(fx, mu + np.sqrt(var), 'k:')
#    plt.plot(fx, mu - np.sqrt(var), 'k:')
#    plt.scatter(data['Z'][-1], np.min(data['y']), marker = 'x', color = 'g')
#    plt.title('N is ' + str(len(data['y'])))
#    plt.show()
        
#
#
#R.iterate(500, 10000)
#
#
#R.data
#
#    xx = np.linspace(-1, 1)
#    mu, var, EI = gp.test(data, np.reshape(xx, [-1,1]))
#    
#    
#    plt.figure()
#    fx = np.linspace(-1,1)
#    fy = np.squeeze(fun.evaluate(np.linspace(-1,1)))
#    plt.plot(fx, fy)
#    plt.scatter(data['X'], data['y'])
#    plt.plot(xx, (max(fy) - min(fy)) / (max(EI) - min(EI)) * (EI) + min(fy), '-.')
#    plt.plot(xx, mu, 'k')
#    plt.plot(xx, mu + 2 * np.sqrt(var), 'k:')
#    plt.plot(xx, mu - 2 * np.sqrt(var), 'k:')
#    plt.scatter(next_x, np.mean(data['y']), marker = 'x')
#    plt.title('N is ' + str(len(data['y'])))
#    plt.show()    
#    fun.update(next_x, data)
#
#N = 10
#D = 2
#K = 1
#
#test = preprocessing.MinMaxScaler((-1,1))
#
#a = np.random.normal(size = [100, 1])
#
#b = test.fit_transform(a)
#
#
#
#try:
#    data['y']
#except:
#    data['y'] = {}