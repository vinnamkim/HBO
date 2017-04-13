#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:56:00 2017

@author: vinnam
"""

from GP import GP
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import settings
import functions

class REMBO:
    def __init__(self, fun, K, N, ACQ_FUN, SEARCH_METHOD, iter_fit):
        D = fun.D
        
        data = {}
        data['A'] = np.eye(K, dtype = settings.dtype)
        data['b'] = np.sqrt(D) * np.ones(shape = [K, 1], dtype = settings.dtype)
        
        W = np.random.normal(size = [D, K]).astype(dtype = settings.dtype)
        
        data['Z'] = np.random.uniform(low = -np.sqrt(D), high = np.sqrt(D), size = [N, K]).astype(dtype = settings.dtype)
        data['X'], data['y'] = fun.evaluate(np.matmul(data['Z'], np.transpose(W)))
        data['max_fun'] = np.max(data['y'])
        
        scaler = preprocessing.MinMaxScaler((-1,1))
        
        data['y_scaled'] = scaler.fit_transform(data['y'])
        data['scaled_max_fun'] = np.array(1.0, dtype = settings.dtype)
        
#        types = ['Z', 'y_scaled', 'scaled_max_fun']
        types = ['Z', 'y', 'max_fun']
        
        data['beta'] = fun.beta(data)
        
        gp = GP(K, ACQ_FUN = ACQ_FUN, SEARCH_METHOD = SEARCH_METHOD)
        
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
        
        next_z = gp.finding_next(data, types, Iter_random = iter_next)
        next_x, next_y = fun.evaluate(np.matmul(next_z, np.transpose(W)))
        
        data['Z'] = np.append(data['Z'], next_z, axis = 0)
        data['X'] = np.append(data['X'], next_x, axis = 0)
        data['y'] = np.append(data['y'], next_y, axis = 0)
        
        data['y_scaled'] = scaler.fit_transform(data['y_scaled'])
        
        data['beta'] = fun.beta(data)
        
        gp.fitting(data, types, iter_fit)
        
        return next_x
        
fun = functions.sinc_simple2()
R = REMBO(fun, 1, 2, ACQ_FUN = 'EI', SEARCH_METHOD = 'random', iter_fit = 500)

for i in xrange(10):
    data = R.data
    gp = R.gp
    W = R.W
    
    fx = np.random.uniform(-np.sqrt(2),np.sqrt(2), [100, 1])
    fy = fun.evaluate(np.matmul(W, fx.transpose()).transpose())[1]
#   fx = np.matmul(fx, fun.W)
    fxfy = np.concatenate([fx, fy], axis = 1)
    fxfy = fxfy[fxfy[:, 0].argsort()]
    
    mu, var, EI = gp.test(data, R.types, fx)
    
    pfxpp = np.concatenate([fx, np.reshape(mu, [-1, 1]), np.reshape(var, [-1, 1]), np.reshape(EI, [-1, 1])], axis = 1)
    pfxpp = pfxpp[pfxpp[:, 0].argsort()]
    next_x = R.iterate(500, 10000)
    
    plt.figure()
    plt.plot(fxfy[:, 0], fxfy[:, 1])
    plt.scatter(data['Z'], data['y'])
    plt.plot(pfxpp[:, 0], (np.max(fy) - np.min(fy)) * (pfxpp[:, 3]) / (np.max(EI) - np.min(EI)) + min(fy), '-.')
    plt.plot(pfxpp[:, 0], pfxpp[:, 1], 'k')
    plt.plot(pfxpp[:, 0], pfxpp[:, 1] + np.sqrt(pfxpp[:, 2]), 'k:')
    plt.plot(pfxpp[:, 0], pfxpp[:, 1] - np.sqrt(pfxpp[:, 2]), 'k:')
    plt.scatter(data['Z'][-1], np.min(data['y']), marker = 'x', color = 'g')
    plt.title('N is ' + str(len(data['y'])))
    plt.show()
        
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