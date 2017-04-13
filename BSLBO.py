#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:56:00 2017

@author: vinnam
"""

from MHGP import MHGP
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import settings
import functions

class BSLBO:
    def __init__(self, fun, K, N, M, ACQ_FUN):
#        N = 10
#        M = 10
#        K = 1
#        fun = functions.sinc_simple2()
#        ACQ_FUN = 'EI'
        
        D = fun.D
        data = {}
        
        data['X'], data['y'] = fun.evaluate(np.random.uniform(low = -1.0, high = 1.0, size = [N, D]).astype(settings.dtype))
        data['max_fun'] = np.max(data['y'])
        scaler = preprocessing.MinMaxScaler((-1,1))
                
        data['y_scaled'] = scaler.fit_transform(data['y'])
        data['scaled_max_fun'] = np.array(1.0, dtype = settings.dtype)
        data['beta'] = fun.beta(data)
                
        types = ['X', 'y_scaled', 'scaled_max_fun']
#        types = ['X', 'y', 'max_fun']
        
        gp = MHGP(M, K, D, ACQ_FUN = ACQ_FUN)
        
        gp.fitting(data, types)
        
        self.K = K
        self.D = D
        self.fun = fun
        self.data = data
        self.types = types
        self.gp = gp
        self.scaler = scaler
        self.ACQ_FUN = ACQ_FUN
        self.M = M
    
    def iterate(self, num_sample, effective_dims):
        M = self.M
        D = self.D
        K = self.K
        data = self.data
        fun = self.fun
        types = self.types
        scaler = self.scaler
        gp = self.gp
        ACQ_FUN = self.ACQ_FUN
        
        W = gp.fitted_params['mu']
        WT = np.transpose(W)
        WWT = np.matmul(W, WT)
        
        B = np.transpose(np.linalg.solve(WWT, W)) # D x K
        
        next_z = gp.finding_next(data, types, num_sample, effective_dims)
        next_x, next_y = fun.evaluate(np.matmul(next_z, np.transpose(B)))
        
        data['X'] = np.append(data['X'], next_x, axis = 0)
        data['y'] = np.append(data['y'], next_y, axis = 0)
        
        data['y_scaled'] = scaler.fit_transform(data['y'])
        
        data['max_fun'] = np.max(data['y'])
        data['beta'] = fun.beta(data)
        
        #M = len(data['y'])
        
        self.gp = MHGP(M, K, D, ACQ_FUN = ACQ_FUN)
        
        self.gp.fitting(data, types)
        
        self.data = data
        
        return next_x
        
fun = functions.sinc_simple2()
R = BSLBO(fun, 1, 2, 1, ACQ_FUN = 'EI')

for i in xrange(3):
    data = R.data
    gp = R.gp
    W = gp.fitted_params['mu']
    WT = np.transpose(W)
    WWT = np.matmul(W, WT)
    B = np.transpose(np.linalg.solve(WWT, W)) # D x K
    D = fun.D
    C = np.matmul(W, np.ones([D, 1]))
    fx = np.linspace(-np.sqrt(D) * np.abs(np.squeeze(C)), np.sqrt(D) * np.abs(np.squeeze(C)), num = 100).reshape([100, 1])
    fy = fun.evaluate(np.matmul(fx, np.transpose(B)))[1]
    
    mu, var, EI = gp.test(data, R.types, fx)
    
    pfxpp = np.concatenate([fx, np.reshape(mu, [-1, 1]), np.reshape(var, [-1, 1]), np.reshape(EI, [-1, 1])], axis = 1)
    pfxpp = pfxpp[pfxpp[:, 0].argsort()]
    next_x = R.iterate(10000, np.array([1.]))
    
    plt.figure()
    plt.plot(fx, fy)
    plt.scatter(np.matmul(data['X'], W.transpose()), data['y'])
    plt.plot(pfxpp[:, 0], (np.max(fy) - np.min(fy)) * (pfxpp[:, 3]) / (np.max(EI) - np.min(EI)) + min(fy), '-.')
    plt.plot(pfxpp[:, 0], pfxpp[:, 1], 'k')
    plt.plot(pfxpp[:, 0], pfxpp[:, 1] + np.sqrt(pfxpp[:, 2]), 'k:')
    plt.plot(pfxpp[:, 0], pfxpp[:, 1] - np.sqrt(pfxpp[:, 2]), 'k:')
    plt.scatter(np.matmul(data['X'][-1], W.transpose()), np.min(data['y']), marker = 'x', color = 'g')
    plt.title('N is ' + str(len(data['y'])))
    plt.show()
    
from scipy.optimize import linprog
    
def find_enclosingbox(A, b):
    m = A.shape[0]
    n = A.shape[1]
    
    c_plus = np.zeros([n, n])
    c_minus = np.zeros([n, n])
    
    for i in xrange(n):
        c_plus
        
        linprog()
        