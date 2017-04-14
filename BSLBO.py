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
        
        W = gp.fitted_params['mu'].transpose()
        WT = np.transpose(W)
        WTW = np.matmul(WT, W)
        B = np.transpose(np.linalg.solve(WTW, WT)) # D x K
        
        next_z = gp.finding_next(data, types, B, num_sample, effective_dims)
        next_x, next_y = fun.evaluate(np.matmul(next_z, np.transpose(B)))
        
        data['X'] = np.append(data['X'], next_x, axis = 0)
        data['y'] = np.append(data['y'], next_y, axis = 0)
        
        data['y_scaled'] = scaler.fit_transform(data['y'])
        
        data['max_fun'] = np.max(data['y'])
        data['beta'] = fun.beta(data)
        
        M = len(data['y'])
        
        self.gp = MHGP(M, K, D, ACQ_FUN = ACQ_FUN)
        
        self.gp.fitting(data, types)
        
        self.data = data
        self.M = M
        
        return next_x
        
    

def X_to_Z(X, W):
    return np.matmul(X, W)  # X : N x D,  W : D x K
                        
def Z_to_Xhat(Z, W):
    WT = np.transpose(W)
    WTW = np.matmul(WT, W)
    B = np.transpose(np.linalg.solve(WTW, WT)) # D x K
    
    return np.matmul(Z, B.transpose()) # Z : N x K,  B : D x K
                    

#fun = functions.sinc_simple2()
fun = functions.sinc_simple10()
R = BSLBO(fun, 1, 10, 10, ACQ_FUN = 'UCB')

for i in xrange(10):
    data = R.data
    gp = R.gp
    W = gp.fitted_params['mu'].transpose()
    
    D = fun.D
    C = np.matmul(W.transpose(), np.ones([D, 1]))
    fx = np.linspace(-np.sqrt(D) * np.abs(np.squeeze(C)), np.sqrt(D) * np.abs(np.squeeze(C)), num = 100).reshape([100, 1])
    fy = fun.evaluate(Z_to_Xhat(fx, W))[1]
    
    mu, var, EI = gp.test(data, R.types, fx)
    
    pfxpp = np.concatenate([fx, np.reshape(mu, [-1, 1]), np.reshape(var, [-1, 1]), np.reshape(EI, [-1, 1])], axis = 1)
    pfxpp = pfxpp[pfxpp[:, 0].argsort()]
    next_x = R.iterate(10000, np.array([True]))
    
#    plt.figure()
#    plt.plot(fx, fy)
#    plt.scatter(X_to_Z(data['X'], W), data['y'])
#
#    plt.plot(fx, fun.evaluate(Z_to_Xhat(fx, fun.W))[1])
#    plt.scatter(X_to_Z(data['X'], fun.W), data['y'])
#    
    plt.figure()
    plt.plot(fx, fy)
    plt.scatter(X_to_Z(data['X'], W), data['y'])
#    plt.plot(pfxpp[:, 0], (np.max(fy) - np.min(fy)) * (pfxpp[:, 3]) / (np.max(EI) - np.min(EI)) + min(fy), '-.')
    plt.plot(pfxpp[:, 0], pfxpp[:, 1], 'k')
    plt.plot(pfxpp[:, 0], pfxpp[:, 1] + np.sqrt(pfxpp[:, 2]), 'k:')
    plt.plot(pfxpp[:, 0], pfxpp[:, 1] - np.sqrt(pfxpp[:, 2]), 'k:')
    plt.scatter(np.matmul(data['X'][-1], W), np.min(data['y']), marker = 'x', color = 'g')
    plt.title('N is ' + str(len(data['y'])))
    plt.show()
    
    
                    