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
import SI

def Sample_unitball(N, D, epsilon):
    Y = np.random.normal(size = [N, D]).astype(settings.dtype)
    Y_sum = np.sqrt(np.sum(np.square(Y), axis = 1)).reshape([N, 1])
    X = Y / Y_sum
    
    return X * (1. - epsilon)

class SIBO:
    def __init__(self, fun, K, m_X, m_Phi, ACQ_FUN, SEARCH_METHOD, iter_fit):
        D = fun.D
        
        data = {}
        data['A'] = np.eye(K, dtype = settings.dtype)
        data['b'] = np.sqrt(D) * np.ones(shape = [K, 1], dtype = settings.dtype)
        
        epsilon = settings.SIBO_epsilon
        C2 = settings.SIBO_C2
        
        X_center = Sample_unitball(m_X, D, epsilon)
        X_center, y_center = fun.evaluate(X_center)
        
        Phi = [np.random.uniform(size = [D, m_X]).astype(settings.dtype) for i in xrange(m_Phi)]
        Phi = [1 / np.sqrt(m_Phi).astype(settings.dtype) * (2 * np.round(P) - 1) for P in Phi]
        
        Direction = [fun.evaluate(X_center + epsilon * np.transpose(P)) for P in Phi]
        
        X_direction = np.array([Xy[0] for Xy in Direction]).squeeze()
        y_direction = np.array([Xy[1] for Xy in Direction]).squeeze()
            
        y_SI = 1 / epsilon * np.sum(y_direction - y_center.transpose(), axis = 1).reshape([-1, 1])
        
        W = SI.SI(Phi, y_SI, epsilon, K, C2)
        
#        data['X'] = np.concatenate((X_center, X_direction.reshape([-1, D])), axis = 0)
#        data['y'] = np.concatenate((y_center, y_direction.reshape([-1, 1])), axis = 0)
        
        data['X'] = X_center
        data['y'] = y_center
        
        data['Z'] = np.matmul(data['X'], W)
        data['max_fun'] = np.max(data['y'])
        
        scaler = preprocessing.MinMaxScaler((-1,1))
        
        data['y_scaled'] = scaler.fit_transform(data['y'])
        data['scaled_max_fun'] = np.array(1.0, dtype = settings.dtype)
        
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

R = SIBO(fun, 1, 5, 2, ACQ_FUN = 'EI', SEARCH_METHOD = 'random', iter_fit = 500)

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
    