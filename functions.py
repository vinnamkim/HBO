#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 16:31:16 2017

@author: vinnam
"""

import numpy as np
import settings

class function:
    def evaluate(self, x):
        eval_x = self.x_scale * (x + self.x_bias)
        eval_x = np.clip(eval_x, self.x_min, self.x_max)
        return self.f(eval_x)
    
    def update(self, next_x, data):
        next_y = self.evaluate(next_x)
        try:
            data['X'] = np.append(data['X'], next_x, axis = 0)
        except:
            data['X'] = next_x
        try:
            data['y'] = np.append(data['y'], next_y, axis = 0)
        except:
            data['y'] = next_y
            
        data['max_fun'] = np.max(data['y'])
        
        return
    
    def beta(self, data):
        N = len(data['y'])
        D = self.D
        t = N + 1
        a = settings.UCB_a
        b = settings.UCB_b
        r = settings.UCB_r
        delta = settings.UCB_delta
        
        beta = 2 * (np.log(2) + 2 * np.log(t) + 2 * np.log(np.pi) - np.log(3) - np.log(delta)) + \
        2 * D * (2 * np.log(t) + np.log(D) + np.log(b) + np.log(r) + 0.5 * np.log(np.log(4) + np.log(D) + np.log(a) - np.log(delta)))
        
        return beta
        
    def gen_data(self, N):
        D = self.D
        FLOATING_TYPE = self.FLOATING_TYPE
        
        init_X = np.array(np.random.uniform(low = -1.0, high = 1.0, size = [N, D]), dtype = FLOATING_TYPE)
        init_y = np.reshape(np.array([self.evaluate(x) for x in init_X], dtype = FLOATING_TYPE), [-1, 1])
        
        max_fun = np.max(init_y)
        
        t = N + 1
        a = settings.UCB_a
        b = settings.UCB_b
        r = settings.UCB_r
        delta = settings.UCB_delta
        
        beta = 2 * (np.log(2) + 2 * np.log(t) + 2 * np.log(np.pi) - np.log(3) - np.log(delta)) + \
        2 * D * (2 * np.log(t) + np.log(D) + np.log(b) + np.log(r) + 0.5 * np.log(np.log(4) + np.log(D) + np.log(a) - np.log(delta)))
        
        return {'X' : init_X, 'y' : init_y, 'D' : D, 'max_fun' : max_fun, 'beta' : beta}
        
class sinc_simple(function):
    def __init__(self, FLOATING_TYPE = settings.dtype):
        self.FLOATING_TYPE = FLOATING_TYPE
        self.D = 1
        self.x_scale = np.reshape(np.array([1.0], dtype = self.FLOATING_TYPE), [1, -1])
        self.x_bias = np.reshape(np.array([0.0], dtype = self.FLOATING_TYPE), [1, -1])
        self.x_max = np.ones([self.D], dtype = self.FLOATING_TYPE)
        self.x_min = -np.ones([self.D], dtype = self.FLOATING_TYPE)
        
    def f(self, x):
        return np.sinc(np.pi * x)

class sinc_simple2(function):
    def __init__(self, FLOATING_TYPE = settings.dtype):
        self.FLOATING_TYPE = FLOATING_TYPE
        self.D = 2
        self.x_scale = np.reshape(np.array([1.0], dtype = self.FLOATING_TYPE), [1, -1])
        self.x_bias = np.reshape(np.array([0.0], dtype = self.FLOATING_TYPE), [1, -1])
        self.x_max = np.ones([self.D], dtype = self.FLOATING_TYPE)
        self.x_min = -np.ones([self.D], dtype = self.FLOATING_TYPE)
        self.W = np.random.normal(size = [self.D, 1])
        
    def f(self, x):
        return np.sinc(np.pi * np.matmul(x, self.W))

#
#class sinc_simple():
#    def __init__(self, FLOATING_TYPE = settings.dtype):
#        self.FLOATING_TYPE = FLOATING_TYPE
#        self.D = 1
#        self.x_scale = np.reshape(np.array([1.0], dtype = self.FLOATING_TYPE), [1, -1])
#        self.x_bias = np.reshape(np.array([0.0], dtype = self.FLOATING_TYPE), [1, -1])
#        
#    def evaluate(self, x):
#        eval_x = self.x_scale * (x + self.x_bias)
#        return np.sinc(eval_x * np.pi)
#        