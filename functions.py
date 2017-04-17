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
        return eval_x, self.f(eval_x)
    
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

class sinc_simple10(function):
    def __init__(self, FLOATING_TYPE = settings.dtype):
        self.FLOATING_TYPE = FLOATING_TYPE
        self.D = 10
        self.x_scale = np.reshape(np.array([1.0], dtype = self.FLOATING_TYPE), [1, -1])
        self.x_bias = np.reshape(np.array([0.0], dtype = self.FLOATING_TYPE), [1, -1])
        self.x_max = np.ones([self.D], dtype = self.FLOATING_TYPE)
        self.x_min = -np.ones([self.D], dtype = self.FLOATING_TYPE)
        self.W = np.random.normal(size = [self.D, 1])
        
    def f(self, x):
        return np.sinc(np.pi * np.matmul(x, self.W))
 
class brainin(function):
    def __init__(self, D, FLOATING_TYPE = settings.dtype):
        self.FLOATING_TYPE = FLOATING_TYPE
        self.D = D
        self.x_max = np.array([10., 15.], dtype = self.FLOATING_TYPE)
        self.x_min = np.array([-5., 0.], dtype = self.FLOATING_TYPE)
        
        self.x_scale = 0.5 * (self.x_max - self.x_min)
        self.x_bias = (self.x_max + self.x_min) / (self.x_max - self.x_min)
        
        self.eff_indices = np.random.permutation(10)[0:2]
#        self.eff_vec = np.zeros([D, 1], dtype = self.FLOATING_TYPE)
#        self.eff_vec[self.eff_indices] = 1.
    
    def f(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        return np.square(x2 - 5.1 / np.square(2 * np.pi) * np.square(x1) + 5 * x1 / np.pi - 6) + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    
    def evaluate(self, x):
        eval_x = np.clip(x, -1., 1.)
        eff_x = eval_x[:, self.eff_indices]
        return eval_x, -self.f(self.x_scale * (eff_x + self.x_bias)).reshape([-1, 1])
        
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