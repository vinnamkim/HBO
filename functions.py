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
        return self.f(eval_x)
    
    def update(self, next_x, data):
        next_y = self.evaluate(next_x)
        data['X'] = np.append(data['X'], next_x, axis = 0)
        data['y'] = np.append(data['y'], next_y, axis = 0)
        data['max_fun'] = np.max(data['y'])
        return
        
    def gen_data(self, N):
        D = self.D
        FLOATING_TYPE = self.FLOATING_TYPE
        
        init_X = np.array(np.random.uniform(low = -1.0, high = 1.0, size = [N, D]), dtype = FLOATING_TYPE)
        init_y = np.reshape(np.array([self.evaluate(x) for x in init_X], dtype = FLOATING_TYPE), [-1, 1])
        
        return {'X' : init_X, 'y' : init_y, 'D' : D, 'max_fun' : np.max(init_y)}
        
class sinc_simple(function):
    def __init__(self, FLOATING_TYPE = settings.dtype):
        self.FLOATING_TYPE = FLOATING_TYPE
        self.D = 1
        self.x_scale = np.reshape(np.array([1.0], dtype = self.FLOATING_TYPE), [1, -1])
        self.x_bias = np.reshape(np.array([0.0], dtype = self.FLOATING_TYPE), [1, -1])
    
    def f(self, x):
        return np.sinc(np.pi * x)

class sinc_simple2(function):
    def __init__(self, FLOATING_TYPE = settings.dtype):
        self.FLOATING_TYPE = FLOATING_TYPE
        self.D = 2
        self.x_scale = np.reshape(np.array([1.0], dtype = self.FLOATING_TYPE), [1, -1])
        self.x_bias = np.reshape(np.array([0.0], dtype = self.FLOATING_TYPE), [1, -1])
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