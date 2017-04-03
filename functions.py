#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 16:31:16 2017

@author: vinnam
"""

import numpy as np

class sinc_simple:
    def __init__(self, FLOATING_TYPE = 'float32'):
        self.FLOATING_TYPE = FLOATING_TYPE
        self.D = 1
        self.scale = np.reshape(np.array([1.0], dtype = self.FLOATING_TYPE), [1, -1])
        self.bias = np.reshape(np.array([0.0], dtype = self.FLOATING_TYPE), [1, -1])
        
    def evaluate(self, x):
        eval_x = self.scale * (x + self.bias)
        return np.sinc(eval_x * np.pi)
        
    def dim(self):
        return self.D