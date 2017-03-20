#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 20:31:28 2017

@author: vinnam
"""

import numpy as np
import matplotlib.pyplot as plt

class sinc_2d:
    def __init__(self, N, D, d, tau, K = 2, FLOATING_TYPE = np.float32):
        self.X = np.asarray(np.random.uniform(size = [N, D]), dtype = FLOATING_TYPE)
        self.W = np.asarray(np.concatenate((np.random.normal(size = [K, d]), np.zeros(shape = [K, D - d])), axis = 1), dtype = FLOATING_TYPE)
        
        self.X_p = np.matmul(self.X, np.transpose(self.W))
        
        self.sinc = np.sinc(self.X_p[:, 0])
        
        self.x2 = -np.square(2 * (self.X_p[:, 1] - 0.5 * (max(self.X_p[:, 1]) + min(self.X_p[:, 1]))) / (max(self.X_p[:, 1]) - min(self.X_p[:, 1])))
        
        self.Y = np.reshape(self.sinc + self.x2 + tau * np.random.normal(size = N), [-1, 1])
        
    def data(self):
        return {'X' : self.X,
                'W' : self.W,
                'X_p' : self.X_p,
                'sinc' : self.sinc,
                'x2' : self.x2,
                'Y' : self.Y}
        
    def show_Y(self):
        plt.figure()
        line1 = plt.scatter(self.X_p[:, 0], self.Y, label = 'x1')
        line2 = plt.scatter(self.X_p[:, 1], self.Y, label = 'x2')
        plt.legend(handles=[line1, line2])
        plt.show()
        return    
    def show_fun(self):
        plt.figure()
        line1 = plt.scatter(self.X_p[:, 0], self.sinc, label = 'sinc')
        line2 = plt.scatter(self.X_p[:, 1], self.x2, label = 'square')
        plt.legend(handles=[line1, line2])
        plt.show()
        return
    