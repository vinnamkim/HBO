#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 23:01:03 2017

@author: vinnam
"""
import settings
import numpy as np
from scipy.optimize import linprog

def sample_enclosingbox(A, b, num_sample):
    x_min, x_max = find_enclosingbox(A, b)
    D = A.shape[1]
    
    sample = np.zeros([1, D], dtype = settings.dtype)
    
    while(len(sample) < num_sample):
        x = np.random.uniform(low = x_min, high = x_max, size = [num_sample, D]).astype(settings.dtype)
        Ax = np.matmul(A, x.transpose())
        cond = np.prod(b - Ax >= 0, axis = 0) * np.prod(b + Ax >= 0, axis = 0)
        
        sample = np.append(sample, x[cond.astype(bool)], axis = 0)
        
    return sample[1:num_sample + 1]
        
def find_enclosingbox(A, b):
    n = A.shape[1]
    
    c = np.zeros([n, n], dtype = settings.dtype)
    
    x_min = np.zeros(n, dtype = settings.dtype)
    x_max = np.zeros(n, dtype = settings.dtype)
    
    AA = np.concatenate((A, -A), axis = 0)
    bb = np.concatenate((b, b), axis = 0)
    
    for i in xrange(n):
        c[i][i] = 1.
        
        sol = linprog(c[i], A_ub = AA, b_ub = bb, bounds = (None, None))
               
        x_min[i] = sol.fun
        x_max[i] = -sol.fun
             
    return x_min, x_max


def hit_and_run(x_star, A, b, Burnin = 100, Iter = 1000):
    D = np.shape(b)[0]
    
    for i in xrange(Burnin):
        Ax = np.matmul(A, np.transpose(x_star))
        LEFT = -b - Ax
        RIGHT = b - Ax
        
        alpha = np.random.normal(size = [D, 1])
        A_alpha = np.matmul(A, alpha)
        l = LEFT / A_alpha 
        r = RIGHT / A_alpha
        
        pos = A_alpha > 0
        neg = A_alpha < 0
        
        theta_min = np.max(np.concatenate((l[pos], r[neg])))
        theta_max = np.min(np.concatenate((r[pos], l[neg])))
        
        theta = np.random.uniform(low = theta_min, high = theta_max)
        
        x_star = x_star + theta * np.transpose(alpha)
        
    X = np.zeros(shape = [Iter, D], dtype = x_star.dtype)
    
    for i in xrange(Iter):
        Ax = np.matmul(A, np.transpose(x_star))
        LEFT = -b - Ax
        RIGHT = b - Ax
        
        alpha = np.random.normal(size = [D, 1])
        A_alpha = np.matmul(A, alpha)
        l = LEFT / A_alpha 
        r = RIGHT / A_alpha
        
        pos = A_alpha > 0
        neg = A_alpha < 0
        
        theta_min = np.max(np.concatenate((l[pos], r[neg])))
        theta_max = np.min(np.concatenate((r[pos], l[neg])))
        
        theta = np.random.uniform(low = theta_min, high = theta_max)
        
        x_star = x_star + theta * np.transpose(alpha)
        
        X[i, :] = x_star
        
    return X

def test():
    import matplotlib.pyplot as plt
    
    A = np.array([[0.5, 0.75], [0, 1]])
    A = np.random.normal(size = [2, 2])
    b = np.ones([2, 1])
    
    x1 = np.linspace(-4., 4.)
    
    x2 = (b - A[:, 0].reshape([2, 1]) * x1) / A[:, 1].reshape([2, 1])
    
    x3 = (-b - A[:, 0].reshape([2, 1]) * x1) / A[:, 1].reshape([2, 1])
    
    test = hit_and_run(np.zeros([1,2]), A, b, 10000, 100)
    plt.plot(x1, x2[0, :])
    plt.plot(x1, x2[1, :])
    plt.plot(x1, x3[0, :])
    plt.plot(x1, x3[1, :])
    plt.scatter(test[:, 0], test[:, 1])
    
    test = sample_enclosingbox(A, b, 100)
    plt.plot(x1, x2[0, :])
    plt.plot(x1, x2[1, :])
    plt.plot(x1, x3[0, :])
    plt.plot(x1, x3[1, :])
    plt.scatter(test[:, 0], test[:, 1])
    
    plt.plot(x1, x2[0, :])
    plt.plot(x1, x2[1, :])
    plt.plot(x1, x3[0, :])
    plt.plot(x1, x3[1, :])
    x_min, x_max = find_enclosingbox(A, b)
    plt.scatter(x_min[0], x_min[1])
    plt.scatter(x_max[0], x_max[1])

