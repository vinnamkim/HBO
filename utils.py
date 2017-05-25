#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 09:44:59 2017

@author: vinnam
"""
import numpy as np

def X_to_Z(X, W):
    return np.matmul(X, W)  # X : N x D,  W : D x K
                        
def Z_to_Xhat(Z, W):
    WT = np.transpose(W)
    WTW = np.matmul(WT, W)
    B = np.transpose(np.linalg.solve(WTW, WT)) # D x K
    
    return np.matmul(Z, B.transpose()) # Z : N x K,  B : D x K

class ObjectiveWrapper(object):
    """
    A simple class to wrap the objective function in order to make it more
    robust.

    The previously seen state is cached so that we can easily access it if the
    model crashes.
    """

    def __init__(self, objective, step):
        self._objective = objective
        self._step = step
        self._previous_x = None

    def __call__(self, x):
        f, g = self._objective(x)
        g_is_fin = np.isfinite(g)
        if self._step == 0:
            g[-1] = 0.
            g[-2] = 0.
        if np.all(g_is_fin):
            self._previous_x = x  # store the last known good value
            return f, g
        else:
            print("Warning: inf or nan in gradient: replacing with zeros")
            return f, np.where(g_is_fin, g, 0.)


class ObjectiveWrapper1(object):
    def __init__(self, objective, A):
        self._objective = objective
        self._A = A
        self._previous_x = None

    def __call__(self, x):
        f, g = self._objective(x, self._A)
        g_is_fin = np.isfinite(g)

        if np.all(g_is_fin):
            self._previous_x = x  # store the last known good value
            return f, g
        else:
            print("Warning: inf or nan in gradient: replacing with zeros")
            return f, np.where(g_is_fin, g, 0.)
