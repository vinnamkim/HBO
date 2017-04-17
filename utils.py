#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 09:44:59 2017

@author: vinnam
"""

def X_to_Z(X, W):
    return np.matmul(X, W)  # X : N x D,  W : D x K
                        
def Z_to_Xhat(Z, W):
    WT = np.transpose(W)
    WTW = np.matmul(WT, W)
    B = np.transpose(np.linalg.solve(WTW, WT)) # D x K
    
    return np.matmul(Z, B.transpose()) # Z : N x K,  B : D x K
                    
