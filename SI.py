#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 11:19:34 2017

@author: vinnam
"""

from cvxpy import *
import numpy as np

def Danzig_Selecter(Phi, y, lam, D, m_X, m_Phi):
#    D = 2
#    m_X = 10
#    m_Phi = 3
#    Phi = [np.random.uniform(size = [D, m_X]) for i in xrange(m_Phi)]
#    Phi = [1 / np.sqrt(m_Phi) * (2 * np.round(P) - 1) for P in Phi]
#    y = np.random.normal(size = [m_Phi, 1])
#    lam = 0.2
#    
    X = Variable(D, m_X) # D x m_X
    vec_X = vec(X)
    
    vec_Phi_T = vstack([vec(P).T for P in Phi])
    
    Phi_X = vec_Phi_T * vec_X
    
    residual = y - Phi_X
    
    const = norm(sum([e * P for e, P in zip(residual, Phi)]))
    
    constraints = [const <= lam]
    
    # Form objective.
    obj = Minimize(norm(X, 'nuc'))
    
    # Form and solve problem.
    prob = Problem(obj, constraints)
    
    prob.solve() # Returns the optimal value.
    
    return X.value
    
def SI(Phi, y, epsilon, K, C):
    D = Phi[0].shape[0]
    m_X = Phi[0].shape[1]
    m_Phi = y.shape[0]
    lam = 1.2 * C * epsilon * D * m_X * np.square(K) / (2 * np.sqrt(m_Phi))
    X_hat = Danzig_Selecter(Phi, y, lam, D, m_X, m_Phi)
    U, D, V = np.linalg.svd(X_hat)
    
    return np.array(U[:, 0:K])
    