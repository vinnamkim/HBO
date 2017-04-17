#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:56:00 2017

@author: vinnam
"""

from MHGP import MHGP
import numpy as np
from sklearn import preprocessing
import settings
import functions
from util_func import X_to_Z, Z_to_Xhat
from sampling import find_enclosingbox

class BSLBO:
    def __init__(self, fun, K, N, M, Kr, L, Max_M, ACQ_FUN):
#        N = 10
#        M = 10
#        K = 1
#        fun = functions.sinc_simple2()
#        ACQ_FUN = 'EI'
        
        D = fun.D
        data = {}
        
        data['X'], data['y'] = fun.evaluate(np.random.uniform(low = -1.0, high = 1.0, size = [N, D]).astype(settings.dtype))
        data['max_fun'] = np.max(data['y'])
        scaler = preprocessing.MinMaxScaler((-1,1))
                
        data['y_scaled'] = scaler.fit_transform(data['y'])
        data['scaled_max_fun'] = np.array(1.0, dtype = settings.dtype)
        data['beta'] = fun.beta(data)
                
        types = ['X', 'y_scaled', 'scaled_max_fun']
#        types = ['X', 'y', 'max_fun']
        
        M = min(Max_M, M)
        
        gp = MHGP(M, K, D, ACQ_FUN = ACQ_FUN)
        
        gp.fitting(data, types)
        
        self.K = K
        self.M = M
        self.D = D
        self.fun = fun
        self.data = data
        self.types = types
        self.gp = gp
        self.scaler = scaler
        self.ACQ_FUN = ACQ_FUN
        self.L = L
        self.Kr = Kr
        self.Max_M = Max_M
        
    def iterate(self, num_sample):
        M = self.M
        D = self.D
        K = self.K
        L = self.L
        Kr = self.Kr
        
        data = self.data
        fun = self.fun
        types = self.types
        scaler = self.scaler
        gp = self.gp
        ACQ_FUN = self.ACQ_FUN
        
        Mu = gp.fitted_params['mu']
        cond = np.sqrt(gp.l_square) > L
        
        if np.any(cond):
            W = Mu[cond, :].reshape([-1, D])
        else:
            W = Mu[np.argsort(gp.l_square)[0:Kr], :]
        
#        print np.sqrt(gp.l_square)
#        print W.shape
        
        W = W.transpose()
        WT = np.transpose(W)
        WTW = np.matmul(WT, W)
        A = np.transpose(np.linalg.solve(WTW, WT)) # D x Ke
        Ke = A.shape[1]
        
        b = np.sqrt(Ke) * np.ones([D, 1])
        
        next_x_uncliped, next_x_obj = gp.finding_next(data, types, A, b, num_sample)
        next_x, next_y = fun.evaluate(next_x_uncliped)
        
        data['X'] = np.append(data['X'], next_x, axis = 0)
        data['y'] = np.append(data['y'], next_y, axis = 0)
         
        data['y_scaled'] = scaler.fit_transform(data['y'])
        
        data['max_fun'] = np.max(data['y'])
        data['beta'] = fun.beta(data)
        
        M = min(len(data['y']), self.Max_M)
        
        if len(data['y']) == M:
            self.gp = MHGP(M, K, D, ACQ_FUN = ACQ_FUN)
        
        self.gp.fitting(data, types)
        
        self.data = data
        self.M = M
        
        return next_x

def test():
    import matplotlib.pyplot as plt
    
    fun = functions.brainin(10)
    #fun = functions.sinc_simple2()
    #fun = functions.sinc_simple10()
    #fun = functions.sinc_simple()
    R = BSLBO(fun, 5, 500, 100, 2, 0.5, 100, ACQ_FUN = 'UCB')
    
    for i in xrange(10):
        data = R.data
        gp = R.gp
        
    #    W = gp.fitted_params['mu'].transpose()
        W = gp.fitted_params['mu']
        W = W[np.argmax(gp.l_square), :].reshape([-1, R.D])
        W = W.transpose()
        
    #    W = fun.W
        WT = np.transpose(W)
        WTW = np.matmul(WT, W)
        B = np.transpose(np.linalg.solve(WTW, WT)) # D x K
        D = fun.D
        
        fx_min, fx_max = find_enclosingbox(B, np.sqrt(D) * np.ones([D, 1]))
        fx = np.linspace(fx_min, fx_max, num = 100).reshape([100, 1])
        fy = fun.evaluate(Z_to_Xhat(fx, W))[1]
        
        mu, var, EI = gp.test(data, R.types, Z_to_Xhat(fx, W))
        
        next_x = R.iterate(1000)
        EI_scaled = preprocessing.MinMaxScaler((np.min(fy),np.max(fy))).fit_transform(EI.reshape([-1, 1]))
        
        plt.figure()
        plt.plot(fx, fy)
        plt.scatter(X_to_Z(data['X'], W), data['y'])
        plt.plot(fx, mu, 'k')
        plt.plot(fx, mu + np.sqrt(var), 'k:')
        plt.plot(fx, mu - np.sqrt(var), 'k:')
        plt.plot(fx, EI_scaled, '-.')
        plt.scatter(X_to_Z(data['X'][-1], W), np.min(data['y']), marker = 'x', color = 'g')
        plt.title('N is ' + str(len(data['y'])))
        plt.show()
    
    
    
#    Psi2_star = np.sum(gp.debug(data, R.types, data['X'], 'Psi2_star').transpose().reshape([-1, 20, 20]).transpose([0, 2, 1]), axis = 0)
#    Psi2 = gp.debug(data, R.types, data['X'], 'Psi2')
    
    
#    debugs = gp.debug(data, R.types, np.matmul(fx, W.transpose()))
#    fx1 = np.random.uniform(-1., 1., size = [1, 2])
#    mu, var, EI = gp.test(data, R.types, fx1)    
#    a = gp.debug(data, R.types, fx1)
#    
#    np.trace(np.matmul(np.matmul(a[1], a[1].transpose()), a[2]).reshape([50, 50]))
#    
#    np.exp(2 * gp.fitted_params['log_sigma_f']) * np.matmul(a[2], a[1])
#    
#    from scipy.io import savemat
#    savemat('matdat', {'Lm' : Psi2_star, 'K_uu_Inv' : K_uu_Inv, 'Psi2_star' : Psi2_star})
##    [self.debugs['var_star1'], self.debugs['var_star2'], self.debugs['var_star3'], self.debugs['K_uu_Inv'], self.debugs['A_Inv'], self.debugs['Psi2_star'], self.debugs['Lstar']]
#    Lstar = np.squeeze(a[6])
#    Lm = a[7]
#    La = a[8]
#    Psi2_star = np.squeeze(a[5])
#    K_uu_Inv = a[3]
#    np.trace(np.linalg.solve(Lm, np.transpose(np.linalg.solve(Lm, Psi2_star))))
#    np.trace(solve_triangular(Lm, np.transpose(solve_triangular(Lm, Psi2_star))))
#    np.trace(np.matmul(K_uu_Inv, Psi2_star))
#    np.trace(np.matmul(K_uu_Inv, np.matmul(Lstar, Lstar.transpose())))
#    np.matmul(Lstar, Lstar.transpose())
#    
#    np.trace(np.matmul(a[3], np.squeeze(a[5])))
#    np.exp(2 * gp.fitted_params['log_tau']) * np.trace(np.matmul(a[4], np.squeeze(a[5])))
#    np.exp(2 * gp.fitted_params['log_tau']) * a[0]
#    
#    np.exp(2 * gp.fitted_params['log_sigma_f']) * (np.exp(2 * gp.fitted_params['log_tau']) * a[0] - a[1] + np.exp(2 * gp.fitted_params['log_sigma_f']) * a[2] + 1)
#    
#    np.exp(2 * gp.fitted_params['log_tau']) * a[0] - a[1] + 1
#                   
#
#Alpha = gp.debug(data, R.types, data['X'], 'Alpha')
#La = gp.debug(data, R.types, data['X'], 'La')
#A = gp.debug(data, R.types, data['X'], 'A')
#A_Inv = gp.debug(data, R.types, data['X'], 'A_Inv')
#LaInvLmInv = gp.debug(data, R.types, data['X'], 'LaInvLmInv')
#Psi1_star = gp.debug(data, R.types, data['X'], 'Psi1_star')
#YPsi1InvLmInvLa = gp.debug(data, R.types, data['X'], 'YPsi1InvLmInvLa')
#
#gp.debug(data, R.types, data['X'], 'test_BB') - YPsi1InvLmInvLa.transpose()
#
#np.linalg.solve(np.matmul(Lm, La), np.matmul(Psi1_star, data['y']))
#np.linalg.solve(La, np.linalg.solve(Lm, np.matmul(Psi1_star, data['y'])))
#
#Lm = gp.debug(data, R.types, data['X'], 'Lm')
#La = gp.debug(data, R.types, data['X'], 'La')
#
#K_uu_Inv = gp.debug(data, R.types, data['X'], 'K_uu_Inv')
#K_uu = gp.debug(data, R.types, data['X'], 'K_uu')
#mu_star = gp.debug(data, R.types, data['X'], 'mu_star')
#
#var_star1 = gp.debug(data, R.types, data['X'], 'var_star1')
#var_star11 = gp.debug(data, R.types, data['X'], 'var_star11')
#
#var_star2 = gp.debug(data, R.types, data['X'], 'var_star2')
#var_star22 = gp.debug(data, R.types, data['X'], 'var_star22')
#
#print var_star2
#print var_star22
#
#var_star3 = gp.debug(data, R.types, data['X'], 'var_star3')
#var_star33 = gp.debug(data, R.types, data['X'], 'var_star33')
#
#print var_star3
#print var_star33
#
#var_star1 = gp.debug(data, R.types, data['X'][1].reshape([-1, 2]), 'var_star1')
#var_star11 = gp.debug(data, R.types, data['X'][1].reshape([-1, 2]), 'var_star11')
#
#Psi2_star1 = gp.debug(data, R.types, data['X'][0].reshape([-1, 2]), 'Psi2_star')
#var_star1 = gp.debug(data, R.types, data['X'][0].reshape([-1, 2]), 'var_star1')
#np.matmul(Psi1_star, Alpha)
#
#
#import tensorflow as tf
#sess = tf.InteractiveSession()
#test = tf.reshape(tf.range(18.), [2, 3, 3])
#test = test
#Lm = tf.matrix_band_part(tf.reshape(tf.range(1., 10.), [3, 3]), -1, 0)
#La = tf.matrix_band_part(tf.reshape(tf.range(3., 12.), [3, 3]), -1, 0)
#
#print sess.run(tf.matrix_triangular_solve(La, tf.matrix_triangular_solve(Lm, \
#tf.transpose(tf.matrix_triangular_solve(La, tf.matrix_triangular_solve(Lm, test[0]))))))
#
#print sess.run(tf.matrix_triangular_solve(tf.matmul(Lm, La), \
#tf.transpose(tf.matrix_triangular_solve(tf.matmul(Lm, La), test[0]))))
#
#print sess.run(tf.trace(tf.matrix_triangular_solve(tf.matmul(Lm, La), \
#tf.transpose(tf.matrix_triangular_solve(tf.matmul(Lm, La), test[0])))))
#
#print sess.run(tf.trace(tf.matrix_solve(tf.matmul(tf.matmul(Lm, La), tf.transpose(tf.matmul(Lm, La))), test[0])))
#
#
#M = 3
#
#test1 = tf.transpose(tf.reshape(tf.transpose(test, [0, 2, 1]), [-1, M]))
#
#print sess.run(test)
#print sess.run(test1)
#
#test2 = tf.matrix_triangular_solve(La, tf.matrix_triangular_solve(Lm, test1))
#
#print sess.run(test2[:, :3])
#print sess.run(tf.matrix_triangular_solve(La, tf.matrix_triangular_solve(Lm, test[0])))
#
#test3 = tf.transpose(tf.reshape(tf.transpose(test2), [-1, M, M]), [0, 2, 1])
#
#print sess.run(test3)
#
#test4 = tf.transpose(tf.reshape(test3, [-1, M]))
#
#print sess.run(test4[:, :3])
#print sess.run(tf.transpose(tf.matrix_triangular_solve(tf.matmul(Lm, La), test[0])))
#
#test5 = tf.matrix_triangular_solve(La, tf.matrix_triangular_solve(Lm, test4))
#
#test6 = tf.transpose(tf.reshape(tf.transpose(test5), [-1, M, M]), [0, 2, 1])
#
#print sess.run(test6[0])
#
#print sess.run(tf.trace(test6))
#
#
