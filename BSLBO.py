#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:56:00 2017

@author: vinnam
"""

from MHGP import MHGP
import numpy as np
import tensorflow as tf
from sklearn import preprocessing, decomposition
from scipy.optimize import minimize
import settings
import functions
import matplotlib.pyplot as plt
from util_func import X_to_Z, Z_to_Xhat
from sampling import find_enclosingbox, sample_enclosingbox

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

def x_to_dict(x, M, K, D):
    result = {}
    result['Z'] = x[:(M * K)].reshape([M, K])
    result['mu'] = x[(M * K) : (M * K + K * D)].reshape([K, D])
    result['log_Sigma'] = x[(M * K + K * D) : (M * K + 2 * (K * D))].reshape([K, D])
    result['log_sigma_f'] = x[-2]
    result['log_tau'] = x[-1]
    
    return result

class BSLBO:
    def __init__(self, fun, K, N, Kr, L, Max_M, ACQ_FUN):
#        N = 10
#        M = 10
#        K = 1
#        fun = functions.sinc_simple2()
#        ACQ_FUN = 'EI'
        self.FLOATING_TYPE = settings.dtype        

        D = fun.D
        data = {}
        
        data['X'], data['y'] = fun.evaluate(np.random.uniform(low = -1.0, high = 1.0, size = [N, D]).astype(settings.dtype))
        data['max_fun'] = np.max(data['y'])
        scaler = preprocessing.MinMaxScaler((-1,1))
                
        data['y_scaled'] = scaler.fit_transform(data['y'])
        data['scaled_max_fun'] = np.array(1.0, dtype = settings.dtype)
        data['beta'] = fun.beta(data)
                
#        types = ['X', 'y_scaled', 'scaled_max_fun']
        types = ['X', 'y', 'max_fun']
        
        M = min(Max_M, N)
        
        self.fitted_params = {'Z' : None, 'mu' : None, 'log_Sigma' : None, 'log_sigma_f' : None, 'log_tau' : None}
        self.N = N
        self.M = M
        self.K = K
        self.D = D
        self.fun = fun
        self.data = data
        self.types = types
        self.scaler = scaler
        self.ACQ_FUN = ACQ_FUN
        self.L = L
        self.Kr = Kr
        self.Max_M = Max_M
        
        gp = MHGP(K, D, ACQ_FUN = ACQ_FUN)
        
        self.gp = gp
        self.session = tf.Session(graph = self.gp.graph)        
        self.fitting()
        
    def train_obj(self, x):
        M = self.M
        D = self.D
        K = self.K
        param_dict = x_to_dict(x, M, K, D)
        
        feed_dict = {self.gp.inputs['X'] : self.data['X'], self.gp.inputs['y'] : self.data['y']}
        
        for param in param_dict.keys():
            feed_dict[self.gp.params[param]] = param_dict[param]
        
        f, g = self.session.run([self.gp.train_f, self.gp.train_g], feed_dict)
        
        return f, g
        
    def init_params(self, init_method = 'pca'):
        FLOATING_TYPE = self.FLOATING_TYPE
        
        ####### INIT VALUES OF PARAMETERS #######
        
        X = self.data[self.types[0]]
        y = self.data[self.types[1]]
        N = self.N
        M = self.M
        D = self.D
        K = self.K
        
        init_value = {}
        
        if init_method is 'pca':
            pca = decomposition.PCA(n_components = K)
            pca.fit(X)
            init_value['mu'] = np.array(pca.components_, dtype = FLOATING_TYPE)
        else:
            init_value['mu'] = np.array(np.random.normal(size = [K, D]), dtype = FLOATING_TYPE)
            
        Mu = np.matmul(X, np.transpose(init_value['mu']))
        
        inputScales = 10 / np.square(np.max(Mu, axis = 0) - np.min(Mu, axis = 0))
        
        init_value['mu'] = init_value['mu'] * np.reshape(inputScales, [-1, 1])
            
        init_value['log_Sigma'] = 0.5 * np.log(1 / np.array(D, dtype = FLOATING_TYPE) + (0.001 / np.array(D, dtype = FLOATING_TYPE)) * np.random.normal(size = [K, D]))
        
        init_value['Z'] = np.matmul(X, np.transpose(init_value['mu']))[np.random.permutation(N)[0:M], :]
        
#        init_value['Z'] = np.random.uniform(low = -np.sqrt(D), high = np.sqrt(D), size = [M, K])
        
        init_value['log_sigma_f'] = 0.5 * np.log(np.var(y) + 1e-6)
        
        init_value['log_tau'] = 0.5 * np.log((np.var(y) + 1e-6) / 100)
        
#        print init_value
        
        return np.concatenate([init_value[param].reshape(-1) for param in ['Z', 'mu', 'log_Sigma', 'log_sigma_f', 'log_tau']])
        
    def fitting(self, init_method = 'pca', method = 'CG', max_iter1 = 100, max_iter2 = 500):
        M = self.M
        D = self.D
        K = self.K
        
        train_step1 = ObjectiveWrapper(self.train_obj, 0)
        train_step2 = ObjectiveWrapper(self.train_obj, 1)
        
        x0 = self.init_params(init_method = init_method)
        
        result = minimize(fun = train_step1,
                          x0 = x0,
                          method = method,
                          jac = True,
                          tol = None,
                          callback = None,
                          options = {'maxiter' : max_iter1, 'gtol' : np.finfo('float64').min})
    
        x0 = result.x
        
        result = minimize(fun = train_step2,
                          x0 = x0,
                          method = method,
                          jac = True,
                          tol = None,
                          callback = None,
                          options = {'maxiter' : max_iter2, 'gtol' : np.finfo('float64').min})
        
        self.train_result = result
        
        self.fitted_params = x_to_dict(result.x, M, K, D)
        
        feed_dict = {self.gp.inputs['X'] : self.data['X'], self.gp.inputs['y'] : self.data['y']}
        
        for param in self.fitted_params.keys():
            feed_dict[self.gp.params[param]] = self.fitted_params[param]
            
        self.l_square = self.session.run(self.gp.l_square, feed_dict)
        
    def test(self, x_star):
        X = self.data[self.types[0]]
        y = self.data[self.types[1]]
        max_fun = data[self.types[2]]
        beta = data['beta']
        
        try:
            if x_star.shape[1] is not self.D:
                print 'Dimension error'
                return None
        except:
            print 'Dimension error'
        
        feed_dict = {self.gp.inputs['X'] : X, self.gp.inputs['y'] : y, self.gp.acq_inputs['x_star'] : x_star, self.gp.acq_inputs['max_fun'] : max_fun, self.gp.acq_inputs['beta'] : beta}
        
        for param in self.fitted_params.keys():
            feed_dict[self.gp.params[param]] = self.fitted_params[param]
            
        mu, var, F_acq = self.session.run([self.gp.mu_star, self.gp.var_star, self.gp.acq_f], feed_dict)
        
        return [mu, var, F_acq]
    
    def finding_next(self, data, types, A, b, num_sample):
        X = self.data[self.types[0]]
        y = self.data[self.types[1]]
        max_fun = self.data[self.types[2]]
        beta = self.data['beta']
        
        max_obj = np.finfo(self.FLOATING_TYPE).min
        x_next = sample_enclosingbox(A, b, 1)
        
        for n in xrange(10):
            try:
                z_star = sample_enclosingbox(A, b, num_sample / 10)
                
                x_star = np.matmul(z_star, A.transpose())
                
                feed_dict = {self.gp.inputs['X'] : X,
                             self.gp.inputs['y'] : y,
                             self.gp.acq_inputs['x_star'] : x_star,
                             self.gp.acq_inputs['max_fun'] : max_fun,
                             self.gp.acq_inputs['beta'] : beta}
                
                for param in self.fitted_params.keys():
                    feed_dict[self.gp.params[param]] = self.fitted_params[param]
                    
                obj = self.session.run(self.gp.acq_f, feed_dict)
                
                temp = np.max(obj)
                
                if temp > max_obj:
                    x_next = x_star[np.argmax(obj)]
                    max_obj = temp
            except:
                print 'LLT error'
                continue
            
        return np.reshape(x_next, [1, -1]), obj
    
    def iterate(self, num_sample):
        M = self.M
        D = self.D
        L = self.L
        Kr = self.Kr
        
        data = self.data
        fun = self.fun
        types = self.types
        scaler = self.scaler
        
        Mu = self.fitted_params['mu']
        cond = np.sqrt(self.l_square) > L
        
        if np.any(cond):
            W = Mu[cond, :].reshape([-1, D])
        else:
            W = Mu[np.argsort(self.l_square)[0:Kr], :]
        
#        print np.sqrt(gp.l_square)
#        print W.shape
        
        W = W.transpose()
        WT = np.transpose(W)
        WTW = np.matmul(WT, W)
        A = np.transpose(np.linalg.solve(WTW, WT)) # D x Ke
        Ke = A.shape[1]
        
        b = np.sqrt(Ke) * np.ones([D, 1])
        
        next_x_uncliped, next_x_obj = self.finding_next(data, types, A, b, num_sample)
        next_x, next_y = fun.evaluate(next_x_uncliped)
        
        data['X'] = np.append(data['X'], next_x, axis = 0)
        data['y'] = np.append(data['y'], next_y, axis = 0)
         
        data['y_scaled'] = scaler.fit_transform(data['y'])
        
        data['max_fun'] = np.max(data['y'])
        data['beta'] = fun.beta(data)
        
        M = min(len(data['y']), self.Max_M)
        N = len(data['y'])
        
        self.data = data
        self.M = M
        self.N = N
        
        self.fitting()
        
        return next_x

fun = functions.sinc_simple2()

R = BSLBO(fun, 1, 10, 1, 10, 100, 'UCB')
#print R.fitted_params
#R.fitting(method = 'CG', max_iter1 = 100, max_iter2 = 5000)
data = R.data
gp = R.gp

#    W = gp.fitted_params['mu'].transpose()
W = R.fitted_params['mu']
W = W[np.argmax(R.l_square), :].reshape([-1, R.D])
W = W.transpose()

#    W = fun.W
WT = np.transpose(W)
WTW = np.matmul(WT, W)
B = np.transpose(np.linalg.solve(WTW, WT)) # D x K
D = fun.D

fx_min, fx_max = find_enclosingbox(B, np.sqrt(1) * np.ones([D, 1]))
fx = np.linspace(fx_min, fx_max, num = 100).reshape([100, 1])
fy = fun.evaluate(Z_to_Xhat(fx, W))[1]

mu, var, EI = R.test(Z_to_Xhat(fx, W))
next_x = R.iterate(10000)

EI_scaled = preprocessing.MinMaxScaler((np.min(fy),np.max(fy))).fit_transform(EI.reshape([-1, 1]))
plt.figure()
plt.plot(fx, fy)
plt.scatter(X_to_Z(data['X'], W), data['y'])
plt.plot(fx, mu, 'k')
plt.plot(fx, mu + np.sqrt(var), 'k:')
plt.plot(fx, mu - np.sqrt(var), 'k:')
plt.plot(fx, EI_scaled, '-.')
#print R.train_result
plt.scatter(X_to_Z(data['X'][-1], W), np.min(data['y']), marker = 'x', color = 'g')
plt.title('N is ' + str(len(data['y'])))
plt.show()
        
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
##        N = 10
#        M = 10
#        K = 1
#        fun = functions.sinc_simple2()
#        ACQ_FUN = 'EI'
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
