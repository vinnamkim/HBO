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
                
        types = ['X', 'y_scaled', 'scaled_max_fun']
#        types = ['X', 'y', 'max_fun']
        
#        data['X'] = np.append(data['X'], data['X'][-1].reshape([1, -1]), axis = 0)
#        data['y'] = np.append(data['y'], data['y'][-1].reshape([1, -1]), axis = 0)
        
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
        self.initiated = True
        
        gp = MHGP(K, D, ACQ_FUN = ACQ_FUN)
        
        self.gp = gp
        self.session = tf.Session(graph = self.gp.graph)        
        self.fitting()
        
    def train_obj(self, x):
        M = self.M
        D = self.D
        K = self.K
        param_dict = x_to_dict(x, M, K, D)
        
        feed_dict = {self.gp.inputs['X'] : self.data[self.types[0]], self.gp.inputs['y'] : self.data[self.types[1]]}
        
        for param in param_dict.keys():
            feed_dict[self.gp.params[param]] = param_dict[param]
        
        f, g = self.session.run([self.gp.train_f, self.gp.train_g], feed_dict)
        
        return f, g
        
    def acq_obj(self, z_star, A):
        feed_dict = {self.gp.inputs['X'] : self.data[self.types[0]],
                     self.gp.inputs['y'] : self.data[self.types[1]],
                     self.gp.acq_inputs['z_star'] : np.reshape(z_star, [1, -1]),
                     self.gp.acq_inputs['A'] : A,
                     self.gp.acq_inputs['max_fun'] : self.data[self.types[2]],
                     self.gp.acq_inputs['beta'] : self.data['beta']}
        
        for param in self.fitted_params.keys():
            feed_dict[self.gp.params[param]] = self.fitted_params[param]
            
        f, g = self.session.run([self.gp.acq_f, self.gp.acq_g], feed_dict)
        
        return np.negative(f).squeeze(), np.negative(g).squeeze()
    
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
        
    def fitting(self, init_method = 'random', method = 'L-BFGS-B', max_iter1 = 100, max_iter2 = 100, fit_iter = 10):
        M = self.M
        D = self.D
        K = self.K
        
        train_step1 = ObjectiveWrapper(self.train_obj, 0)
        train_step2 = ObjectiveWrapper(self.train_obj, 1)
            
        if self.initiated is True:
            prev_obj = np.finfo('float64').max
            cond = True
            
            while(cond):
                x0 = self.init_params(init_method = init_method)
                
                result = minimize(fun = train_step1,
                                  x0 = x0,
                                  method = method,
                                  jac = True,
                                  tol = None,
                                  callback = None,
                                  options = {'maxiter' : max_iter1})
#                                  options = {'maxiter' : max_iter1, 'gtol' : np.finfo('float64').min})
            
                x0 = result.x
        #        print x0[-2:-1]
                
                result = minimize(fun = train_step2,
                                  x0 = x0,
                                  method = method,
                                  jac = True,
                                  tol = None,
                                  callback = None,
                                  options = {'maxiter' : max_iter2})
#                                  options = {'maxiter' : max_iter2, 'gtol' : np.finfo('float64').min})
                                  #options = {'maxiter' : max_iter2, 'gtol' : np.finfo('float64').min})
                
                if result.fun < prev_obj:
                    cond = False
                    
            self.fitted_params = x_to_dict(result.x, M, K, D)
            self.initiated = False
                
        else:
            feed_dict = {self.gp.inputs['X'] : self.data[self.types[0]], self.gp.inputs['y'] : self.data[self.types[1]]}
        
            for param in self.fitted_params.keys():
                feed_dict[self.gp.params[param]] = self.fitted_params[param]
            
            prev_obj = self.session.run(self.gp.train_f, feed_dict)
            
            print prev_obj
            
            for n in xrange(fit_iter):
                x0 = self.init_params(init_method = init_method)
                    
                result = minimize(fun = train_step1,
                                  x0 = x0,
                                  method = method,
                                  jac = True,
                                  tol = None,
                                  callback = None,
                                  options = {'maxiter' : max_iter1})
                                  #options = {'maxiter' : max_iter1, 'gtol' : np.finfo('float64').min})
            
                x0 = result.x
        #        print x0[-2:-1]
                
                result = minimize(fun = train_step2,
                                  x0 = x0,
                                  method = method,
                                  jac = True,
                                  tol = None,
                                  callback = None,
                                  options = {'maxiter' : max_iter2})
                
                if result.fun < prev_obj:
                    self.fitted_params = x_to_dict(result.x, M, K, D)
                    prev_obj = result.fun
            
        print prev_obj
        
        self.train_result = result
        
        feed_dict = {self.gp.inputs['X'] : self.data[self.types[0]], self.gp.inputs['y'] : self.data[self.types[1]]}
        
        for param in self.fitted_params.keys():
            feed_dict[self.gp.params[param]] = self.fitted_params[param]
            
        self.l_square = self.session.run(self.gp.l_square, feed_dict)
        
    def test(self, A, z_star):
        X = self.data[self.types[0]]
        y = self.data[self.types[1]]
        max_fun = self.data[self.types[2]]
        beta = self.data['beta']
        
#        try:
#            if x_star.shape[1] is not self.D:
#                print 'Dimension error'
#                return None
#        except:
#            print 'Dimension error'
        
        feed_dict = {self.gp.inputs['X'] : X, self.gp.inputs['y'] : y, self.gp.acq_inputs['z_star'] : z_star, self.gp.acq_inputs['A'] : A,
                     self.gp.acq_inputs['max_fun'] : max_fun, self.gp.acq_inputs['beta'] : beta}
        
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
        
        for n in xrange(10):
            try:
                z_star = sample_enclosingbox(A, b, num_sample / 10)
                
#                x_star = np.matmul(z_star, A.transpose())
                
                feed_dict = {self.gp.inputs['X'] : X,
                             self.gp.inputs['y'] : y,
                             self.gp.acq_inputs['z_star'] : z_star,
                             self.gp.acq_inputs['A'] : A,
                             self.gp.acq_inputs['max_fun'] : max_fun,
                             self.gp.acq_inputs['beta'] : beta}
                
                for param in self.fitted_params.keys():
                    feed_dict[self.gp.params[param]] = self.fitted_params[param]
                    
                obj = self.session.run(self.gp.acq_f, feed_dict)
                
                temp = np.max(obj)
                
                if temp > max_obj:
                    z_next = z_star[np.argmax(obj)]
                    max_obj = temp
            except:
                print 'LLT error'
                continue
            
        x0 = z_next
#        print x0
        acq_step = ObjectiveWrapper1(self.acq_obj, A)
        
#        print acq_step(x0)
        
        result = minimize(fun = acq_step,
                          x0 = x0,
                          method = 'L-BFGS-B',
                          jac = True,
                          tol = None,
                          callback = None,
                          options = {'maxiter' : 100})
        
#        print result.x
#        print max_obj, np.negative(result.fun)
        
        return np.reshape(np.matmul(np.reshape(result.x, [1, -1]), A.transpose()), [1, -1]), result.fun
    
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

def test():
    import matplotlib.pyplot as plt
    import functions
    fun = functions.brainin(10)
    fun = functions.sinc_simple2()
    #fun = functions.sinc_simple10()
    #fun = functions.sinc_simple()
    R = BSLBO(fun, 2, 10, 1, 0.5, 100, ACQ_FUN = 'UCB')
    
    for i in xrange(10):
        data = R.data
        
    #    W = gp.fitted_params['mu'].transpose()
        W = R.fitted_params['mu']
        W = W[np.argmax(R.l_square), :].reshape([-1, R.D])
        W = W.transpose()
        
    #    W = fun.W
        WT = np.transpose(W)
        WTW = np.matmul(WT, W)
        B = np.transpose(np.linalg.solve(WTW, WT)) # D x K
        D = fun.D
        
        fx_min, fx_max = find_enclosingbox(B, np.sqrt(D) * np.ones([D, 1]))
        fx = np.linspace(fx_min, fx_max, num = 100).reshape([100, 1])
        fy = fun.evaluate(Z_to_Xhat(fx, W))[1]
        
        mu, var, EI = R.test(B, fx)
        
        next_x = R.iterate(1000)
        EI_scaled = preprocessing.MinMaxScaler((np.min(fy),np.max(fy))).fit_transform(EI.reshape([-1, 1]))
        
        plt.figure()
        plt.plot(fx, fy)
        plt.scatter(X_to_Z(data['X'], W), data['y'])
        plt.plot(fx, mu, 'k')
        plt.plot(fx, mu + np.sqrt(var), 'k:')
        plt.plot(fx, mu - np.sqrt(var), 'k:')
        plt.plot(fx, EI_scaled, '-.')
#        print X_to_Z(data['X'][-1], W)
        plt.scatter(X_to_Z(data['X'][-1], W), np.min(data['y']), marker = 'x', color = 'g')
        plt.title('N is ' + str(len(data['y'])))
        plt.show()
    
    return R
        
#R = test()

#import matplotlib.pyplot as plt
#for i in np.linspace(-5,5) :
#    plt.scatter(i, R.acq_obj(i, B)[0], color = 'k')
#    plt.scatter(i, R.acq_obj(i, B)[1], color = 'b')