#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:56:00 2017

@author: vinnam
"""

from GP import GP
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import preprocessing
import settings
import tensorflow as tf
from scipy.optimize import minimize

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

class REMBO:
    def __init__(self, fun, K, N, ACQ_FUN, iter_fit):
#        import functions
#        fun = functions.sinc_simple2()
#        N = 10
#        K = 1
#        ACQ_FUN = 'EI'
#        iter_fit = 500
        D = fun.D
        
        data = {}
        
        A = np.random.normal(size = [D, K]).astype(dtype = settings.dtype)
        
        data['Z'] = np.random.uniform(low = -np.sqrt(D), high = np.sqrt(D), size = [N, K]).astype(dtype = settings.dtype)
        data['X'], data['y'] = fun.evaluate(np.matmul(data['Z'], np.transpose(A)))
        
        data['max_fun'] = np.max(data['y'])
        
        scaler = preprocessing.MinMaxScaler((-1,1))
        
        data['y_scaled'] = scaler.fit_transform(data['y'])
        data['scaled_max_fun'] = np.array(1.0, dtype = settings.dtype)
        
#        types = ['Z', 'y_scaled', 'scaled_max_fun']
        types = ['X', 'y_scaled', 'scaled_max_fun']
#        types = ['X', 'y', 'max_fun']
        
        data['beta'] = fun.beta(data)
        
        self.D = D
        self.K = K
        self.data = data
        self.fun = fun
        self.A = A
        self.types = types
        self.scaler = scaler
        
        gp = GP(D, K, ACQ_FUN = ACQ_FUN)
        self.gp = gp
        self.session = tf.Session(graph = self.gp.graph)
        
        var_y = np.var(data[types[1]])
        
        try:
            init_value = {'log_sigma' : 0.5 * np.log(var_y), 'log_noise' : 0.5 * np.log(var_y / 100)}
        except:
            init_value = {'log_sigma' : 0.5 * np.log(var_y + 1e-9), 'log_noise' : 0.5 * np.log((var_y + 1e-9) / 100)}
            
        self.fitted_params = {'log_length' : np.random.normal()}
        self.fitted_params.update(init_value)
        
        self.fitting(iter_fit)
        
    
    def train_obj(self, x):
        feed_dict = {self.gp.inputs['X'] : self.data[self.types[0]], self.gp.inputs['y'] : self.data[self.types[1]],
                     self.gp.params['log_length'] : x[0], self.gp.params['log_sigma'] : x[1], self.gp.params['log_noise'] : x[2]}
        
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
    
    def test(self, z_star):
        X = self.data[self.types[0]]
        y = self.data[self.types[1]]
        max_fun = self.data[self.types[2]]
        beta = self.data['beta']
        A = self.A
        
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
    
    def finding_next(self, num_sample):
        X = self.data[self.types[0]]
        y = self.data[self.types[1]]
        max_fun = self.data[self.types[2]]
        beta = self.data['beta']
        D = self.D
        K = self.K
        A = self.A
        
        max_obj = np.finfo(settings.dtype).min
        
        for n in xrange(10):
            z_star = np.random.uniform(low = -np.sqrt(D), high = np.sqrt(D), size = [num_sample // 10, K]).astype(dtype = settings.dtype)
            
            feed_dict = {self.gp.inputs['X'] : X, 
                         self.gp.inputs['y'] : y,
                         self.gp.acq_inputs['z_star'] : z_star, self.gp.acq_inputs['A'] : A, self.gp.acq_inputs['max_fun'] : max_fun, self.gp.acq_inputs['beta'] : beta}
                
            for param in self.fitted_params.keys():
                feed_dict[self.gp.params[param]] = self.fitted_params[param]
                    
            obj = self.session.run(self.gp.acq_f, feed_dict)
                
            temp = np.max(obj)
                
            if temp > max_obj:
                z_next = z_star[np.argmax(obj)]
                max_obj = temp
            
        x0 = z_next
#        print x0
        acq_step = ObjectiveWrapper1(self.acq_obj, A)
        
#        print acq_step(x0)
        
        bnds = tuple([(-np.sqrt(D), np.sqrt(D)) for i in xrange(K)])
        
        result = minimize(fun = acq_step,
                          x0 = x0,
                          method = 'L-BFGS-B',
                          bounds = bnds,
                          jac = True,
                          tol = None,
                          callback = None,
                          options = {'disp' : True, 'maxiter' : 100})
        
#        print result.x
#        print max_obj, np.negative(result.fun)
        
#        print result.x
#        print 'EI grad'
#        print -result.fun
        return np.reshape(result.x, [1, -1]), result.fun
    
    def fitting(self, iter_fit, method = 'L-BFGS-B'):
        ####### INIT VALUES OF PARAMETERS #######        
        x0 = [self.fitted_params['log_length'], self.fitted_params['log_sigma'], self.fitted_params['log_noise']]
        train_step1 = ObjectiveWrapper(self.train_obj, 0)
        train_step2 = ObjectiveWrapper(self.train_obj, 1)
        
        prev_f, _ = self.train_obj(x0)
        
#        print 'Unoptimized'
#        print x0
        
        result = minimize(fun = train_step1,
                          x0 = x0,
                          method = method,
                          jac = True,
                          tol = None,
                          callback = None,
                          options = {'maxiter' : iter_fit // 2, 'gtol' : np.finfo('float64').min})
    
        x0 = result.x

#        print 'Optimized 1'
#        print x0
        
        result = minimize(fun = train_step2,
                          x0 = x0,
                          method = method,
                          jac = True,
                          tol = None,
                          callback = None,
                          options = {'maxiter' : iter_fit // 2, 'gtol' : np.finfo('float64').min})
        
        next_f = result.fun
        
        if prev_f > next_f:
            self.fitted_params['log_length'] = result.x[0]
            self.fitted_params['log_sigma'] = result.x[1]
            self.fitted_params['log_noise'] = result.x[2]
        
#        print 'Optimized 2'
#        print result.x
        return
        
    def iterate(self, iter_fit, iter_next):
        A = self.A 
        fun = self.fun
        data = self.data
        scaler = self.scaler
        
        next_z, obj = self.finding_next(iter_next)
        
        next_x, next_y = fun.evaluate(np.matmul(next_z.reshape([1, -1]), A.transpose()))
        
        data['Z'] = np.append(data['Z'], next_z, axis = 0)
        data['X'] = np.append(data['X'], next_x, axis = 0)
        data['y'] = np.append(data['y'], next_y, axis = 0)
        
        data['y_scaled'] = scaler.fit_transform(data['y'])
        
        data['beta'] = fun.beta(data)
        
        self.fitting(iter_fit)
        
        return next_x
#        
#

def test():
    import functions
    import matplotlib.pyplot as plt
    
#    fun = functions.brainin(10)
    fun = functions.sinc_simple2()
    #fun = functions.sinc_simple10()
    #fun = functions.sinc_simple()
    R = REMBO(fun, 1, 10, ACQ_FUN = 'EI', iter_fit = 500)
    
    for i in xrange(3):
        data = R.data
        A = R.A
    
        fx = np.linspace(-np.sqrt(R.D),np.sqrt(R.D), 100).reshape([-1, 1])
        fx_high = np.matmul(fx, A.transpose())
        fy = fun.evaluate(fx_high)[1]
    
        mu, var, EI = R.test(fx)
        
        EI_scaled = preprocessing.MinMaxScaler((np.min(fy),np.max(fy))).fit_transform(EI.reshape([-1, 1]))
                                              
        next_x = R.iterate(500, 10000)
        
#        print 'EI grid'
#        print np.max(EI)
        
        plt.figure()
        plt.plot(fx, fy)
        plt.scatter(data['Z'], data['y'])    
        plt.plot(fx, EI_scaled, '-.')
        plt.plot(fx, mu, 'k')
        plt.plot(fx, mu + np.sqrt(var), 'k:')
        plt.plot(fx, mu - np.sqrt(var), 'k:')
        plt.scatter(data['Z'][-1], np.min(data['y']), marker = 'x', color = 'g')
        plt.title('N is ' + str(len(data['y'])))
        plt.show()
    
    return R
        
R = test()
#
#for i in xrange(10):
#    data = R.data
#    gp = R.gp
#    W = R.W
#    
#    fx = np.linspace(-np.sqrt(R.D),np.sqrt(R.D), 100).reshape([-1, 1])
#    fx_high = np.matmul(W, fx.transpose()).transpose()
#    fy = fun.evaluate(fx_high)[1]
#
#    mu, var, EI = gp.test(data, R.types, fx_high)
#    
#    EI_scaled = preprocessing.MinMaxScaler((np.min(fy),np.max(fy))).fit_transform(EI.reshape([-1, 1]))
#                                          
#    next_x = R.iterate(500, 10000)
#    
#    plt.figure()
#    plt.plot(fx, fy)
#    plt.scatter(data['Z'], data['y'])    
#    plt.plot(fx, EI_scaled, '-.')
#    plt.plot(fx, mu, 'k')
#    plt.plot(fx, mu + np.sqrt(var), 'k:')
#    plt.plot(fx, mu - np.sqrt(var), 'k:')
#    plt.scatter(data['Z'][-1], np.min(data['y']), marker = 'x', color = 'g')
#    plt.title('N is ' + str(len(data['y'])))
#    plt.show()
#        
#
#
#R.iterate(500, 10000)
#
#
#R.data
#
#    xx = np.linspace(-1, 1)
#    mu, var, EI = gp.test(data, np.reshape(xx, [-1,1]))
#    
#    
#    plt.figure()
#    fx = np.linspace(-1,1)
#    fy = np.squeeze(fun.evaluate(np.linspace(-1,1)))
#    plt.plot(fx, fy)
#    plt.scatter(data['X'], data['y'])
#    plt.plot(xx, (max(fy) - min(fy)) / (max(EI) - min(EI)) * (EI) + min(fy), '-.')
#    plt.plot(xx, mu, 'k')
#    plt.plot(xx, mu + 2 * np.sqrt(var), 'k:')
#    plt.plot(xx, mu - 2 * np.sqrt(var), 'k:')
#    plt.scatter(next_x, np.mean(data['y']), marker = 'x')
#    plt.title('N is ' + str(len(data['y'])))
#    plt.show()    
#    fun.update(next_x, data)
#
#N = 10
#D = 2
#K = 1
#
#test = preprocessing.MinMaxScaler((-1,1))
#
#a = np.random.normal(size = [100, 1])
#
#b = test.fit_transform(a)
#
#
#
#try:
#    data['y']
#except:
#    data['y'] = {}