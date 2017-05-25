#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
PESSL3
H[p(y|x)] : GP_K 사용
- E_p(W|D)[p(y|D,x,W)] : GP_W 사용
p(W|D) : MHGP 사용
"""

from MHGP_PESSL import MHGP
from GP import GP
import numpy as np
import tensorflow as tf
from sklearn import preprocessing, decomposition
from scipy.optimize import minimize
import settings
from util_func import X_to_Z, Z_to_Xhat
from sampling import find_enclosingbox, sample_enclosingbox
from pyDOE import lhs
from utils import ObjectiveWrapper, ObjectiveWrapper1
import GP_K, GP_W

class PESSL:
    def __init__(self, fun, K, N, Max_M, N_xlist = 10000):
#        N = 10
#        M = 10
#        K = 1
#        fun = functions.sinc_simple2()
#        ACQ_FUN = 'EI'
        self.FLOATING_TYPE = settings.dtype
        
        D = fun.D
        
        xlist = 2 * lhs(D, N_xlist) - 1
        np.random.shuffle(xlist)
        
        data = {}
        
        data['X'], data['y'] = fun.evaluate(xlist[range(N)])
        
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
        self.log_length = None
        self.N = N
        self.M = M
        self.K = K
        self.D = D
        self.fun = fun
        self.data = data
        self.types = types
        self.scaler = scaler
        
        self.Max_M = Max_M
        self.initiated = True
        self.xlist = np.delete(xlist, range(N), axis = 0)
        
        self.mhgp = MHGP(K, D)
        self.gp_W = GP_W.GP(D)
        self.gp_K = GP_K.GP(D)

        self.session_mhgp = tf.Session(graph = self.mhgp.graph)
        self.session_gpW = tf.Session(graph = self.gp_W.graph)
        self.session_gpK = tf.Session(graph = self.gp_K.graph)

        self.fitting()
        
    def train_obj(self, x):
        M = self.M
        D = self.D
        K = self.K
        param_dict = x_to_dict(x, M, K, D)
        
        feed_dict = {self.mhgp.inputs['X'] : self.data[self.types[0]], self.mhgp.inputs['y'] : self.data[self.types[1]]}
        
        for param in param_dict.keys():
            feed_dict[self.mhgp.params[param]] = param_dict[param]
        try:
            f, g = self.session_mhgp.run([self.mhgp.train_f, self.mhgp.train_g], feed_dict)
        except:
            f = np.finfo('float64').max
            g = np.array([np.nan for i in xrange(len(x))])
        return f, g

    def train_obj_K(self, x):
        feed_dict = {self.gp_K.inputs['X']: self.data[self.types[0]], self.gp_K.inputs['y']: self.data[self.types[1]],
                     self.gp_K.params['log_length']: x[0], self.gp_K.params['log_sigma']: x[1],
                     self.gp_K.params['log_noise']: x[2]}

        f, g = self.session_gpK.run([self.gp_K.train_f, self.gp_K.train_g], feed_dict)

        return f, g

    def entropy_W(self, x_star, W):
        feed_dict = {self.gp_W.inputs['X'] : self.data[self.types[0]],
                       self.gp_W.inputs['y'] : self.data[self.types[1]],
                       self.gp_W.acq_inputs['x_star'] : x_star,
                       self.gp_W.params['log_sigma'] : self.fitted_params['log_sigma_f'],
                       self.gp_W.params['log_noise']: self.fitted_params['log_tau'],
                       self.gp_W.params['W']: W}

        return self.session_gpW.run(self.gp_W.entropy_star, feed_dict)

    def entropy_K(self, x_star):
        feed_dict = {self.gp_K.inputs['X']: self.data[self.types[0]],
                       self.gp_K.inputs['y']: self.data[self.types[1]],
                       self.gp_K.acq_inputs['x_star']: x_star,
                       self.gp_K.params['log_sigma']: self.fitted_params['log_sigma_f'],
                       self.gp_K.params['log_noise']: self.fitted_params['log_tau'],
                       self.gp_K.params['log_length']: self.log_length}

        return self.session_gpK.run(self.gp_K.entropy_star, feed_dict)

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

        init_value['log_sigma_f'] = 0.5 * np.log(np.var(y) + 1e-6)
        
        init_value['log_tau'] = init_value['log_sigma_f'] - np.log(1e+3)

        return np.concatenate([init_value[param].reshape(-1) for param in ['Z', 'mu', 'log_Sigma', 'log_sigma_f', 'log_tau']])
        
    def fitting(self, init_method = 'pca', method = 'L-BFGS-B', max_iter1 = 200, max_iter2 = 0, fit_iter = 1):
        M = self.M
        D = self.D
        K = self.K
        
        train_step1 = ObjectiveWrapper(self.train_obj, 0)
        train_step2 = ObjectiveWrapper(self.train_obj, 1)
        train_step3 = ObjectiveWrapper(self.train_obj_K, 0)

        tau_lb = 1e-6

        x0 = self.init_params(init_method=init_method)

        bnds = [(None, None) for x in x0]
        bnds[-1] = (np.log(tau_lb), None)

        result = minimize(fun=train_step1,
                          x0=x0,
                          method=method,
                          bounds=tuple(bnds),
                          jac=True,
                          tol=None,
                          callback=None,
                          options={'maxiter': max_iter1})
        #options = {'maxiter' : max_iter1, 'gtol' : np.finfo('float64').min})

        x0 = result.x

        result = minimize(fun=train_step2,
                          x0=x0,
                          method=method,
                          bounds=tuple(bnds),
                          jac=True,
                          tol=None,
                          callback=None,
                          options={'maxiter': max_iter2})

        self.fitted_params = x_to_dict(result.x, M, K, D)
        
        self.train_result = result

        mhgp_loglike = -result.fun

        feed_dict = {self.mhgp.inputs['X'] : self.data[self.types[0]], self.mhgp.inputs['y'] : self.data[self.types[1]]}
        
        for param in self.fitted_params.keys():
            feed_dict[self.mhgp.params[param]] = self.fitted_params[param]
            
        self.l_square = self.session_mhgp.run(self.mhgp.l_square, feed_dict)

        x0 = np.random.normal(size = [3])
        x0[1] = self.fitted_params['log_sigma_f']
        x0[2] = self.fitted_params['log_tau']

        result = minimize(fun=train_step3,
                          x0=x0,
                          method=method,
                          jac=True,
                          tol=None,
                          callback=None,
                          options={'maxiter': max_iter2})

        self.log_length = result.x[0]

        gpK_loglike = -result.fun

        return mhgp_loglike, gpK_loglike
        
    def test(self, x_star, n_W_star):
        X = self.data[self.types[0]]
        y = self.data[self.types[1]]

        feed_dict = {self.mhgp.inputs['X'] : X, self.mhgp.inputs['y'] : y, self.mhgp.x_star : x_star}

        for param in self.fitted_params.keys():
            feed_dict[self.mhgp.params[param]] = self.fitted_params[param]

        mu, var = self.session_mhgp.run([self.mhgp.mu_W, self.mhgp.var_W], feed_dict)

        loc = self.fitted_params['mu']
        scale = np.exp(self.fitted_params['log_Sigma'])

        x_star_entropy2 = np.average([self.entropy_W(x_star, np.random.normal(loc, scale)) for i in xrange(n_W_star)],
                                     axis=0)

        x_star_entropy1 = self.entropy_K(x_star)
        
        return [mu, var, x_star_entropy1, x_star_entropy2]
    
    def finding_next(self, n_W_star):
        X = self.data[self.types[0]]
        y = self.data[self.types[1]]

        xlist = self.xlist
        
        try:
            x_star = xlist
            loc = self.fitted_params['mu']
            scale = np.exp(self.fitted_params['log_Sigma'])

            x_star_entropy2 = np.average([self.entropy_W(x_star, np.random.normal(loc, scale)) for i in xrange(n_W_star)], axis = 0)
            
            x_star_entropy1 = self.entropy_K(x_star)
            
            obj = (x_star_entropy1 - x_star_entropy2)
            
            x_next_idx = np.argmax(obj)
            
        except:
            print 'x_next error'
            
        return x_next_idx
    
    def iterate(self, n_W_star):
        M = self.M
        xlist = self.xlist
        data = self.data
        fun = self.fun
        types = self.types
        scaler = self.scaler
        
        x_next_idx = self.finding_next(n_W_star)
        next_x, next_y = fun.evaluate(xlist[[x_next_idx]])
        
        #print next_x_obj
        
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
        self.xlist = np.delete(xlist, [x_next_idx], axis = 0)
        
        self.fitting()
        
        return next_x
    
    def iterate_random(self):
        M = self.M
        xlist = self.xlist
        data = self.data
        fun = self.fun
        scaler = self.scaler
        
        next_x = np.random.uniform(xlist[[0]])
        next_x, next_y = fun.evaluate(next_x)
        
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
        self.xlist = np.delete(xlist, [0], axis = 0)
        
        self.fitting()
        
        return next_x
        

def test():
    import matplotlib.pyplot as plt
    import functions
    #fun = functions.brainin(10)
    #fun = functions.sinc_simple2()
    fun = functions.sinc_simple10()
    #fun = functions.sinc_simple()
    R = PESSL(fun, 1, 5, 100)
    
    for i in xrange(3):
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
        
        mu, var, EI1, EI2 = R.test(Z_to_Xhat(fx, W), 100)
        
        #for i in xrange(len(EI1)):
        #    print EI1[i] - EI2[i]
            
        EI = EI1-EI2
        #print np.max(EI)
        R.iterate(100)
        EI_scaled = preprocessing.MinMaxScaler((np.min(-1.),np.max(1.))).fit_transform(EI.reshape([-1, 1]))
        fy_scaled = preprocessing.MinMaxScaler((np.min(-1.),np.max(1.))).fit_transform(fy.reshape([-1, 1]))
        y_scaled = preprocessing.MinMaxScaler((np.min(-1.),np.max(1.))).fit_transform(data['y'].reshape([-1, 1]))
        
        plt.figure()
        plt.plot(fx, fy_scaled)
        plt.scatter(X_to_Z(data['X'], W), y_scaled)
        plt.plot(fx, mu, 'k')
        plt.plot(fx, mu + np.sqrt(var), 'k:')
        plt.plot(fx, mu - np.sqrt(var), 'k:')
        plt.plot(fx, EI_scaled, '-.')
#        print X_to_Z(data['X'][-1], W)
        plt.scatter(X_to_Z(data['X'][-1], W), -1., marker = 'x', color = 'g')
        plt.title('N is ' + str(len(data['y'])))
        plt.show()
    
    return R

def x_to_dict(x, M, K, D):
    result = {}
    result['Z'] = x[:(M * K)].reshape([M, K])
    result['mu'] = x[(M * K): (M * K + K * D)].reshape([K, D])
    result['log_Sigma'] = x[(M * K + K * D): (M * K + 2 * (K * D))].reshape([K, D])
    result['log_sigma_f'] = x[-2]
    result['log_tau'] = x[-1]

    return result

if __name__ == '__main__':
    R = test()

#import matplotlib.pyplot as plt
#for i in np.linspace(-5,5) :
#    plt.scatter(i, R.acq_obj(i, B)[0], color = 'k')
#    plt.scatter(i, R.acq_obj(i, B)[1], color = 'b')