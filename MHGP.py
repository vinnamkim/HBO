#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 16:41:35 2017

@author: vinnam
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sinc_2d
from sklearn import preprocessing
from sklearn import decomposition

class MHGP:
    def __init__(self, M, K, D, FLOATING_TYPE = 'float32', JITTER_VALUE = 1e-5, KL_TYPE = 'ML2', LEARNING_RATE = 1e-1):
        self.FLOATING_TYPE = FLOATING_TYPE
        self.M = M
        self.K = K
        self.D = D
        
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            ####### CONSTANTS #######
            
            # N, K, D, M are constants
            N_ = tf.placeholder(dtype = FLOATING_TYPE, shape = [], name = 'M')
            M_ = tf.constant(M, name = 'M', dtype = FLOATING_TYPE)
            D_ = tf.constant(D, name = 'D', dtype = FLOATING_TYPE)
            K_ = tf.constant(K, name = 'K', dtype = FLOATING_TYPE)
            pi_ = tf.constant(np.pi, name = 'PI', dtype = FLOATING_TYPE)
            
            
            ####### DATA #######
            
            # X is a N x D placeholder
            # y is a N x 1 placeholder
            # Z is a M x K variable
            # mu is a K x D variable 
            # Sigma is a K x D variable
            # tau is a variable
            # sigma_f is a variable
            X = tf.placeholder(name = 'X', shape = [None, D], dtype = FLOATING_TYPE)
            y = tf.placeholder(name = 'y', shape = [None, 1], dtype = FLOATING_TYPE)
            Z = tf.get_variable('Z', initializer = tf.random_uniform_initializer(), shape = [M, K], dtype = FLOATING_TYPE)
            mu = tf.get_variable('mu', initializer = tf.random_uniform_initializer(), shape = [K, D], dtype = FLOATING_TYPE)
            log_Sigma = tf.get_variable('log_Sigma', initializer = tf.random_uniform_initializer(), shape = [K, D], dtype = FLOATING_TYPE)
            log_sigma_f = tf.get_variable('log_sigma_f', initializer = tf.random_uniform_initializer(), shape = [], dtype = FLOATING_TYPE)
            log_tau = tf.get_variable('log_tau', initializer = tf.random_uniform_initializer(), shape = [], dtype = FLOATING_TYPE)
            
            self.inputs = {'N' : N_,
                          'X' : X,
                          'y' : y,
                          'Z' : Z,
                          'mu' : mu,
                          'log_Sigma' : log_Sigma,
                          'log_sigma_f' : log_sigma_f,
                          'log_tau' : log_tau}
            
            ####### TRANSFORMED PARAMETERS #######
            
            sigma_f_sq = tf.exp(2 * log_sigma_f)
            tau_sq = tf.exp(2 * log_tau)
            Sigma_sq = tf.exp(2 * log_Sigma)
            sigma_f_sq_div_tau_sq = tf.exp(2 * (log_sigma_f - log_tau))
            
            ####### VARIATIONAL LOWER BOUND #######
            
            # K_{u,u} is M x M matrix
            r = tf.reshape(tf.reduce_sum(tf.square(Z), 1), [-1, 1])
            K_uu = tf.exp(-0.5 * (r - 2 * tf.matmul(Z, tf.transpose(Z)) + tf.transpose(r))) # Check
            Lm = tf.cholesky(K_uu + JITTER_VALUE * tf.eye(M))
            
            # Psi1 is a M x M matrix
            div = tf.transpose(tf.expand_dims(tf.matmul(tf.square(X), tf.transpose(Sigma_sq)), 0)) # K x N x 1
            div_plusone = div + 1 # K x N x 1 # Check
            mu_x = tf.transpose(tf.expand_dims(tf.matmul(X, tf.transpose(mu)), 0)) # K x N x 1
            ZZ = tf.transpose(tf.expand_dims(Z, 0), [2, 0, 1]) # K x 1 x M
            
            Psi1 = tf.reduce_prod(tf.rsqrt(div_plusone) * tf.exp(- tf.square(mu_x - ZZ) / (2 * div_plusone)), axis = 0) # K x N x M
            
            # Psi2 is a M x M matrix
            ZZ_t = tf.transpose(ZZ, [0, 2, 1]) # K x M x 1
            twodiv_plusone = tf.transpose(tf.expand_dims(2 * div + 1, -1), [1, 0, 2, 3]) # N x K x 1 x 1
            Psi21 = tf.exp(-0.25 * tf.reduce_sum(tf.square(ZZ - ZZ_t), axis = 0)) # Check
            
            # Psi21 Check
            mu_xx = tf.transpose(tf.expand_dims(mu_x, -1), [1, 0, 2, 3]) # N x K x 1 x 1
            ZZZ = tf.expand_dims(0.5 * (ZZ + ZZ_t), 0) # 1 x K x M x M
            Psi22 = tf.reduce_sum(tf.reduce_prod(tf.rsqrt(twodiv_plusone) * tf.exp(-tf.square(mu_xx - ZZZ) / twodiv_plusone), axis = 1), axis = 0) # M x M
            
            Psi2 = Psi21 * Psi22 # M x M
            
            ## CHECK
            
            Psi1InvLm = tf.transpose(tf.matrix_triangular_solve(Lm, tf.transpose(Psi1))) # N x M
            
            C = tf.matrix_triangular_solve(Lm, tf.transpose(tf.matrix_triangular_solve(Lm, Psi2))) # M x M
            
            A = tau_sq * tf.eye(M) + sigma_f_sq * C + JITTER_VALUE * tf.eye(M) # M x M
            
            La = tf.cholesky(A) # M x M
            
            YPsi1InvLm = tf.matmul(tf.transpose(y), Psi1InvLm) # 1 x M
            
            YPsi1InvLmInvLa = tf.transpose(tf.matrix_triangular_solve(La, tf.transpose(YPsi1InvLm))) # 1 x M
            
            YYall = tf.reduce_sum(tf.square(y))
            
            F012 = - (N_ - M_) * log_tau - 0.5 * N_ * tf.log(2 * pi_) - \
            (0.5 / tau_sq) * YYall - tf.reduce_sum(tf.log(tf.diag_part(La)))
            
            F3 = (0.5 * sigma_f_sq_div_tau_sq) * tf.reduce_sum(tf.square(YPsi1InvLmInvLa))
            
            TrK = - (0.5 * sigma_f_sq_div_tau_sq) * (N_ - tf.reduce_sum(tf.diag_part(C)))
            
            F = F012 + F3 + TrK
            
            mu_Sigma_sq = tf.reduce_sum(tf.square(mu) + Sigma_sq, 1) # D 
            
            ####### KL DIVERGENCE ########
            
            if KL_TYPE is 'ML2':
                KL = - tf.reduce_sum(log_Sigma) + 0.5 * D_ * tf.reduce_sum(tf.log(mu_Sigma_sq)) # Check
                l_square = mu_Sigma_sq / D_
                
            #KL = - (D_ * 0.5 + alpha) * tf.reduce_sum(tf.log(2 * beta + tf.reduce_sum(tf.square(mu) + Sigma_sq, 1))) + 0.5 * tf.reduce_sum(tf.log(Sigma_sq))
            
            ####### TRAINING STEP #######
            
            OBJ_train = - (F - KL)
    
            opt = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
    
            train_op1 = opt.minimize(OBJ_train, var_list = [Z, mu, log_Sigma])
    
            train_op2 = opt.minimize(OBJ_train)
            
            ####### OUTPUTS #######
            
            self.outputs = {'F' : F,
                            'KL' : KL,
                            'OBJ' : OBJ_train,
                            'l_square': l_square,
                            'train_op1' : train_op1,
                            'train_op2' : train_op2}
        
    def fitting(self, data, Iter1 = 100, Iter2 = 500, init_method = 'pca'):
        FLOATING_TYPE = self.FLOATING_TYPE
        
        ####### INIT VALUES OF PARAMETERS #######
        
        X = data['X']
        y = data['y']
        N = data['N']
        M = self.M
        K = self.K
        D = self.D
        
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
        
        init_value['log_sigma_f'] = 0.5 * np.log(np.var(y))
        
        init_value['log_tau'] = 0.5 * np.log(np.var(y) / 100)
    
        with tf.Session(graph=self.graph) as sess:
            ####### INITIALIZATION #######
            
            sess.run(tf.global_variables_initializer())
            
            for key in init_value.keys():
                sess.run(self.inputs[key].assign(init_value[key]))
            
            feed_dict = {self.inputs['N'] : np.float(N),
                         self.inputs['X'] : X,
                         self.inputs['y'] : y}
            
            ####### TRAIN_STEP1 #######
            
            print('TRAIN_STEP1')
            
            for i in range(Iter1):
                try:
                    _, new_obj = sess.run([self.outputs['train_op1'], self.outputs['OBJ']], feed_dict)
                    
                    if i % 10 is 0: print(i, new_obj)
                    
                except:
                    break
            
            ####### TRAIN_STEP2 #######
            
            print('TRAIN_STEP2')
            
            for i in range(Iter2):
                try:
                    _, new_obj = sess.run([self.outputs['train_op1'], self.outputs['OBJ']], feed_dict)
                    
                    if i % 10 is 0: print(i, new_obj)
                    
                except:
                    break
        
        return [sess.run(self.inputs['mu']), sess.run(self.outputs['l_square'])]
            

def main(N = 200, D = 10, M = 50, K = 10, Iter1 = 100, Iter2 = 500):
    mhgp = MHGP(M, K, D)
    
    sim = sinc_2d.sinc_2d(N, D, 5, 0.01)
    data = sim.data()
    data['N'] = N
    
    fit = mhgp.fitting(data)
    
    W_hat = fit[0]
    
    lsq_hat = fit[1]
    print(lsq_hat)
    
    idx_list = np.argsort(lsq_hat)[::-1]
    
    X_p_hat = np.matmul(data['X'], np.transpose(W_hat[idx_list, :]))
    
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_X_p = min_max_scaler.fit_transform(data['X_p'])
    
    for i in range(2):
        plt.figure(figsize = (8, 16))
        scaled_X_p_hat = min_max_scaler.fit_transform(X_p_hat)
        for j in range(K):
            plt.subplot('42'+str(j))
            proj = plt.scatter(scaled_X_p_hat[:, j], data['Y'], label = 'proj')
            true = plt.scatter(scaled_X_p[:, i], data['Y'], label = 'true')
            scaled_X_p_hat = min_max_scaler.fit_transform(X_p_hat)
        scaled_X_p_hat = min_max_scaler.fit_transform(-X_p_hat)
        for j in range(K):
            plt.subplot('42'+str(j+4))
            proj = plt.scatter(scaled_X_p_hat[:, j], data['Y'], label = 'proj')
            true = plt.scatter(scaled_X_p[:, i], data['Y'], label = 'true')
        plt.legend(handles = [proj, true])
        plt.savefig('fig'+str(i)+'.png')
    
#    sim.show_fun()
    sim.show_Y()
    
main()