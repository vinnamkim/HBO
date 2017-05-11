#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 16:41:35 2017

@author: vinnam
"""

import numpy as np
import tensorflow as tf
import settings
from sklearn import decomposition
from sampling import sample_enclosingbox

def acq_fun(mu_star, var_star, max_fun, beta, method):
    if method is 'EI':
        std_star = tf.sqrt(var_star)
        dist = tf.contrib.distributions.Normal(mu = 0., sigma = 1.)
        diff = (mu_star - max_fun)
        Z = diff / tf.sqrt(var_star)
        EI = diff * dist.cdf(Z) + std_star * dist.pdf(Z)
        return EI # L x 1
    elif method is 'UCB':
        return mu_star + tf.sqrt(beta) * tf.sqrt(var_star)
    
class MHGP:
    def __init__(self, K, D, KL_TYPE = 'ML2', LEARNING_RATE = 1e-1, ACQ_FUN = 'EI'):
        FLOATING_TYPE = settings.dtype
        JITTER_VALUE = settings.jitter
#        D = 10
#        K = 5
        
        self.FLOATING_TYPE = FLOATING_TYPE
        self.JITTER_VALUE = JITTER_VALUE
        self.ACQ_FUN = ACQ_FUN
        
        self.l_square = None
        
        self.K = K
        self.D = D
        
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            ####### VARIABLES #######
            # X is a N x D placeholder
            # y is a N x 1 placeholder
            # Z is a M x K variable
            # mu is a K x D variable 
            # Sigma is a K x D variable
            # tau is a variable
            # sigma_f is a variable
            X = tf.placeholder(name = 'X', shape = [None, D], dtype = FLOATING_TYPE)
            y = tf.placeholder(name = 'y', shape = [None, 1], dtype = FLOATING_TYPE)
            
            self.inputs = {'X' : X,
                          'y' : y}
            
            Z = tf.placeholder(name = 'Z', shape = [None, K], dtype = FLOATING_TYPE)
            mu = tf.placeholder(name = 'mu', shape = [K, D], dtype = FLOATING_TYPE)
            log_Sigma = tf.placeholder(name = 'log_Sigma', shape = [K, D], dtype = FLOATING_TYPE)
            log_sigma_f = tf.placeholder(name = 'log_sigma_f', shape = [], dtype = FLOATING_TYPE)
            log_tau = tf.placeholder(name = 'log_tau', shape = [], dtype = FLOATING_TYPE)
            
            self.vi_params = {'Z' : Z, 'mu' : mu, 'log_Sigma' : log_Sigma}
            
            self.kernel_params = {'log_sigma_f' : log_sigma_f, 'log_tau' : log_tau}
            
            self.params = {}
            self.params.update(self.vi_params)
            self.params.update(self.kernel_params)
            
            ####### CONSTANTS #######
            
            # N, K, D, M are constants
            N = tf.shape(y)[0]
            M = tf.shape(Z)[0]
            
            N_ = tf.cast(N, dtype = FLOATING_TYPE, name = 'N')
            M_ = tf.cast(M, dtype = FLOATING_TYPE, name = 'M')
            D_ = tf.constant(D, name = 'D', dtype = FLOATING_TYPE)
            #K_ = tf.constant(K, name = 'K', dtype = FLOATING_TYPE)
            pi_ = tf.constant(np.pi, name = 'PI', dtype = FLOATING_TYPE)
            I_M = tf.eye(M, dtype = FLOATING_TYPE)
            
            ####### TRANSFORMED PARAMETERS #######
            
            sigma_f_sq = tf.exp(2 * log_sigma_f)
            tau_sq = tf.exp(2 * log_tau)
            Sigma_sq = tf.exp(2 * log_Sigma)
            sigma_f_sq_div_tau_sq = tf.exp(2 * (log_sigma_f - log_tau))
            
            ####### VARIATIONAL LOWER BOUND #######
            
            # K_{u,u} is M x M matrix
            r = tf.reshape(tf.reduce_sum(tf.square(Z), 1), [-1, 1])
            K_uu = tf.exp(-0.5 * (r - 2 * tf.matmul(Z, tf.transpose(Z)) + tf.transpose(r))) # Check
            Lm = tf.cholesky(K_uu + JITTER_VALUE * I_M)
            
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
            
#            A = tau_sq * I_M + sigma_f_sq * C + JITTER_VALUE * I_M # M x M
            
            A = tau_sq * I_M + sigma_f_sq * C # M x M
            
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
    
#            opt = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
#    
#            train_op1 = opt.minimize(OBJ_train, var_list = [Z, mu, log_Sigma])
#    
#            train_op2 = opt.minimize(OBJ_train)
            
            ####### ACQUISITION FUNCTION INPUS #######
            
            z_star = tf.placeholder(name = 'z_star', shape = [None, None], dtype = FLOATING_TYPE)
            A = tf.placeholder(name = 'A', shape = [D, None], dtype = FLOATING_TYPE)
            max_fun = tf.placeholder(name = 'max_fun', shape = [], dtype = FLOATING_TYPE)
            beta = tf.placeholder(name = 'beta', shape = [], dtype = FLOATING_TYPE)
            
            self.acq_inputs = {'z_star' : z_star, 'A' : A, 'max_fun' : max_fun, 'beta' : beta}
            
            x_star = tf.matmul(z_star, tf.transpose(A))
            
            ####### x_star DISTRIBUTION #######
            
            mu_x_star = tf.transpose(tf.expand_dims(tf.matmul(x_star, tf.transpose(mu)), 0)) # K x L x 1
            
            div_star = tf.transpose(tf.expand_dims(tf.matmul(tf.square(x_star), tf.transpose(Sigma_sq)), 0)) # K x L x 1
            div_plusone_star = div_star + 1 # K x L x 1 # Check
            twodiv_plusone_star = tf.transpose(tf.expand_dims(2 * div_star + 1, -1), [1, 0, 2, 3]) # L x K x 1 x 1
            
            Psi1_star = tf.reduce_prod(tf.rsqrt(div_plusone_star) * tf.exp(- tf.square(mu_x_star - ZZ) / (2 * div_plusone_star)), axis = 0) # L x M
            
            Psi1_starInvLm = tf.matrix_triangular_solve(Lm, tf.transpose(Psi1_star)) # M x L
            
            Psi1_starInvLmInvLa = tf.matrix_triangular_solve(La, Psi1_starInvLm) # M x L
                                      
            mu_star = tf.squeeze(sigma_f_sq * tf.matmul(tf.transpose(Psi1_starInvLmInvLa), tf.transpose(YPsi1InvLmInvLa))) # L
            
            #### CHECK ####
                                
            mu_xx_star = tf.transpose(tf.expand_dims(mu_x_star, -1), [1, 0, 2, 3]) # L x K x 1 x 1
            
            Psi22_star = tf.reduce_prod(tf.rsqrt(twodiv_plusone_star) * tf.exp(-tf.square(mu_xx_star - ZZZ) / twodiv_plusone_star), axis = 1) # L x M x M
            
            Psi2_star1 = Psi21 * Psi22_star # L x M x M
            
            Psi2_star = tf.transpose(tf.reshape(tf.transpose(Psi2_star1, [0, 2, 1]), [-1, M])) # M x (L x M)
            
#            LmInv = tf.matrix_triangular_solve(Lm, I_M)
#            LaInv = tf.matrix_triangular_solve(La, I_M)
#            LaInvLmInv = tf.matmul(LaInv, LmInv)
#            
#            K_uu_Inv = tf.matmul(tf.transpose(LmInv), LmInv)
#            
#            A_Inv = tf.matmul(tf.transpose(LaInvLmInv), LaInvLmInv)
            
#            test_AA = tf.matmul(A_Inv, tf.matmul(Lm, tf.transpose(tf.matmul(Lm, A))))
#            test_BB = tf.matmul(tf.matrix_triangular_solve(tf.matmul(Lm, La), tf.transpose(Psi1)), y)
#            Alpha = tf.matrix_triangular_solve(tf.transpose(La), tf.matrix_triangular_solve(tf.transpose(Lm), tf.transpose(YPsi1InvLmInvLa), lower = False), lower = False)
#            Alpha = tf.matmul(tf.matmul(A_Inv, tf.transpose(Psi1)), y)
            # M x 1

            #1
#            Psi2_starInvLm = tf.matrix_triangular_solve(Lm, Psi2_star)
#            
#            var_star1 = tf.matrix_triangular_solve(La, Psi2_starInvLm)
#            
#            var_star1 = tf.transpose(tf.reshape(tf.transpose(var_star1), [-1, M, M]), [0, 2, 1])
#            
#            var_star1 = tf.transpose(tf.reshape(var_star1, [-1, M]))
#            
#            var_star1 = tf.matrix_triangular_solve(La, tf.matrix_triangular_solve(Lm, var_star1))
#            
#            var_star1 = tf.transpose(tf.reshape(tf.transpose(var_star1), [-1, M, M]), [0, 2, 1])
#            
#            var_star1 = sigma_f_sq * tf.trace(var_star1)
#            
#            var_star11 = tf.trace(tf.reshape(tf.transpose(tf.matmul(sigma_f_sq * A_Inv, Psi2_star)), [-1, M, M]))
#            
            ########################
            Psi2_starInvLm = tf.matrix_triangular_solve(Lm, Psi2_star)
            
            Psi2_starInvLm = tf.transpose(tf.reshape(tf.transpose(Psi2_starInvLm), [-1, M, M]), [0, 2, 1])
            
            InvLmPsi2_starInvLm = tf.matrix_triangular_solve(Lm, tf.transpose(tf.reshape(Psi2_starInvLm, [-1, M])))
            
            InvLmPsi2_starInvLm = tf.transpose(tf.reshape(tf.transpose(InvLmPsi2_starInvLm), [-1, M, M]), [0, 2, 1])
            
            
            InvLmPsi2_starInvLmLa = tf.matrix_triangular_solve(La, tf.transpose(tf.reshape(tf.transpose(InvLmPsi2_starInvLm, [0, 2, 1]), [-1, M])))
            
            InvLmPsi2_starInvLmLa = tf.transpose(tf.reshape(tf.transpose(InvLmPsi2_starInvLmLa), [-1, M, M]), [0, 2, 1])
            
            InvLaInvLmPsi2_starInvLmLa = tf.matrix_triangular_solve(La, tf.transpose(tf.reshape(InvLmPsi2_starInvLmLa, [-1, M])))
            
            InvLaInvLmPsi2_starInvLmLa = tf.transpose(tf.reshape(tf.transpose(InvLaInvLmPsi2_starInvLmLa), [-1, M, M]), [0, 2, 1])
            
            var_star1 = sigma_f_sq * tf.trace(InvLaInvLmPsi2_starInvLmLa)
            
#            var_star11 = tf.trace(tf.reshape(tf.transpose(tf.matmul(sigma_f_sq * A_Inv, Psi2_star)), [-1, M, M]))
            ####################
            
            #2
            var_star2 = sigma_f_sq * tf.trace(InvLmPsi2_starInvLm)
            
#            var_star22 = tf.trace(tf.reshape(tf.transpose(tf.matmul(sigma_f_sq * K_uu_Inv, Psi2_star)), [-1, M, M]))
            
            #3
            var_star3 = tf.square(sigma_f_sq) * tf.squeeze(tf.matmul(YPsi1InvLmInvLa, tf.transpose(tf.reshape(tf.matmul(YPsi1InvLmInvLa, tf.transpose(tf.reshape(InvLaInvLmPsi2_starInvLmLa, [-1, M]))), [-1, M]))))
            
#            var_star33 = tf.squeeze(tf.matmul(tf.reshape(tf.matmul(tf.transpose(sigma_f_sq * Alpha), Psi2_star), [-1, M]), sigma_f_sq * Alpha))
            
            var_star =  tau_sq * var_star1 - var_star2 + var_star3 + sigma_f_sq * tf.ones_like(mu_star) - tf.square(mu_star)
            
            F_acq = acq_fun(mu_star, var_star, max_fun, beta, method = ACQ_FUN)
            
#            ####### ACQUISITION FUNCTION INPUS #######
#            
#            z_star = tf.placeholder(name = 'z_star', shape = [None, K], dtype = FLOATING_TYPE)
#            max_fun = tf.placeholder(name = 'max_fun', shape = [], dtype = FLOATING_TYPE)
#            beta = tf.placeholder(name = 'beta', shape = [], dtype = FLOATING_TYPE)
#            
#            self.acq_inputs = {'z_star' : z_star, 'max_fun' : max_fun, 'beta' : beta}
#            
#            ####### z_star DISTRIBUTION #######
#            
#            zz_star = tf.expand_dims(z_star, 1)  # z_star : L x K -> L x 1 x K
#            
#            sigma_f = tf.exp(log_sigma_f)
#            
#            k_ustar_u = tf.exp(-0.5 * tf.reduce_sum(tf.square(Z - zz_star), axis = -1)) # L x M
#            
#            k_ustar_uInvLm = tf.matrix_triangular_solve(Lm, tf.transpose(k_ustar_u)) # M x L
#            
#            k_ustar_uInvLmInvLa = tf.matrix_triangular_solve(La, k_ustar_uInvLm) # M x L
#            
#            mu_star = tf.squeeze(sigma_f * tf.matmul(tf.transpose(k_ustar_uInvLmInvLa), tf.transpose(YPsi1InvLmInvLa))) # L x 1
#            
#            var_star = tf.ones_like(mu_star, dtype = FLOATING_TYPE) - tf.reduce_sum(tf.transpose(tf.square(k_ustar_uInvLm)), axis = -1) + \
#            tau_sq * tf.reduce_sum(tf.transpose(tf.square(k_ustar_uInvLmInvLa)), axis = -1)
#            
#            F_acq = acq_fun(mu_star, var_star, max_fun, beta, method = ACQ_FUN)
            
            ####### OUTPUTS #######
            
#            self.outputs = {'F' : F,
#                            'KL' : KL,
#                            'OBJ' : OBJ_train,
#                            'l_square': l_square,
#                            'train_op1' : train_op1,
#                            'train_op2' : train_op2,
#                            'mu_star' : mu_star,
#                            'var_star' : var_star,
#                            'F_acq' : F_acq}
            # Z : M x K, W : K x D
            # x_star : L x D
            # k_fstar_u : L x M

            self.train_f = OBJ_train
            self.train_g = tf.concat([tf.reshape(g, [-1]) for g in tf.gradients(OBJ_train, [Z, mu, log_Sigma, log_sigma_f, log_tau])], 0)
            self.l_square = l_square
            self.mu_star = mu_star
            self.var_star = var_star
            self.acq_f = F_acq
            self.acq_g = tf.gradients(F_acq, z_star)
            self.init = tf.global_variables_initializer()
        
#            self.debugs = locals()
            
    def fitting(self, data, types, N, M, Iter1 = 100, Iter2 = 500, init_method = 'pca'):
        FLOATING_TYPE = self.FLOATING_TYPE
        
        ####### INIT VALUES OF PARAMETERS #######
        
        X = data[types[0]]
        y = data[types[1]]
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
        
        init_value['Z'] = np.random.uniform(low = -np.sqrt(D), high = np.sqrt(D), size = [M, K])
        
        init_value['log_sigma_f'] = 0.5 * np.log(np.var(y) + 1e-6)
        
        init_value['log_tau'] = 0.5 * np.log((np.var(y) + 1e-6) / 100)
        
        with tf.Session(graph=self.graph) as sess:
            ####### INITIALIZATION #######
            
            sess.run(tf.global_variables_initializer())
            
            for key in init_value.keys():
                sess.run(self.params[key].assign(init_value[key]))
                #print key, np.sum(self.inputs[key].eval() - init_value[key])
            
            feed_dict = {self.inputs['X'] : X,
                         self.inputs['y'] : y}
            
            obj = sess.run(self.outputs['OBJ'], feed_dict)
            
            train_dict1 = {'train_op' : self.outputs['train_op1'],
                           'OBJ' : self.outputs['OBJ'],
                           'l_square' : self.outputs['l_square']}
            
            train_dict1.update(self.params)
            
            train_dict2 = {'train_op' : self.outputs['train_op2'],
                           'OBJ' : self.outputs['OBJ'],
                           'l_square' : self.outputs['l_square']}
            
            train_dict2.update(self.params)
            
            for key in self.fitted_params.keys():
                self.fitted_params[key] = self.params[key].eval()
            
            self.l_square = sess.run(self.outputs['l_square'], feed_dict)
            
            ####### TRAIN_STEP1 #######
            
#            print('TRAIN_STEP1')
            
            for i in xrange(Iter1):
                try:
                    train_step = sess.run(train_dict1, feed_dict)
                    
                    if train_step['OBJ'] < obj:
                        obj = train_step['OBJ']
                        self.l_square = train_step['l_square']
                        for key in self.fitted_params.keys():
                            self.fitted_params[key] = train_step[key]
                       
#                    if i % 10 is 0:
#                        print i, train_step['OBJ']
                    
                except Exception as inst:
                    print inst
                    break
            
            ####### TRAIN_STEP2 #######
            
#            print('TRAIN_STEP2')
            
            for i in xrange(Iter2):
                try:
                    train_step = sess.run(train_dict2, feed_dict)
                    
                    if train_step['OBJ'] < obj:
                        obj = train_step['OBJ']
                        self.l_square = train_step['l_square']
                        for key in self.fitted_params.keys():
                            self.fitted_params[key] = train_step[key]
                       
#                    if i % 10 is 0:
#                        print i, train_step['OBJ']
                    
                except Exception as inst:
                    print inst
                    break
    
    def test(self, data, types, x_star):
        X = data[types[0]]
        y = data[types[1]]
        max_fun = data[types[2]]
        beta = data['beta']
        
        try:
            if x_star.shape[1] is not self.D:
                print 'Dimension error'
                return None
        except:
            print 'Dimension error'
        
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            
            feed_dict = {self.inputs['X'] : X, self.inputs['y'] : y, self.acq_inputs['x_star'] : x_star, self.acq_inputs['max_fun'] : max_fun, self.acq_inputs['beta'] : beta}
            
            for key in self.fitted_params.keys():
                sess.run(self.params[key].assign(self.fitted_params[key]))
            
            mu, var, F_acq = sess.run([self.outputs['mu_star'], self.outputs['var_star'], self.outputs['F_acq']], feed_dict)
        
        return [mu, var, F_acq]
    
    def debug(self, data, types, x_star, var):
        X = data[types[0]]
        y = data[types[1]]
        max_fun = data[types[2]]
        beta = data['beta']
        
        try:
            if x_star.shape[1] is not self.D:
                print 'Dimension error'
                return None
        except:
            print 'Dimension error'
            
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            
            feed_dict = {self.inputs['X'] : X, self.inputs['y'] : y, self.acq_inputs['x_star'] : x_star, self.acq_inputs['max_fun'] : max_fun, self.acq_inputs['beta'] : beta}
            
            for key in self.fitted_params.keys():
                sess.run(self.params[key].assign(self.fitted_params[key]))
            
#            debugs = sess.run([self.debugs['var_star1'], self.debugs['var_star2'], self.debugs['var_star3'], self.debugs['K_uu_Inv'], self.debugs['A_Inv'], self.debugs['Psi2_star'], self.debugs['Lm'], self.debugs['La'], self.debugs['Psi1_star']], feed_dict)
            
            debugs = sess.run(self.debugs[var], feed_dict)
        
        return debugs
    
    def finding_next(self, data, types, A, b, num_sample):
        X = data[types[0]]
        y = data[types[1]]
        max_fun = data[types[2]]
        beta = data['beta']
        
        z_star = sample_enclosingbox(A, b, num_sample)
        
        x_star = np.matmul(z_star, A.transpose())
        
        feed_dict = {self.inputs['X'] : X,
                     self.inputs['y'] : y,
                     self.acq_inputs['x_star'] : x_star,
                     self.acq_inputs['max_fun'] : max_fun,
                     self.acq_inputs['beta'] : beta}
        
        with tf.Session(graph=self.graph) as sess:
            ## INITIALIZE
            sess.run(tf.global_variables_initializer())
            
            ## FITTED PARAMS INIT
            for key in self.fitted_params.keys():
                sess.run(self.params[key].assign(self.fitted_params[key]))
            try:    
                obj = sess.run(self.outputs['F_acq'], feed_dict = feed_dict)
            except Exception as inst:
                print inst

        x_next = x_star[np.argmax(obj)]
        
        return np.reshape(x_next, [1, -1]), obj

#def main(N = 100, D = 10, M = 10, K = 10, Iter1 = 100, Iter2 = 500):
#    mhgp = MHGP(M, K, D)
#    
#    sim = sinc_2d.sinc_2d(N, D, 5, 0.01)
#    data = sim.data()
#    
#    fit = mhgp.fitting(data)
#    
#    W_hat = mhgp.fitted_params['mu']
#    
#    lsq_hat = mhgp.l_square
#    
#    print(lsq_hat)
#    
#    idx_list = np.argsort(lsq_hat)[::-1]
#    
#    X_p_hat = np.matmul(data['X'], np.transpose(W_hat[idx_list, :]))
#    
#    min_max_scaler = preprocessing.MinMaxScaler()
#    scaled_X_p = min_max_scaler.fit_transform(data['X_p'])
#    
#    for i in xrange(2):
#        plt.figure(figsize = (8, 16))
#        
#        scaled_X_p_hat = min_max_scaler.fit_transform(X_p_hat)
#        for j in xrange(4):
#            plt.subplot('42'+str(j))
#            proj = plt.scatter(scaled_X_p_hat[:, j], data['y'], label = 'proj')
#            true = plt.scatter(scaled_X_p[:, i], data['y'], label = 'true')
#            
#        scaled_X_p_hat = min_max_scaler.fit_transform(-X_p_hat)
#        for j in xrange(4):
#            plt.subplot('42'+str(j+4))
#            proj = plt.scatter(scaled_X_p_hat[:, j], data['y'], label = 'proj')
#            true = plt.scatter(scaled_X_p[:, i], data['y'], label = 'true')
#            
#        plt.legend(handles = [proj, true])
#        plt.savefig('fig'+str(i)+'.png')
#    
#    sim.show_fun()
#    sim.show_Y()
#    
#main()