#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:57:04 2017

@author: vinnam
"""

import numpy as np
from functions import sinc_simple
import matplotlib.pyplot as plt

######################## EI 1D

fun = sinc_simple()
N = 10

data = fun.gen_data(N)
A = np.reshape(np.array([1.0]), [1,1])
data['A'] = A
data['b'] = np.ones([1, 1])

gp = GP(data['D'], SEARCH_METHOD = 'random')

for i in xrange(3):
    gp.fitting(data, Iter = 500)
    
    xx = np.linspace(-1, 1)
    mu, var, EI = gp.test(data, np.reshape(xx, [-1,1]))
    next_x = gp.finding_next(data)
    
    plt.figure()
    fx = np.linspace(-1,1)
    fy = np.squeeze(fun.evaluate(np.linspace(-1,1)))
    plt.plot(fx, fy)
    plt.scatter(data['X'], data['y'])
    plt.plot(xx, (max(fy) - min(fy)) / (max(EI) - min(EI)) * (EI) + min(fy), '-.')
    plt.plot(xx, mu, 'k')
    plt.plot(xx, mu + 2 * np.sqrt(var), 'k:')
    plt.plot(xx, mu - 2 * np.sqrt(var), 'k:')
    plt.scatter(next_x, np.mean(data['y']), marker = 'x')
    plt.title('N is ' + str(len(data['y'])))
    plt.show()    
    fun.update(next_x, data)

############################ EI 2D

from functions import sinc_simple2
import matplotlib.pyplot as plt

fun = sinc_simple2()
N = 1000

data = fun.gen_data(N)
A = np.eye(2)
b = np.ones([2, 1])
data['A'] = A
data['b'] = b

gp = GP(data['D'], SEARCH_METHOD = 'random')

for i in xrange(3):
    gp.fitting(data, Iter = 500)
    
    fx = np.random.uniform(-1,1, [100, 2])
    fy = fun.evaluate(fx)
    pfx = np.matmul(fx, fun.W)
    fxfy = np.concatenate([pfx, fy], axis = 1)
    fxfy = fxfy[fxfy[:, 0].argsort()]
    
    mu, var, EI = gp.test(data, fx)
    
    pfxpp = np.concatenate([pfx, np.reshape(mu, [-1, 1]), np.reshape(var, [-1, 1]), np.reshape(EI, [-1, 1])], axis = 1)
    pfxpp = pfxpp[pfxpp[:, 0].argsort()]
    
    next_x = gp.finding_next(data, Iter_random = 10000)
    
    plt.figure()
    plt.plot(fxfy[:, 0], fxfy[:, 1])
    plt.scatter(np.matmul(data['X'], fun.W), data['y'])
    plt.plot(pfxpp[:, 0], (np.max(fy) - np.min(fy)) * (pfxpp[:, 3]) / (np.max(EI) - np.min(EI)) + min(fy), '-.')
    plt.plot(pfxpp[:, 0], pfxpp[:, 1], 'k')
    plt.plot(pfxpp[:, 0], pfxpp[:, 1] + np.sqrt(pfxpp[:, 2]), 'k:')
    plt.plot(pfxpp[:, 0], pfxpp[:, 1] - np.sqrt(pfxpp[:, 2]), 'k:')
    plt.scatter(np.matmul(next_x, fun.W), -0.25, marker = 'x', color = 'g')
    plt.title('N is ' + str(len(data['y'])))
    plt.show()
        
    fun.update(next_x, data)

######################## UCB 1D

fun = sinc_simple()
N = 10

data = fun.gen_data(N)
A = np.reshape(np.array([1.0]), [1,1])
data['A'] = A
data['b'] = np.ones([1, 1])

gp = GP(data['D'], ACQ_FUN = 'UCB', SEARCH_METHOD = 'random')

for i in xrange(3):
    gp.fitting(data, Iter = 500)
    
    xx = np.linspace(-1, 1)
    mu, var, EI = gp.test(data, np.reshape(xx, [-1,1]))
    next_x = gp.finding_next(data)
    
    plt.figure()
    fx = np.linspace(-1,1)
    fy = np.squeeze(fun.evaluate(np.linspace(-1,1)))
    plt.plot(fx, fy)
    plt.scatter(data['X'], data['y'])
    plt.plot(xx, (max(fy) - min(fy)) / (max(EI) - min(EI)) * (EI) + min(fy), '-.')
    plt.plot(xx, mu, 'k')
    plt.plot(xx, mu + 2 * np.sqrt(var), 'k:')
    plt.plot(xx, mu - 2 * np.sqrt(var), 'k:')
    plt.scatter(next_x, np.mean(data['y']), marker = 'x')
    plt.title('N is ' + str(len(data['y'])))
    plt.show()    
    fun.update(next_x, data)

############################ UCB 2D

from functions import sinc_simple2
import matplotlib.pyplot as plt

fun = sinc_simple2()
N = 10

data = fun.gen_data(N)
A = np.eye(2)
b = np.ones([2, 1])
data['A'] = A
data['b'] = b

gp = GP(data['D'], ACQ_FUN = 'UCB', SEARCH_METHOD = 'random')

for i in xrange(3):
    gp.fitting(data, Iter = 500)
    
    fx = np.random.uniform(-1,1, [100, 2])
    fy = fun.evaluate(fx)
    pfx = np.matmul(fx, fun.W)
    fxfy = np.concatenate([pfx, fy], axis = 1)
    fxfy = fxfy[fxfy[:, 0].argsort()]
    
    mu, var, EI = gp.test(data, fx)
    
    pfxpp = np.concatenate([pfx, np.reshape(mu, [-1, 1]), np.reshape(var, [-1, 1]), np.reshape(EI, [-1, 1])], axis = 1)
    pfxpp = pfxpp[pfxpp[:, 0].argsort()]
    
    next_x = gp.finding_next(data, Iter_random = 10000)
    
    plt.figure()
    plt.plot(fxfy[:, 0], fxfy[:, 1])
    plt.scatter(np.matmul(data['X'], fun.W), data['y'])
    plt.plot(pfxpp[:, 0], (np.max(fy) - np.min(fy)) * (pfxpp[:, 3]) / (np.max(EI) - np.min(EI)) + min(fy), '-.')
    plt.plot(pfxpp[:, 0], pfxpp[:, 1], 'k')
    plt.plot(pfxpp[:, 0], pfxpp[:, 1] + np.sqrt(pfxpp[:, 2]), 'k:')
    plt.plot(pfxpp[:, 0], pfxpp[:, 1] - np.sqrt(pfxpp[:, 2]), 'k:')
    plt.scatter(np.matmul(next_x, fun.W), -0.25, marker = 'x', color = 'g')
    plt.title('N is ' + str(len(data['y'])))
    plt.show()
        
    fun.update(next_x, data)
    
    
    
    
    
#t = t.astype('float32')
xx = np.linspace(-1, 1)
#xx = xx.astype('float32')
#xx = data['X']
mu = np.zeros(len(xx))
var = np.zeros(len(xx))
EI = np.zeros(len(xx))
#LL = np.linalg.cholesky(np.exp(2 * gp.fitted_params['log_sigma']) * np.exp(-(0.5 / np.exp(2 * gp.fitted_params['log_length'])) * np.square(t - np.transpose(t))) + (1e-5 + np.exp(2 * gp.fitted_params['log_noise'])) * np.eye(N))
#LL = LL.astype('float32')

for idx in xrange(len(xx)):
    a,b,c = gp.test(data, np.reshape(xx[idx], [1,1]))
    mu[idx] = a
    var[idx] = b
    EI[idx] = c
#    ll = np.exp(2 * gp.fitted_params['log_sigma']) * np.exp(-(0.5 / np.exp(2 * gp.fitted_params['log_length'])) * np.square(t - np.transpose(np.reshape(xx[idx], [1,1]))))
#    mu[idx] = np.squeeze(np.matmul(np.transpose(np.linalg.solve(LL, ll)), np.linalg.solve(LL, data['y'])))
    
#     mu[idx] = np.squeeze(np.matmul(np.transpose(np.linalg.solve(L, ll)), np.linalg.solve(L, data['y'])))
fx = np.linspace(-1,1)
fy = np.squeeze(fun.evaluate(np.linspace(-1,1)))
plt.plot(fx, fy)
plt.scatter(data['X'], data['y'])
plt.plot(xx, (max(fy) - min(fy)) / (max(EI) - min(EI)) * (EI) + min(fy), '-.')
plt.plot(xx, mu, 'k')
plt.plot(xx, mu + 2 * var, 'k:')
plt.plot(xx, mu - 2 * var, 'k:')

next_x = gp.finding_next(data)
plt.plot(xx, EI)
plt.scatter(next_x, 0)



next_x = gp.finding_next(data)
next_x = gp.finding_next(data)
import numpy as np

N = 500
D = 1
X = np.random.uniform(size = [N, D])
length = 0.5
noise = 0.001
r = np.reshape(np.sum(np.square(X), axis = 1), [-1, 1])
d = r - 2 * np.matmul(X, np.transpose(X)) + np.transpose(r)
L = np.linalg.cholesky(np.exp(-(0.5 / np.square(length)) * d) + np.square(noise) * np.eye(N))
y = np.matmul(L, np.random.normal(size = [N, 1]))
data = {'X' : X, 'y' : y}

gp = GP(D)
gp.fitting(data)
gp.test(data, data['X'][0])

for param in gp.fitted_params:
    print param, np.exp(gp.fitted_params[param])
    
import matplotlib.pyplot as plt
plt.scatter(X, y)


############### HITANDRUN TEST #################

A = np.random.normal(size = [2, 2])
b = np.ones([2, 1])
N = 2
x = np.reshape(np.linspace(-N, N), [-1, 1])
Ax = np.transpose(A[:, 0] * x)

y1 = (b - Ax) / np.reshape(A[:, 1], [-1, 1])
y2 = (-b - Ax) / np.reshape(A[:, 1], [-1, 1])

x_star = np.reshape(np.array([0., 0.]), [1, -1])
result = []
result.append(x_star)

for i in xrange(100):
    x_star = hit_and_run(x_star, A, b, 2)
    result.append(x_star)
result = np.squeeze(result)    

plt.plot(x, y1[0, :])
plt.plot(x, y1[1, :])
plt.plot(x, y2[0, :])
plt.plot(x, y2[1, :])
plt.scatter(result[:, 0], result[:, 1])