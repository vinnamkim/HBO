#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 12:46:14 2017

@author: vinnam
"""

import matplotlib.pyplot as plt
import numpy as np
import os


file_path1 = "./result/brain_hoo_10D_BSLBO_UCB_yscaled.npy"
file_path2 = "./result/brain_hoo_10D_REMBO_UCB_yscaled.npy"

BSLBO = np.load(file_path1)
REMBO = np.load(file_path2)

optimum = -0.397887

BSLBO = optimum - BSLBO
REMBO = optimum - REMBO

BSLBO_ln = np.log10(BSLBO)
REMBO_ln = np.log10(REMBO)

x = np.arange(1, 501)
plt.fill_between(x, np.average(BSLBO_ln, axis = 1) - np.sqrt(np.var(BSLBO_ln, axis = 1)), np.average(BSLBO_ln, axis = 1) + np.sqrt(np.var(BSLBO_ln, axis = 1)), color = 'blue', alpha = 0.5)
plt.plot(x, np.average(BSLBO_ln, axis = 1), 'b-')

plt.fill_between(x, np.average(REMBO_ln, axis = 1) - np.sqrt(np.var(REMBO_ln, axis = 1)), np.average(REMBO_ln, axis = 1) + np.sqrt(np.var(REMBO_ln, axis = 1)), color = 'yellow', alpha = 0.5)
plt.plot(x, np.average(REMBO_ln, axis = 1), 'g-')

BSLBO_regret = np.zeros_like(BSLBO)
REMBO_regret = np.zeros_like(REMBO)

for i in xrange(BSLBO_regret.shape[0]):
    BSLBO_regret[i, :] = np.min(BSLBO[0:(i+1), :], axis = 0)
    
for i in xrange(REMBO_regret.shape[0]):
    REMBO_regret[i, :] = np.min(REMBO[0:(i+1), :], axis = 0)
    
BSLBO_regret_ln = np.log10(BSLBO_regret)
REMBO_regret_ln = np.log10(REMBO_regret)

fig = plt.figure()

plt.fill_between(x, np.average(BSLBO_regret_ln, axis = 1) - np.sqrt(np.var(BSLBO_regret_ln, axis = 1)), np.average(BSLBO_regret_ln, axis = 1) + np.sqrt(np.var(BSLBO_regret_ln, axis = 1)), color = 'blue', alpha = 0.5)
plt.plot(x, np.average(BSLBO_regret_ln, axis = 1), 'b-')
plt.fill_between(x, np.average(REMBO_regret_ln, axis = 1) - np.sqrt(np.var(REMBO_regret_ln, axis = 1)), np.average(REMBO_regret_ln, axis = 1) + np.sqrt(np.var(REMBO_regret_ln, axis = 1)), color = 'yellow', alpha = 0.5)
plt.plot(x, np.average(REMBO_regret_ln, axis = 1), color = 'yellow')

fig.savefig('Brainin_10D.png', dpi = 1000)

fig = plt.figure()

for i in xrange(BSLBO_regret_ln.shape[1]):
    ax1 = plt.subplot(4,5,i+1)
    ax1.plot(BSLBO_regret_ln[:, i])
    ax1.set_ylim([np.min(BSLBO_regret_ln), np.max(BSLBO_regret_ln)])
    
fig.savefig('BSLBO_regret.png', dpi = 1000)

fig = plt.figure()

for i in xrange(REMBO_regret_ln.shape[1]):
    ax2 = plt.subplot(4,5,i+1)
    ax2.plot(REMBO_regret_ln[:, i])
    ax2.set_ylim([np.min(REMBO_regret_ln), np.max(REMBO_regret_ln)])

fig.savefig('REMBO_regret.png', dpi = 1000)

plt.figure()
for i in xrange(REMBO_regret_ln.shape[1]):
    plt.plot(x, REMBO_regret_ln[:, i], 'y')
    plt.plot(x, BSLBO_regret_ln[:, i], 'b')

