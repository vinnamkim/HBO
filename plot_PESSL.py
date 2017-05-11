#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 23:25:45 2017

@author: vinnam
"""

import os
import numpy as np
import matplotlib.pyplot as plt

path = '/home/vinnam/result_PESSL'
res = []

for f in os.listdir(path):
    filepath = os.path.join(path, f)
    
    if os.path.isfile(filepath):
        temp = np.load(filepath)
        res.append(temp[()]['S'])
    
mean = np.average(res, axis = 0)
var = np.var(res, axis = 0)

x = range(len(mean))

plt.plot(x, mean[:, 0], 'b')
plt.plot(x, mean[:, 1], 'g')
plt.fill_between(x, mean[:, 0] - np.sqrt(var[:, 0]), mean[:, 0] + np.sqrt(var[:, 0]), color = 'b', alpha = 0.1)
plt.fill_between(x, mean[:, 1] - np.sqrt(var[:, 1]), mean[:, 1] + np.sqrt(var[:, 1]), color = 'g', alpha = 0.1)
