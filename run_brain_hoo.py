#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 00:16:24 2017

@author: vinnam
"""

from BSLBO import BSLBO
import functions
import os
import numpy as np

def main():
    runs = 20
    init_N = 5
    num_Iter = 500
    
    result = []
    
    for i in xrange(runs):
        fun = functions.brainin(10)
        
        R = BSLBO(fun, 5, init_N, init_N, 4, 0.5, 100, ACQ_FUN = 'UCB')
        
        R = BSLBO(fun, 5, init_N, 100, 4, 0.5, 100, ACQ_FUN = 'UCB')
        
        for j in xrange(num_Iter - init_N):
            print 'RUN : ' + str(i + 1) + ' ' + 'ITER : ' + str(j + 1)
            R.iterate(10000)
    
        result.append(R.data['y'])
    
    result = np.concatenate(result, axis = 1)
    
    file_path = "./result/brain_hoo_10D_BSLBO"
    directory = os.path.dirname(file_path)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    np.save(file_path, result)
    
if __name__ == "__main__":
    main()