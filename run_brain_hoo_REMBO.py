#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 00:16:24 2017

@author: vinnam
"""
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

from REMBO_high import REMBO
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
        
        R = REMBO(fun, 5, init_N, ACQ_FUN = 'UCB', SEARCH_METHOD = 'random', iter_fit = 500)
        
        for j in xrange(num_Iter - init_N):
            print 'RUN : ' + str(i + 1) + ' ' + 'ITER : ' + str(j + 1)
            R.iterate(500, 10000)
    
        result.append(R.data['y'])
    
    result = np.concatenate(result, axis = 1)
    
    file_path = "./result/brain_hoo_10D_REMBO"
    directory = os.path.dirname(file_path)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    np.save(file_path, result)
    
if __name__ == "__main__":
    main()