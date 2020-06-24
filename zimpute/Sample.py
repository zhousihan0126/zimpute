#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:50:04 2019

@author: sihanzhou
"""
import numpy as np
import random


def sample(M,sample_rate):
    
    num=M.shape[0]*M.shape[1]
    
    zeros=int(num*sample_rate)
    
    ones=num-zeros
    
    s=[0]*zeros+[1]*ones
    
    random.shuffle(s)       
    
    ss=np.array(s)
    
    result=ss.reshape(M.shape[0],M.shape[1])
    
    result=M*result
    
    return result


        
            



            
            
            
            
        
M=np.loadtxt(r"/Users/sihanzhou/Downloads/data_Guo_fix.tsv",delimiter="\t",skiprows=0)
     


np.savetxt('/Users/sihanzhou/Downloads/data_Guo_drop.tsv',sample(M,0.2), delimiter = '\t') 

    
