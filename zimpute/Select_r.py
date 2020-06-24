#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:08:15 2019

@author: sihanzhou
"""

#the method about select r
import numpy as np



def select_r(Data_matrix_M):
    
    u,sigma_list,v_T=np.linalg.svd(Data_matrix_M,full_matrices=False)

    sigma_sum=0
    for i in range(0,len(sigma_list)):
        sigma_sum=sigma_sum+pow(sigma_list[i],2)
    total=0
    j=0
    while 1:
        total=total+pow(sigma_list[j],2)
        j+=1
        if total>0.9*sigma_sum:
            break
    r=j
    return r

