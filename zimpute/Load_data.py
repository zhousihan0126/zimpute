#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 19:40:20 2019

@author: sihanzhou
"""

import numpy  as np
import re

def Load_Matrix(infile_path):
    
    #infile_path is string type
    path=infile_path
    
    #.+\.csv or .+\.tsv means any character + suffix(.csv or .tsv)
    #If the match is successful, it returns a Match, otherwise it returns None.
    if re.match(".+\.csv",infile_path,flags=0)!=None:
        
        print("the format of input file is .csv")
        
        Data_matrix_M = np.loadtxt(open(path,"rb"),delimiter=",",skiprows=0)
    
    elif re.match(".+\.tsv",infile_path,flags=0)!=None:
        
        print("the format of input file is .tsv")       
        
        Data_matrix_M = np.loadtxt(open(path,"rb"),delimiter="\t",skiprows=0)
    
    #If the file format is incorrect, output a prompt statement
    else:
        
        print("the format of input file is error")
           
    Data_matrix_M = Data_matrix_M.transpose(1,0)
    
    print("Load the data matrix...")
    
    return Data_matrix_M