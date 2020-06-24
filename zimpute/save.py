#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 11:57:46 2020

@author: sihanzhou
"""
import numpy as np
import re




#Function to save results
def save_result(outfile_path,W):
    
    #Save the datamatrix to the following path
    path=outfile_path
    
    #Matching file suffix names with regular expressions
    if re.match(".+\.csv",path,flags=0)!=None:
        
        np.savetxt(path,W.T, delimiter = ',') 
        
        print("saving result as .csv at"+str(path))
    
    elif re.match(".+\.tsv",path,flags=0)!=None:
        
        np.savetxt(path,W.T, delimiter = '\t') 
        
        print("saving result as .tsv at"+str(path))
    else:
        #If the file format is incorrect, output a prompt statement
        print("the format of input file is error")