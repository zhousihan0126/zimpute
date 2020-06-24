#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:07:16 2019

@author: sihanzhou
"""
import numpy as np
import math
#import Load_data

def Data_Filtering(Data_matrix_M):
    m=Data_matrix_M.shape[0]
    n=Data_matrix_M.shape[1]
    Delete_genes=[]
    #column:genes row:cells
    Min_expresion_value=3
    Min_expression_cells=3
    for j in range(0,n):
        gene_expression_times=0
        for i in range(0,m):
            if Data_matrix_M[i][j]>=Min_expresion_value:
                gene_expression_times+=1
        if gene_expression_times<Min_expression_cells:
            Delete_genes.append(j)
    M_Filtering=np.delete(Data_matrix_M,Delete_genes,axis=1)
    print("Data filtering...")
    return M_Filtering,Delete_genes

def Data_Normlization(Filtering):
    m=Filtering.shape[0]
    n=Filtering.shape[1]
    Row_sum_list=[]
    Row_sum_list_1=[]
    
    for i in range (0,m):
        Row_sum=0
        for j in range(0,n):
            Row_sum=Row_sum+Filtering[i][j]
        Row_sum_list.append(Row_sum)
        Row_sum_list_1.append(Row_sum)
    
    
    Row_sum_list_1.sort()
    half = len(Row_sum_list_1) // 2
    Row_median=(Row_sum_list_1[half] + Row_sum_list_1[~half]) / 2 
    
    for i in range(0,m):
        for j in range(0,n):
            if Row_sum_list[i]!=0:
                Filtering[i][j]=Filtering[i][j]*Row_median/Row_sum_list[i]
    M_Normlization=Filtering
    del Filtering
    for i in range(0,m):
        for j in range(0,n): 
            M_Normlization[i][j]=math.log2(M_Normlization[i][j]+1)
    print("Data normlization...")
    return M_Normlization,Row_median,Row_sum_list










                