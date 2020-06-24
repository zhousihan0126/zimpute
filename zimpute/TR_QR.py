#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time


def Truncated_QR(X,r):
    
    # Maximum error of the calculation
    Error = 1e-6 
    
    #The maximum number of iterations
    Itmax=10
    
    #Get the number of rows and columns of datamatrix
    m,n=np.shape(X)
    
    #initialize LL,S,R
    #L,S,R is diagnol matrix
    
    L=np.eye(m,r)
    
    S=np.eye(r,r)
    
    R=np.eye(r,n)
    
    k=1 # The times of iteration
    
    while 1:
        
        #Q,D=QR(X*R.T)
        Q,D=np.linalg.qr(np.dot(X,R.T))
        
        #L is equal to the first r columns of Q
        L=Q[:,:r]
        
        #Q,D=QR(X.T*L)
        Q,D=np.linalg.qr(np.dot(X.T,L))
        
        R=np.transpose(Q)
        
        #D=(L.T*X*Q)^T
        D=np.transpose(np.dot(L.T,np.dot(X,Q)))
        
        #S is equal to the first r columns and rows of D
        S=np.transpose(D[:r,:r])
        
        #The iteration times plus 1
        k=k+1
        
        #||LSR-X||_F^2<error
        val=np.linalg.norm(np.dot(L,np.dot(S,R))-X,'fro')
        
        #Convergence conditions:val^2<error or k>the maximum iteration
        if pow(val,2) < Error or k > Itmax:
            break
        
    return L,S,R
        
#mxn l=mxr d=rxr r=rxn       
# =============================================================================
# # =============================================================================
# # ==============================bv ===============================================
A=np.random.random((6000,1000))
# #  
start =time.perf_counter()
# # 
L,S,R=Truncated_QR(X=A,r=5) #TR_QR method
# # 
end =time.perf_counter()
# # 
l,d,r=np.linalg.svd(A) #svd method
# # 
end1=time.perf_counter()
# # print(S,"\n","\n",np.dot(L,np.dot(S,R)))
# # print("\n")
# # print(np.diag(d),"\n","\n",np.dot(l,np.dot(np.diag(d),r)))
# # print("Time spent in one iteration(s)")
print("\n","TC_QR:",end-start,"\n","svd:",end1-end)
# #        
# =============================================================================
#     
# =============================================================================
    
    
    
    
    