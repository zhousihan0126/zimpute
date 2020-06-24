#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sihanzhou
"""

import numpy as np
from scipy.fftpack import dctn,idctn
import time
import re
import math
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.decomposition import PCA
import random 
import pandas as pd


#Command line arguments
def Command():
    
    args = argparse.ArgumentParser(description = '======A scRNA imputation method base on low rank complemetion =====',epilog = '==============end============ ')
    
    #The path of input file
    args.add_argument("infile",         type = str,                help = "the path of inputfile")
    
    #The path of output file
    args.add_argument("outfile",        type = str,                  help = "the path of outputfile")
    
    #The truncated value in datamatrix
    args.add_argument("-r",'--rank',  type = int, dest = "rank",   help = u"the rank",default = 1)
    
    #The lambda parameter,the default value is 0.01 
    args.add_argument("-l", "--lambda",  type = float, dest = "Lambda",    help = "lambda",         default = 0.01,choices=[1000,100, 10,1,0.1,0.01,0.0001])
    
    #select if filtering the datamatrix
    args.add_argument('-f',"--filter",   type = str, dest = "filter",    help = "filtering the datamatrix",         default = 'F', choices=['T', 'F'])
    
    #select if normalizing the datamatrix
    args.add_argument("-n","--norm",type = str, dest = 'normalize', help = "normalizing the datamatrix",      default = "F", choices=['T', 'F'])    
    
    #construct a object of args
    args = args.parse_args()
    
    return args



def Load_Matrix(infile_path):
    
    #infile_path is string type
    path=infile_path
    
    #.+\.csv or .+\.tsv means any character + suffix(.csv or .tsv)
    #If the match is successful, it returns a Match, otherwise it returns None.
    if re.match(".+\.csv",infile_path,flags=0)!=None:
        
        print("the format of input file is .csv")
        
        data=pd.read_csv(open(path,"rb"), sep=',')
        
        Data_matrix_M=np.array(data.values[:,1:],dtype='float')
        
    
    elif re.match(".+\.tsv",infile_path,flags=0)!=None:
        
        print("the format of input file is .tsv")       
        
        data=pd.read_csv(open(path,"rb"), sep='\t')
        
        Data_matrix_M=np.array(data.values[:,1:],dtype='float')
        
    #If the file format is incorrect, output a prompt statement
    else:
        
        print("the format of input file is error")
        
    gene_list=list(data.values[:,0])#gene
        
    #cell_list=["cell"+str(i) for i in range(1,Data_matrix_M.shape[1]+1)]

    cell_list=list(data.columns)
           
    Data_matrix_M = Data_matrix_M.transpose(1,0)
    
    print("Load the data matrix...")
    
    return Data_matrix_M,gene_list,cell_list[1:]



def Data_Filtering(Data_matrix_M,Min_expression_value=3,Min_expression_cells=3):
    
    
    #Get the size of datamatrix
    m=Data_matrix_M.shape[0]
    
    n=Data_matrix_M.shape[1]
    
    #which rows are deleted
    Delete_genes=[]
    
    #column:genes row:cells

    
    #Expressed in at least three cells, and the expression value is not less than 3
    for j in range(0,n):
        #Initialized to 0
        gene_expression_times=0
        for i in range(0,m):            
            #Each time it is expressed, the number is increased by 1
            if Data_matrix_M[i][j]>=Min_expression_value:
                gene_expression_times+=1
        
        #Unqualified rows are deleted
        if gene_expression_times<Min_expression_cells:
            Delete_genes.append(j)
   
    #After filtering
    M_Filtering=np.delete(Data_matrix_M,Delete_genes,axis=1)
    
    #Output prompt
    print("Data filtering...")
    
    return M_Filtering,Delete_genes


def Data_Normlization(Filtering_M):
    #Get the size of datamatrix
    m,n=np.shape(Filtering_M)
    
    Row_sum_list=[]
    Row_sum_list_1=[]
    
    for i in range (0,m):
        Row_sum=0
        for j in range(0,n):
            Row_sum=Row_sum+Filtering_M[i][j]
       
        Row_sum_list.append(Row_sum)
        
        Row_sum_list_1.append(Row_sum)
    
    #compute the sum of every row
    Row_sum_list_1.sort()
    
    half = len(Row_sum_list_1) // 2
    
    Row_median=(Row_sum_list_1[half] + Row_sum_list_1[~half]) / 2 
    #compute the median of row
    
    for i in range(0,m):
        for j in range(0,n):
            
            if Row_sum_list[i]!=0:
                Filtering_M[i][j]=Filtering_M[i][j]*Row_median/Row_sum_list[i]
    
    M_Normlization=Filtering_M
    
    #Free up space
    del Filtering_M
    
    for i in range(0,m):
        for j in range(0,n): 
            #Mij=log2(Mij+1)
            #plus psudo 1
            M_Normlization[i][j]=math.log2(M_Normlization[i][j]+1)
    
    #Output prompt
    print("Data normlization...")
    
    return M_Normlization,Row_median,Row_sum_list


def Select_r(Data_matrix_M):
    
    #svd decomposition
    #decompose the datamatrix
    
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
# =============================================================================
# A=np.random.random((10,10))
#  
# start =time.perf_counter()
# 
# L,S,R=Truncated_QR(X=A,r=5) #TR_QR method
# 
# end =time.perf_counter()
# 
# l,d,r=np.linalg.svd(A) #svd method
# 
# end1=time.perf_counter()
# print(S,"\n","\n",np.dot(L,np.dot(S,R)))
# print("\n")
# print(np.diag(d),"\n","\n",np.dot(l,np.dot(np.diag(d),r)))
# print("Time spent in one iteration(s)")
# #print("\n","TC_QR:",end-start,"\n","svd:",end1-end)
#        
#     
# =============================================================================
    
#A low rank constraint matrix completion method based on truncated kernel norm about scRNA-seq data imputation 
def Impute(M,r=1,lamda=0.01,N_flag="F"):
    
    #Start the timer
    start =time.perf_counter()
    
    
    if N_flag=="T":
        
        M,sum_median,row_sum_list=Data_Normlization(M)
        
    
    #Get size of datamatrix
    m,n=np.shape(M)
    
    X=M
    
    #initialize parameters
    
    # μ=1/||M||_l2
    
# =============================================================================
#     if mu==None:
#     
    mu  =1./np.linalg.norm(M,2) 
# =============================================================================
           
    Omega = np.count_nonzero(M)

    #fraion is nonzero rates of datamatrix
    fraion = float(Omega)/(m*n)     
          
    rho = 1.2 + 1.8*fraion
    
    print("Imputation...")
    
    #set the maximum iteration 
    MAX_ITER = 200
    
    #row indicates cells,column indicates genes
    #m:row n:column
    m,n = np.shape(M)
    
    #initialization
    W = X
    
    # Z,Y is a zero matrix
    Y =np.zeros((m,n))
    
    Z = Y
    
    #E is a random mxn matrix
    E = np.random.random((m,n))
    
    #set error equal to e^-5
    #The lower the error, the longer the iteration time
    error = pow(10,-5)
  
# =============================================================================
#         svd method       
#        A,sigma,B_T=np.linalg.svd(X)
#        A=np.transpose(A[:,:r])
#        B=B_T[:r,:]
#        AB = np.dot(np.transpose(A),B)   
# =============================================================================

    for k in range(0,MAX_ITER):
        
        #===================================updata X=======================================
        
        #A is the left singular vector of X,B is the right singular vector of X
        #sigma is the singular value,
        A,sigma,B=Truncated_QR(X,r)
        
        #AB=A*B
        AB = np.dot(A,B)
        
        #the DCT( Discrete Cosine Transform) is a transform associated with the Fourier transform
        #inverse discrete cosine transform(IDCT)
        tem = 1/2*(W - Y/mu+idctn((E+Z/mu),norm='ortho'))
        
        lastX = X
        
        #u,s,v=svd(tem)
        u,sigma,v= np.linalg.svd(tem,full_matrices=0)
                
        ss = sigma-(1/mu/2)
        
        s2 = np.clip(ss,0,max(ss))
        
        #X=u*diag(s2)*v
        X = np.dot(u,np.dot(np.diag(s2,k=0),v) )
        
        #if ||X'-X||_F/||M||_F<error break
        if(np.linalg.norm(X-lastX,'fro')/np.linalg.norm(M,'fro')<error):
            break
    
        #=====================================updata W====================================
        
        lastW=W
       
        W = (AB+Y+mu*X)/mu
        
        #M_observed is a bool matrix,M_observedij=1 means the gene in this cell is expressed
        M_observed=np.float64(M>0)
        
        #M_noobserved is a bool matrix,M_noobservedij=1 means the expression value is equal to 0
        M_noobserved=np.ones((m,n))-M_observed
        
        #W=P_omega^(M)+P_omega(M)
        W=W*M_noobserved+M*M_observed
        
        #if ||W'-W||_F/||M||_F<error break
        if(np.linalg.norm(W-lastW,'fro')/np.linalg.norm(M,'fro')<error):
            break   
      
        #===================================update E==================================
        
        temp = dctn(X,norm='ortho')-Z/mu
        
        d_i=[]
        
        for i in range(0,m):
            
            #||temp[i]||_2
            row_L2_norm=np.linalg.norm(temp[i],ord=2,keepdims=False)
            
            if row_L2_norm>(lamda/mu):           
                
                #append (||temp[i]||_2-lamda/mu)/||temp[i]||_2-lamda/mu
                d_i.append((row_L2_norm-(lamda/mu))/row_L2_norm)
            else:
            
                #d_i is a list with diagnal elements
                d_i.append(0) 
        
        # make d_i to be diagnal matrix D,D is a mxm matrix
        D=np.diag(d_i,k=0)
        
        #E=D*temp
        E=np.dot(D,temp)
        
        #if ||X-W||_F/||M||_F<error break
        if(np.linalg.norm(X-W,'fro')/np.linalg.norm(M,'fro')<error):
            break
        
    #=============================updata Y and Z========================================
    
        Y = Y+mu*(X-W)
        
        Z = Z+mu*(E-dctn(X,norm='ortho'))
        
        #print the times of iteration
    
        #print("iterate"+str(k)+("times"))
   
# =============================================================================================
#         μ is dynamic penalty parameter, we using the adaptive update strategy to calculate μ. 
#         When u becomes ρμ(ρ>0), the convergence speed of the algorithm will increase.         
# =============================================================================================
        
        #val = μ*max(||X'-X||_F,||W'-W||_F)/||M||_F
        val = mu*max(np.linalg.norm(X-lastX, 'fro'), np.linalg.norm(W-lastW, 'fro')) / np.linalg.norm(M, 'fro')
        
        #if val<10^-3:μ=rho*μ
        if (val < pow(10,-3)):    
            
            mu = rho*mu; 
              
        mu = min(mu, pow(10,10));   
        # max_mu = 10e10       
    
    
    if N_flag=="T":
        
        #Data recovery
        #W.shape[0] represents the number of rows
        #W.shape[1] represents the number of columns
        for i in range(0,W.shape[0]):
            for j in range(0,W.shape[1]):                        
                #wij=2^(wij-1)/sum_median*row_sum
                W[i][j]=round((pow(2,W[i][j])-1)/sum_median*row_sum_list[i],1)                
  
# =============================================================================
    #Set to 0 when Wij> 0.5
    W_nonzero=np.float64(W>0.5)         
    
    W=W*W_nonzero          
    
    #End timer      
    end = time.perf_counter()
    
    #print the running time
    print('Running time: %s Seconds'%(end-start))
    
    return W.T

#Function to save results
def Save_result(outfile_path,M_pred,gene_list,cell_list,F_flag="F"):
    
    #Save the datamatrix to the following path
    path=outfile_path
    
    if F_flag=="T":
        
        m,n=np.shape(M_pred)
        
        gene_list=["gene"+str(i) for i in range(1,m)]
        
        cell_list=["cell"+str(i) for i in range(1,n)]
        
        
    
    #Matching file suffix names with regular expressions
    if re.match(".+\.csv",path,flags=0)!=None:
        
        df=pd.DataFrame(M_pred,index=gene_list,columns=cell_list)

        df.to_csv(path, sep=',', header=True, index=True)
        
        print("saving result as .csv at"+str(path))
    
    elif re.match(".+\.tsv",path,flags=0)!=None:
        
        df=pd.DataFrame(M_pred,index=gene_list,columns=cell_list)

        df.to_csv(path, sep='\t', header=True, index=True)

        print("saving result as .tsv at"+str(path))
    
    else:
        #If the file format is incorrect, output a prompt statement
        print("the format of input file is error")


#draw the figure of example with different lambda
def Example_lambda_pic():
        
    #Set font size and format
    font3 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 11,
    }
    
    #The value of truncated rank    
    x = [1,2,10,70,150,300]
    
    #The value of lambda
    lamda1 =[0.124,0.124,0.1260,0.1263,0.1265,0.1269]
    
    lamda2  = [0.11,0.112,0.113,0.115,0.117,0.152]
    
    lamda3 =[0.1012,0.1018,0.1081,0.109,0.11,0.24]
    
    lamda4 =[0.1014,0.1021,0.1105,0.1106,0.12,0.3]
    
    #Drawing a line chart
    plt.plot(x, lamda1, marker='>', ms=4,label='λ=1')
    
    plt.plot(x, lamda2, marker='d', ms=4,label='λ=0.1')
    
    plt.plot(x, lamda3, marker='^', ms=4,label='λ=0.01')
    
    plt.plot(x, lamda4, marker='X', ms=4,label='λ=0.001')
    
    
    plt.legend()  
    #Make the legend effective
    
    
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    #Set the bottom distance parameter
   
    #Set labels for X-axis and y-axis
    plt.xlabel('r',font3) 
    plt.ylabel("Relative error",font3) 
    plt.xlim(1,300)
    
    #Set the title
    plt.title("The relative error with different lambda") #标题
    
    
    plt.show()
        

#draw the figure of example with different mu 
def Example_mu_pic():
    
    #Set font size and format
    font3 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 11,
    }
    
    #The value of truncated rank  
    x = [1,2,10,70,150,300,500]
    
    #The value of mu
    mu1 =[0.4544,0.4548,0.4553,0.4561,0.4563,0.4563,0.4563]
    
    mu2  = [0.4289,0.4292,0.4305,0.4315,0.4317,0.4318,0.4329]
    
    mu3 =[0.3345,0.3356,0.3397,0.3418,0.3507,0.3525,0.3584]
    
    mu4 =[0.1059,0.1104,0.1134,0.1135,0.1217,0.1353,0.1652]
        
    #Drawing a line chart 
    plt.plot(x, mu1, marker='>', ms=3,label=' μ=0.1')
    
    plt.plot(x, mu2, marker='d', ms=3,label=' μ=0.01')
    
    plt.plot(x, mu3, marker='^', ms=3,label=' μ=0.001')
    
    plt.plot(x, mu4, marker='X', ms=3,label=' μ=0.0001')
       
    plt.legend()  
    #Make the legend effective
    
    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    #Set the bottom distance parameter
   
    #Set labels for X-axis and y-axis    
    plt.xlabel('r',font3) #X轴标签
    plt.ylabel("Relative error",font3) #Y轴标签
    
    #Set the X-axis and Y-axis ranges
    plt.xlim(1,500)
    plt.ylim(0,0.7)
    
    #Set the X-axis scale value
    plt.xticks([10,70,150,300,500])
    #plt.yticks([0,0.1,0.2,0.3,0.4,0.6,1])
    
    #Set the title
    plt.title("The relative error with different μ") #标题
    
    plt.show()
        
def Relative_error(M_pred,M_obes):
    
    #error=||M^-M||_F/||M||_F
    relative_error=np.linalg.norm(M_pred-M_obes,'fro')/np.linalg.norm(M_obes,'fro')      

    return relative_error      

#Visualize the results
def tSNE_Visualize(Matrix_raw,Matrix_impute,Target_group,celltype_list,n_components=30):
     
    #The format of Target_group is a list with 1 column like:[0 0 1 1 2 2 1 0] 
    #different numbers mean different cell types
    #cell 1 is 0 type,cell 2 is 0 type
    #celltype_list:["cell A","cell B","cell C"]
    #0=cell A,1=cell B,2=cell C
        
    #Set font size and format
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 11,
    }
    
    #Normalize the raw matrix and zimpute matrix
    raw,m,l=Data_Normlization(Matrix_raw)
    
    zim,m,l=Data_Normlization(Matrix_impute)

    #reduct dimention with PCA
    estimator = PCA(n_components=n_components)
    
    raw=estimator.fit_transform(raw)
    
    estimator = PCA(n_components=n_components)
    
    zim=estimator.fit_transform(zim)
        
    #reduct dimention with tSNE
    X_raw = TSNE(n_components=2,early_exaggeration=12,learning_rate=200,n_iter=2000).fit_transform(raw)
    
    X_zim = TSNE(n_components=2,early_exaggeration=12,learning_rate=200,n_iter=2000).fit_transform(zim)
  
    #Set value of colors  
    color=["skyblue","mediumaquamarine","lightseagreen",
           "goldenrod","mediumslateblue","mediumseagreen",
           "hotpink","darkkhaki","violet","lightcoral",
           "green","red","yellow","black","pink","blue",
           "skyblue","orange",'lavender', 'lavenderblush',       
           'lawngreen','lemonchiffon','lightblue','lightcoral',
           'lightcyan','lightgoldenrodyellow','lightgreen',
           'lightgray','lightpink','lightsalmon','lightseagreen',
           'lightskyblue','lightslategray','lightsteelblue',
           'lightyellow','lime']    
    
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3,figsize=(8,3))
    
    #Visualize
    #Scatter plots of different clusters with different colors
    for i in range(0,X_raw.shape[0]):
        ax1.scatter(X_raw[i,0],X_raw[i,1],s=10,c=color[int(Target_group[i])])
        
    for i in range(0,X_zim.shape[0]):
        ax2.scatter(X_zim[i,0],X_zim[i,1],s=10,c=color[int(Target_group[i])])
  
    #compute the score of clustering
    #use unsupervised metric method
    s1=metrics.silhouette_score(X_raw, Target_group, metric='euclidean')
   
    s2=metrics.silhouette_score(X_zim, Target_group, metric='euclidean')
    
    #Set labels for X-axis and y-axis and title
    ax1.set_xlabel("tSNE1",size=10)
    
    ax1.set_ylabel("tSNE2",size=10)
    
    ax1.set_title('Raw:'+str(round(s1,3)), font1)
    
    #Set labels for X-axis and y-axis and title
    ax2.set_xlabel("tSNE1",size=10)
    
    ax2.set_ylabel("tSNE2",size=10)
    
    ax2.set_title('zimpute:'+str(round(s2,3)), font1)
    
    #Remove the ax3 
    ax3.remove()
    
    #Set the format of legend
    patches = [ mpatches.Patch(color=color[i], label="{:s}".format(celltype_list[i]) ) for i in range(0,len(celltype_list))] 
    
    #Set legend
    plt.legend(handles=patches, bbox_to_anchor=(1.9,0.85), ncol=1,prop={'size':10})
    
    #Set the position of legend
    fig.subplots_adjust(hspace=0.38,wspace = 0.38)
    
    plt.show()
    
def Example_sigma_pic(Matrix):
    

    font3= {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 13,
    }
     
    u,sigma,v=np.linalg.svd(Matrix,full_matrices=False)
    
        
    def formatnum1(x, pos):
        return '$%.1f$x$10^{6}$' % (x/max(10000))
    
  
    #formatter1 = FuncFormatter(formatnum1)

    #plt.yaxis.set_major_formatter(formatter1) 
    
    #The meaning of the parameters in the plot are the horizontal axis value, vertical axis value, color, transparency, and label.
    plt.plot(range(1,len(sigma)+1), sigma,c="sandybrown",lw=2)
    
    #Set labels for X-axis and y-axis and title
    plt.xlabel("The numbers of singular value",size=12)
    
    plt.ylabel("The singular value",size=12)
    
    plt.title('The trend of singular value', font3)
       
    #plt.ylim(0,5000)
 
    plt.show()


#This is a random sampling function, input dropout rate and raw matrix
def Sample(M,sample_rate):
    
    #The size of Matrix
    num=M.shape[0]*M.shape[1]
    
    #How many dropout values 
    zeros=int(num*sample_rate)
    
    ones=num-zeros
    
    
    s=[0]*zeros+[1]*ones
    
    #Randomly arranged
    random.shuffle(s)       
    
    ss=np.array(s)
    
    #reshape the matrix
    result=ss.reshape(M.shape[0],M.shape[1])
    
    result=M*result
    
    return result

#show the relative error with diferent dropout
def Show_error_plot():
    
    font2 = {'family' : 'times New Roman',
    'weight' : 'normal',
    'size' : 12,
    }
    
    
    # example data   
    dropout_rates=np.array([10,30,50,70])
    
    #The relative error of different method
    dropout_1_error=np.array([0.0785,0.2044,0.4677,0.7940])
    
    zimpute_1_error=np.array([0.0256 ,0.0545,0.1029, 0.1868])
    
    scimpute_1_error=np.array([0.0485,0.1223,0.3098,0.7188])
    
    SAVER_1_error=np.array([0.2014,0.2819,0.5131	,0.8253])
    
    MAGIC_1_error=np.array([0.2158,0.2318,0.3662,0.7152])
    
    
    ls = '-'
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    
    #Drawing a line chart with difeerent methods
    ax[0,0].plot(dropout_rates, zimpute_1_error, marker="^", linestyle=ls,label='zimpute')
    
    ax[0,0].plot(dropout_rates, scimpute_1_error,marker="1", linestyle=ls,label='scimpute')
    
    ax[0,0].plot(dropout_rates, SAVER_1_error,marker="s", linestyle=ls,label='SAVER')
    
    ax[0,0].plot(dropout_rates, MAGIC_1_error, marker="x",linestyle=ls,label='MAGIC')
    
    ax[0,0].plot(dropout_rates, dropout_1_error, marker="2",linestyle=ls,label='dropout')
    
    #Set X-label,y-label 
    ax[0, 0].set_xlabel("Dropout rate",size=11)
    
    ax[0, 0].set_ylabel("Relative error",size=11)
    
    #Set title
    ax[0, 0].set_title('mn=2xe5', font2)
    
    #Set position and format of legend
    ax[0,0].legend(loc=6, bbox_to_anchor=(0.01,0.78),prop={'size':9})
    
    #Set the scale range
    ax[0,0].set_ylim((0,1))
    ax[0, 0].set_xlim((10,70))
    # including upper limits
    
    #The relative error with different method
    dropout_2_error=np.array([0.0850,0.2061,0.4594,0.7864])
    
    zimpute_2_error=np.array([0.0275 ,0.0546 , 0.1046,0.1942])
    
    scimpute_2_error=np.array([0.0595,0.1550,0.3470,0.7668])
    
    SAVER_2_error=np.array([0.2245,0.2999,0.5142,0.8174])
    
    MAGIC_2_error=np.array([ 0.1997,0.2232,0.3793,0.7238])
    
    
    #Drawing a line chart with difeerent methods
    ax[0,1].plot(dropout_rates, zimpute_2_error, marker="^", linestyle=ls,label='zimpute')
    
    ax[0,1].plot(dropout_rates, scimpute_2_error,marker="1", linestyle=ls,label='scimpute')
    
    ax[0,1].plot(dropout_rates, SAVER_2_error,marker="s", linestyle=ls,label='SAVER')
    
    ax[0,1].plot(dropout_rates, MAGIC_2_error, marker="x",linestyle=ls,label='MAGIC')
    
    ax[0,1].plot(dropout_rates, dropout_2_error, marker="2",linestyle=ls,label='dropout')
    
    
    #Set X-label,y-label,title ,legend 
    ax[0, 1].set_xlabel("Dropout rate",size=11)
    
    ax[0, 1].set_ylabel("Relative error",size=11)
    
    ax[0, 1].set_title('mn=1xe6', font2)
    
    #Set position and format of legend
    ax[0,1].legend(loc=6, bbox_to_anchor=(0.12,0.78),prop={'size':9})
    
    #Set the scale range
    ax[0, 1].set_xlim((10,70))
    ax[0,1].set_ylim((0,1))
    
    dropout_3_error=np.array([0.2412,0.5091,0.8198,0.9616])
    
    zimpute_3_error=np.array([0.1424,0.2124,0.3140,0.4689])
    
    scimpute_3_error=np.array([ 0.2367,0.4220,0.7196,0.9570])
    
    SAVER_3_error=np.array([0.2936,0.5342,0.8354,0.9743])
    
    MAGIC_3_error=np.array([0.3705,0.4813, 0.7773,0.9499])
   
    #Drawing a line chart with difeerent methods
    ax[1,0].plot(dropout_rates, zimpute_3_error, marker="^", linestyle=ls,label='zimpute')
    
    ax[1,0].plot(dropout_rates, scimpute_3_error,marker="1", linestyle=ls,label='scimpute')
    
    ax[1,0].plot(dropout_rates, SAVER_3_error,marker="s", linestyle=ls,label='SAVER')
    
    ax[1,0].plot(dropout_rates, MAGIC_3_error, marker="x",linestyle=ls,label='MAGIC')
    
    ax[1,0].plot(dropout_rates, dropout_3_error, marker="2",linestyle=ls,label='dropout')
    
    #Set X-label,y-label,title ,legend 
    ax[1, 0].set_xlabel("Dropout rate",size=11)
    ax[1, 0].set_ylabel("Relative error",size=11)
    
    ax[1, 0].set_title('mn=2xe6', font2)
    
    #Set position and format of legend
    ax[1,0].legend(loc=6, bbox_to_anchor=(0.01,0.78),prop={'size':9})
    
    #Set the scale range
    ax[1,0].set_ylim((0,1))
    ax[1, 0].set_xlim((10,70))
   
    dropout_4_error=np.array([0.2456,0.5203,0.8282,0.9661])
    
    zimpute_4_error=np.array([0.1632,0.2313,0.3058,0.6667])
    
    scimpute_4_error=np.array([0.2550,0.4994,0.7943,0.9592])
    
    SAVER_4_error=np.array([0.3082,0.5505,0.8449,0.9873])
    
    MAGIC_4_error=np.array([0.3332,0.4725,0.7902,0.9552])
    
    
    #Drawing a line chart with difeerent methods
    ax[1,1].plot(dropout_rates, zimpute_4_error, marker="^", linestyle=ls,label='zimpute')
    
    ax[1,1].plot(dropout_rates, scimpute_4_error,marker="1", linestyle=ls,label='scimpute')
    
    ax[1,1].plot(dropout_rates, SAVER_4_error,marker="s", linestyle=ls,label='SAVER')
    
    ax[1,1].plot(dropout_rates, MAGIC_4_error, marker="x",linestyle=ls,label='MAGIC')
    
    ax[1,1].plot(dropout_rates, dropout_4_error, marker="2",linestyle=ls,label='dropout')
    
    #Set X-label,y-label,title 
    ax[1, 1].set_xlabel("Dropout rate",size=11)
    ax[1, 1].set_ylabel("Relative error",size=11)
    
    ax[1, 1].set_title('mn=2xe7', font2)
   
    #Set position and format of legend
    ax[1,1].legend(loc=6, bbox_to_anchor=(0.01,0.78),prop={'size':9})
    
    #Set the scale range
    ax[1,1].set_xlim((10,70))
    ax[1,1].set_ylim((0,1))
    
    x_major_locator=plt.MultipleLocator(10)
    
    #Set X-tick interval
    ax[0, 0].xaxis.set_major_locator(x_major_locator)
    
    ax[0, 1].xaxis.set_major_locator(x_major_locator)
    
    ax[1, 0].xaxis.set_major_locator(x_major_locator)
    
    ax[1, 1].xaxis.set_major_locator(x_major_locator)
    
    #Set overall title
    fig.suptitle("Relative error with different scales",fontsize=13)
    
    #Adjust the distance between subplots
    fig.subplots_adjust(hspace=0.35,wspace = 0.28)
   
    plt.show()
 

#Heatmap to show the result after imputation. 
#The horizontal axis represents the cells and the vertical axis represents the genes.       
def Heatmap(M_raw,M_Impu,gene_list,cell_list,interval=[0,20,10,35],show_label="T"):

    font3 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 15,
    }
    m=M_raw.shape[0]
    n=M_raw.shape[1]
    for i in range(0,m):
        for j in range(0,n):                
            M_raw[i][j]=math.log10(M_raw[i][j]+1)
            M_Impu[i][j]=math.log10(M_Impu[i][j]+1)
    
  
    M=M_raw[interval[0]:interval[1],interval[2]:interval[3]]
    M1=M_Impu[interval[0]:interval[1],interval[2]:interval[3]]
    
    
    #cell_list=["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10"]
    
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 12))
    #ax1.set_ylabel("Genes",size=14)
    ax1.set_xlabel("Cells",size=14)
    
    ax1.set_title('Before zimpute', font3)
    #ax1.set_xticklabels(cell_list,rotation=0, fontsize='small')
    if show_label=="T":
        cell_list=cell_list[interval[0]:interval[1]]       
        gene_list=gene_list[interval[2]:interval[3]]
 
        ax1.set_yticklabels(gene_list,rotation=0, fontsize=10)
        ax1.set_xticklabels(cell_list,rotation=-80, fontsize=8)
    
            
        ax2.set_yticklabels(gene_list,rotation=0,fontsize=10)
        ax2.set_xticklabels(cell_list,rotation=-80,fontsize=8)
        ytick=range(0,len(gene_list))
        xtick=range(0,len(cell_list))
    
     #if the show_label==T use xtick and ytick   
        ax1.set_yticks(ytick)
    
        ax2.set_yticks(ytick)
    
        ax1.set_xticks(xtick)
    
        ax2.set_xticks(xtick)
    
    else:
    #if the show_label==F ,do not set the xtick and ytick 
    
        ax1.set_yticks([])
    
        ax2.set_yticks([])
    
        ax1.set_xticks([])
    
        ax2.set_xticks([])
        
    #ax2.set_ylabel("Genes",size=14)
    ax2.set_xlabel("Cells",size=14)
    ax2.set_title('After zimpute', font3)
    
 
    #ax2.set_xticklabels(cell_list,rotation=0, fontsize='small')
    
    pcm1 = ax1.imshow(M.T,interpolation='nearest',cmap="bwr")
    pcm2 = ax2.imshow(M1.T,interpolation='nearest',cmap="bwr")
    
    #Color bars are placed horizontally
    fig.colorbar(pcm1, ax=ax1,orientation="horizontal")
    fig.colorbar(pcm2, ax=ax2,orientation="horizontal")
    #plt.setp(ax1.get_xticklabels(), visible=False)
    #plt.setp(ax1.get_yticklabels(), visible=False)
    #plt.setp(ax2.get_xticklabels(), visible=False)
    #plt.setp(ax2.get_yticklabels(), visible=False)
 
    fig.subplots_adjust(hspace=0.3,wspace = 0.28)
    
    plt.show()            

def p_rate(raw,drop,zimpute):

    m,n=np.shape(raw)
    
    A_raw=np.float64(raw>0)
    
    A_drop=np.float64(drop>0)
    A_zimpute=np.float64(zimpute>0)
    
    zero_expression_numbers=m*n-np.count_nonzero(A_raw)
    
    drop_position=A_raw-A_drop
    
    drop_numbers=np.count_nonzero(drop_position)
    
    #impute_position=A_zimpute-A_drop
    
    #R=drop_position-impute_position
    
    R=A_raw-A_zimpute
    
    noimpute_numbers=np.count_nonzero(np.float64(R>0))
    
    error_impute_numbers=np.count_nonzero(np.float64(R<0))
    
    t_rate=1-noimpute_numbers/drop_numbers
    n_rate=1-error_impute_numbers/zero_expression_numbers
    
    return t_rate,n_rate        
