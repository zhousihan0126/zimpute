#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:47:54 2020

@author: sihanzhou
"""

import numpy as np
import Load_data
import zhousihan

#raw=np.array([[2,2,0,0],[2,2,0,0],[2,2,0,0]])

raw=Load_data.Load_Matrix(r"/Users/sihanzhou/Desktop/work/data_csv/sim_500x2000_fix.csv")

#drop=np.array([[2,0,0,0],[2,0,0,0],[2,0,0,0]])
drop=Load_data.Load_Matrix(r"/Users/sihanzhou/Desktop/work/data_csv/sim_2_50_fix.csv")

#zimpute=zhousihan.Impute(drop,1,0.01,"T")
zimpute=Load_data.Load_Matrix(r"/Users/sihanzhou/Desktop/work/data_csv/scimpute_2_50_fix.csv")
#zimpute=zimpute.T

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

print(t_rate,n_rate)

