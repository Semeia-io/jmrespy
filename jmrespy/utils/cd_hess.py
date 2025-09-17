"""
This file contains Cd_hess, a function that computes Hessian matrix with finite difference (central method)
https://fr.wikipedia.org/wiki/Diff%C3%A9rence_finie
https://fr.wikipedia.org/wiki/Matrice_hessienne
"""

####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import pandas as pd
import numpy as np

#Loading bar
from tqdm import tqdm

def Cd_hess(x, f, h=1.4901161193847656e-08, *args):
    
    #v_dict : Our dictionnary passed in *args
    v_dict = args[0]

    #Preparation of indices and final matrix
    n = len(x)
    res = np.zeros((n, n))
    diag_ind = np.diag_indices(len(res.diagonal()), res.ndim) #Diagonal of Hessian
    triu_ind = np.triu_indices(res.shape[0], k=1) #Upper triangle of Hessian

    #Diagonal
    print("Diagonal")
    for ind in tqdm(diag_ind[0]):

        #Preparation of x vectors we need for computing hessian
        x1 = x.copy()
        x1[ind] = x1[ind] + h
        x2 = x.copy()
        x2[ind] = x2[ind] - h

        #Compute finite difference
        res[ind, ind] = (f(x1, v_dict) - 2*f(x, v_dict) + f(x2, v_dict))/(h*h)

    #Upper triangle of Hessian matrix
    print("Upper triangle")
    for ind in tqdm(range(len(triu_ind[0]))):

        #Preparation of x vectors we need for computing hessian
        i = triu_ind[0][ind]
        j = triu_ind[1][ind]
        x1 = x.copy()
        x1[[i,j]] = [x1[i] + h, x1[j] + h]
        x2 = x.copy()
        x2[[i,j]] = [x2[i] + h, x2[j] - h]
        x3 = x.copy()
        x3[[i,j]] = [x3[i] - h, x3[j] + h]
        x4 = x.copy()
        x4[[i,j]] = [x4[i] - h, x4[j] - h]

        #Compute finite difference
        res[i,j] = (f(x1, v_dict) - f(x2, v_dict) - f(x3, v_dict) + f(x4, v_dict))/(4*h*h)

    return(res + res.T - np.diag(np.diag(res)))