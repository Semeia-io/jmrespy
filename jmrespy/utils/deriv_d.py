"""
Function used to derivate Variance-Covariance matrix of random effects 
"""

####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import numpy as np

def Deriv_d(d):
    ncz  = d.shape[0]
    ind = np.tril_indices(d.shape[0], k=0) #Lower triangle indices of matrix
    lst_mat = [] #List which will contain the matrix of deriv we will retrun
    for i in range(len(ind[0])):
        mat = np.zeros((ncz, ncz))
        ii = np.array([ind[0][i], ind[1][i]])
        mat[ii[0], ii[1]] = 1
        mat[ii[1], ii[0]] = 1
        lst_mat.append(mat)
    return(lst_mat)
