"""
Function to transform our d_vc matrix whith Cholesky
"""

####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import pandas as pd
import numpy as np

def Chol_transf(x):

    #Matrix -> Vec
    if x.ndim != 1:
        diag_ind = np.diag_indices(len(x.diagonal()), x.ndim)
        u = np.linalg.cholesky(x)
        u[diag_ind] = np.log(u[diag_ind])
        u_flat = u[np.tril_indices(u.shape[0], k=0)]
        return(u_flat)

    #Vec -> Matrix
    else:
        nx = len(x)
        k = round((- 1 + np.sqrt(1+8*nx)) / 2)
        mat = np.zeros((k, k))
        mat[np.tril_indices(mat.shape[0], k=0)] = x
        diag_ind = np.diag_indices(len(mat.diagonal()), mat.ndim)
        mat[diag_ind] = np.exp(mat[diag_ind])
        ret = dict(
            res = np.matmul(mat, mat.T),
            l = mat[np.tril_indices(mat.shape[0], k=0)]
        )
        
        return(ret)