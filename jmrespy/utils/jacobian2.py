"""
Function used to compute jacobian matrice
"""

####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import pandas as pd
import numpy as np

def Jacobian2(l, ncz):
    #ind = np.tril_indices(np.zeros((ncz, ncz)).shape[0], k=0) #Upper triangle indices of matrix
    ind = np.triu_indices(np.zeros((ncz, ncz)).shape[0], k=0) #Upper triangle indices of matrix
    n_ind = len(ind[0])
    idj = np.array([i for i in range(n_ind)])
    r_ind = np.squeeze(np.array(np.where(ind[0] == ind[1])))
    lind = []
    for i in range(1, len(r_ind)+1):
        tt = np.zeros((ncz-i+1, ncz-i+1))
        #tt[np.tril_indices(tt.shape[0], k=0)] = [val for val in range(r_ind[i-1], n_ind)]
        tt[np.triu_indices(tt.shape[0], k=0)] = [val for val in range(r_ind[i-1], n_ind)]
        tt = tt + tt.T
        np.fill_diagonal(tt, tt.diagonal()/2)
        lind.append(tt)

    out = np.zeros((n_ind, n_ind))
    for g in range(ncz):
        #g_ind = idj[g == ind[1]]
        g_ind = idj[g == ind[0]]
        vals = l[g_ind]
        for j in g_ind:
            k = np.squeeze(np.where(j == g_ind))
            sel_rows = (lind[g][k,]).astype(int)
            sel_cols = np.repeat(j, len(lind[g][k,]))
            out[sel_rows, sel_cols] = vals[0] * vals if j in r_ind else vals
    out[r_ind,] = 2*out[r_ind,]
    col_ind = np.zeros((ncz, ncz)).astype(int)
    #col_ind[np.tril_indices(col_ind.shape[0], k=0)] = [val for val in range(len(l))]
    col_ind[np.triu_indices(col_ind.shape[0], k=0)] = [val for val in range(len(l))]
    col_ind = col_ind.T
    #return(out[:, col_ind[np.triu_indices(col_ind.shape[0], k=0)]])
    return(out[:, col_ind[np.tril_indices(col_ind.shape[0], k=0)]])