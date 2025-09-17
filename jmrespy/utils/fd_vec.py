"""
Function to compute derivates
"""

####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import numpy as np

def Diff_f_term(v_dict, i, x, f, f0, eps, ex):
    x1 = x.copy()
    x1[i] = x[i] + eps * ex[i]
    diff_f = f(x1, v_dict) - f0
    diff_x = x1[i] - x[i]
    return diff_f / diff_x


def Fd_vec(x, f, eps=1e-05, *args):
    v_dict = args[0] #v_dict : Our dictionnary passed in *args
    n = len(x)
    res = np.zeros((n, n))
    ex = np.maximum(x, 1)
    f0 = f(x, v_dict)
    for i in range(n):
        res[:, i] = Diff_f_term(v_dict, i, x, f, f0, eps, ex)
    return(0.5 * (res + res.T))
