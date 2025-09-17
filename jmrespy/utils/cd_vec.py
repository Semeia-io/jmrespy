"""
Function to compute derivates
"""

####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import pandas as pd
import numpy as np

def Cd_vec(x, f, eps=1e-03, *args):
    v_dict = args[0] #v_dict : Our dictionnary passed in *args
    n = len(x)
    res = np.zeros((n, n))
    ex = np.maximum(x, 1)
    for i in range(n):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] = x[i] + eps * ex[i]
        x2[i] = x[i] - eps * ex[i]
        diff_f = f(x1, v_dict) - f(x2, v_dict)
        diff_x = x1[i] - x2[i]
        res[:, i] = diff_f / diff_x
    return(0.5 * (res + res.T))