"""
Called in Survfit(), 
"""
####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import numpy as np

# Probability
from scipy.stats import norm
from scipy.stats import chi2




####################### -------------------- Rmvt function -------------------- #######################

def Rmvt(
    n,
    mu,
    sigma,
    df
):
    p = len(mu)

    #eigen values/vector of sigma (eigenvalues of a complex Hermitian (conjugate symmetric) or a real symmetric matrix)
    ev = np.linalg.eigh(sigma, UPLO='U')[0]
    evec = np.linalg.eigh(sigma, UPLO='U')[1]
 
    x = mu.reshape(p, 1) + np.matmul(evec * np.repeat(np.sqrt(np.maximum(ev,0)), p).reshape(len(ev),p).transpose(), norm.rvs(size = n*p).reshape(p, n)) / np.repeat(np.sqrt(chi2.rvs(df = df, size = n)/df), p).reshape(p, n)
    
    if n == 1:
        x = x.flatten()
    else:
        x = x.transpose()
        
    return(x)