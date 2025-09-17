"""
Called in Survfit(), 
"""
####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import numpy as np

# Probability
from scipy.special import gammaln





####################### -------------------- Dmvt function -------------------- #######################

def Dmvt(
    x,
    mu,
    sigma,
    df,
    log = False
):
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("'x' must be a numeric array")

    p = len(mu)

    #Convert x to matrix format if it's a vector
    if x.ndim == 1:
        x = x[np.newaxis]

    if sigma.shape != (p, p) or x.shape[1] != p:
        raise ValueError("incompatible arguments")

    #eigen values of sigma (eigenvalues of a complex Hermitian (conjugate symmetric) or a real symmetric matrix)
    ev = np.linalg.eigh(sigma, UPLO='U')[0]

    #Deviation from mu each value of x
    ss = x - np.tile(mu, (x.shape[0],1))

    #inverse of sigma matrix
    inv_sigma = np.linalg.inv(sigma)

    quad = (np.matmul(ss, inv_sigma) * ss).sum(axis=1) / df

    fact = gammaln((df + p)/2) - gammaln(df/2) - 0.5 * (p * (np.log(np.pi) + np.log(df)) + np.log(ev).sum())

    if log:
        return(fact - 0.5 * (df + p) * np.log(1 + quad))
    else:
        return(np.exp(fact) * ((1 + quad)**(- (df + p)/2)))