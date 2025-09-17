"""
This file contains function Dmvnorn to compute
"""

####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import numpy as np

# Probability
from scipy.stats import norm



def Dmvnorm(x, mu, varcov, log=False):

    #Convert x to matrix format if it's a vector
    if x.ndim == 1:
        x = x[np.newaxis]
    
    p = len(mu)

    if len(mu) == 1:
        #(log) likelihood of x following N(mu = mu, sigma = sd) law for one mu
        lk = norm.pdf(x, loc=np.array(mu), scale=np.sqrt(varcov))
        likelihood = lk if not log else np.log(lk)
    else:
        t1 = len(mu) == len(varcov.flatten()) #Indicates if we have as much mu as elements in varcov matrix
        t2 = (np.array(varcov)[np.triu_indices(varcov.shape[0], k=1)] < np.sqrt(np.finfo(float).eps)).all() #Indicates if we have values in covariance which are lower than our informatic tolerance

        #If we are in one of above cases (t1 or t2)
        if t1 or t2:

            #If we only have the issue t2
            if not t1:
                varcov = np.diag(varcov)
            
            nx = x.shape[1]
            #(log) likelihood of x following N(mu = mu, sigma = sd) law for each mu
            lk = norm.pdf(x, loc=np.repeat(np.array(mu), nx), scale=np.repeat(np.sqrt(varcov), nx))
            likelihood = lk if not log else np.log(lk)

        else:

            #eigen values of varcov (eigenvalues of a complex Hermitian (conjugate symmetric) or a real symmetric matrix)
            ev = np.linalg.eigh(varcov, UPLO='U')[0]
            inv_varcov = np.linalg.inv(varcov) #inverse of varcov matrix
            s = x - np.tile(mu, (x.shape[0],1))  #Deviation from mu each value of x
            quad = 0.5 * (np.matmul(s, inv_varcov) * s).sum(axis=1) #log(exp(-t(bi) %*% D^-1 %*% bi/2))
            fact = - 0.5 * (p * np.log(2*np.pi) + np.log(ev).sum()) #log(2pi^(-ncz/2) * det(D)^(-1/2))
            lk = fact - quad
            likelihood = lk if log else np.exp(lk)
        
    #Returned (log) likelihood
    return(likelihood)