"""
based on function nearcor() submitted to R-help by Jens Oehlschlagel on 2007-07-13, and function posdefify() from R package `sfsmisc'
"""

####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import numpy as np

def Near_pd(
    m,
    eig_tol = 1e-06,
    conv_tol = 1e-07,
    posd_tol = 1e-08,
    maxits = 100
):
    n = m.shape[1]
    u = np.zeros((n, n))
    x = m
    iterr = 0
    converged = False

    while iterr < maxits and not converged:
        y = x
        t = y - u
        e = np.linalg.eigh(y, UPLO='U') #eigen values and vectors of y (eigenvalues of a complex Hermitian (conjugate symmetric) or a real symmetric matrix)
        q = e[1] #eigen vectors
        d = e[0] #eigen values
        d2 = np.diag(np.array(d)) if len(d) > 1 else np.array(d)
        p = d > eig_tol * d.flatten()[0]
        qq = q[:,p] #Keep only eihen vectors with eigen values > eig_tol * d.flatten()[0]
        x = np.matmul(np.matmul(qq, d2[:,p][p,:]), qq.T)
        u = x - t
        x = (x + x.T) / 2

        #Check convergence
        conv = np.abs(x - y).sum(axis=1).max() / np.abs(y).sum(axis=1).max()
        iterr = iterr + 1
        converged = conv <= conv_tol
        
    x = (x + x.T) / 2
    e = np.linalg.eigh(x, UPLO='U') #eigen values and vectors of y (eigenvalues of a complex Hermitian (conjugate symmetric) or a real symmetric matrix)
    d = e[0] #eigen values
    eps = posd_tol * abs(d.flatten()[0])
    

    if d[n-1] < eps:
        d[d<eps] = eps
        q = e[1] #eigen vectors
        o_diag = np.diag(x)
        x = np.matmul(q, (d * q.T))        
        d2 = np.sqrt(np.array([max(eps, i) for i in o_diag]) / np.diag(x))
        x = d2[np.newaxis].T * x * np.tile(d2,(n,1))
    
    return((x + x.T) / 2)