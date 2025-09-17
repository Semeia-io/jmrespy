####################### -------------------- Import libraries and modules -------------------- #######################
#Usual libraries
import numpy as np

#Scipy
from scipy.stats import norm

#Display
from tqdm import tqdm

# Modules from our scripts
from .loglik_weibull import LgLik_weibull
from .loglik_weibull import LgLik_weibull_grad




####################### -------------------- Noise_level -------------------- #######################

def Noise_level(x, optim_dict, n_rep, prob):
    """
        Function which computes noise level of log-likelihood function and noise level of its gradient by asymptotic inference.
        log-likelihood noise and norm2 of log-likelihood gradient noise are assumend gaussian
    Args:
        x (np.array) : Vector of theta parameters
        optim_dict (dict) : Dictionnary passed as *args in LgLik_weibull and LgLik_weibull_grad
        n_rep (int) : Number of computation of log-likelihood and its gradient to estimates noise levels
        prob (float) : Cumulative density function value associated to quantile (of N(0,1) distribution) used to determine noise level
    Returns:
        eps_g (np.float) : Noise level of log-likelihood gradient norm2
        eps_f (np.float) : Noise level of log-likelihood function
    """

    value_lglik = []
    value_lglik_grad = []

    #Percentile of Normal N(0,1) distribution
    percentile = norm.ppf(prob)
    
    print('Estim noise')

    #Computation of log likelihoods and gradients
    for i in tqdm(range(n_rep)):
        log_lik = - LgLik_weibull(x, optim_dict)
        value_lglik.append(log_lik)
        lglik_grad = LgLik_weibull_grad(x, optim_dict)
        value_lglik_grad.append(lglik_grad)

    eps_g = percentile * np.sqrt(np.linalg.norm(np.array(value_lglik_grad),axis=1).var(ddof=1))
    eps_f = percentile * np.sqrt(np.array(value_lglik).var(ddof=1))

    return eps_g, eps_f