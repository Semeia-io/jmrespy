"""
Called in Survfit(), 
"""
####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import numpy as np

# Probability
from scipy.stats import norm, multivariate_normal

# Computation
from scipy.special import expit
from scipy.integrate import quad_vec

# Modules from our scripts
from .dmvnorm import Dmvnorm
from .survivals import Integrand_CR




####################### -------------------- Log_posterior_b function -------------------- #######################

def Log_posterior_b(b, *args):

    #v_dict : Our dictionnary passed in *args
    v_dict = args[0]

    #Extract variables we need from dict
    id_l = v_dict['id_l']
    i = v_dict['i']
    y_lme = v_dict['y_lme']
    x_lme = v_dict['x_lme']
    betas = v_dict['betas']
    gammas = v_dict['gammas']
    sigma = v_dict['sigma']
    d_vc = v_dict['d_vc']
    w = v_dict['w']
    parametrization = v_dict['parametrization']
    alpha_value = v_dict['alpha_value']
    alpha_slope = v_dict['alpha_slope']
    method = v_dict['method']
    sigma_t = v_dict['sigma_t']
    surv_keys = v_dict['surv_keys']
    long_keys = v_dict['long_keys']
    ncz_dict = v_dict['ncz_dict']
    family_long = v_dict['family_long']
    last_time = v_dict['last_time'][i]
    ind_fixed = 1

    #x and y design matrices of group i
    x_lme_i = {key: x_lme[key][id_l[key] == i] for key in long_keys}
    y_lme_i = {key: y_lme[key][id_l[key] == i] for key in long_keys}

    #Adding random effects to betas parameters
    begin_idx = 0
    betas_b = {}
    for key in long_keys:
        ncz_key = ncz_dict[key]
        end_idx = begin_idx+ncz_key
        betas_b[key] = betas[key] + b[begin_idx:end_idx]
        begin_idx = end_idx

    #Estimated y by lme (fixed effects and random effects)
    mu_y = {key: np.matmul(x_lme_i[key], betas_b[key]) if family_long[key] == 'gaussian' else expit(np.matmul(x_lme_i[key], betas_b[key])) for key in long_keys}
        
    #log likelihood of observed longitudinal response given b
    log_p_yb = np.array([norm.logpdf(y_lme_i[key], loc=mu_y[key], scale=sigma[key]).sum() if family_long[key] == 'gaussian' else np.log((mu_y[key]**y_lme_i[key]) * ((1-mu_y[key])**(1-y_lme_i[key]))).sum() for key in long_keys]).sum()

    #Log likelihood of random effects : log p(bi | D)
    log_p_b = multivariate_normal.logpdf(x=b, mean=np.zeros(b.shape), cov=d_vc, allow_singular=True)
    
    #Survival part of eta
    eta_tw = {risk : np.matmul(w[risk][i,], gammas[risk]) for risk in surv_keys}
    if last_time == 0:
        log_survival = 0
    else:
        if method in ['weibull-PH-GH', 'weibull-PH-QMC']:
            g_args = (betas_b, parametrization, alpha_value, alpha_slope, ind_fixed, family_long)
            log_survival = np.array([-np.exp(eta_tw[risk]) * sigma_t[risk] * quad_vec(Integrand_CR,0,last_time,quadrature='gk15',norm='2',limit=1,args=(sigma_t, risk) + g_args)[0] for risk in surv_keys]).sum()

    return(-(log_p_yb + log_survival + log_p_b))