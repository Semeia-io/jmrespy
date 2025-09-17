"""
Called in Survfit(), 
"""
####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import numpy as np
import pandas as pd

# Modules from our scripts
from .dmvnorm import Dmvnorm



####################### -------------------- S_b function -------------------- #######################
def S_b(
    jm_obj,
    t,
    b,
    i,
    mats,
    betas,
    gammas,
    alpha_value,
    alpha_slope,
    sigma_t,
    w,
    surv_keys
):

    if t == 0:
        return(1)

    st = mats['st']
    wk = mats['wk']
    p = mats['p']
    x_lme_s = mats['x_lme_s']
    z_lme_s = mats['z_lme_s']
    x_lme_s_deriv = mats['x_lme_s_deriv']
    z_lme_s_deriv = mats['z_lme_s_deriv']
    wsint_f_vl = mats['wsint_f_vl']
    wsint_f_sl = mats['wsint_f_sl']
    parametrization = jm_obj.parametrization
    ind_fixed = jm_obj.derivForm['ind_fixed']
    ind_random = jm_obj.derivForm['ind_random']
    method = jm_obj.method

    #Estimated y for quadrature
    if parametrization in ['value', 'both']:
        y_lme_s = np.matmul(x_lme_s, betas)[np.newaxis].T + (z_lme_s * b).sum(axis=1)[np.newaxis].T

    if parametrization in ['slope', 'both']:
        y_lme_s_deriv = np.matmul(x_lme_s_deriv, betas[ind_fixed])[np.newaxis].T + (z_lme_s_deriv * b[ind_random]).sum(axis=1)[np.newaxis].T

    #Longitudinal y (estimated for quadrature) part of eta (see Rizoupoulos documentation)
    if parametrization == 'value':
        tt = {risk : wsint_f_vl * alpha_value[risk] * y_lme_s for risk in surv_keys}
    elif parametrization == 'slope':
        tt = {risk : wsint_f_sl * alpha_slope[risk] * y_lme_s_deriv for risk in surv_keys} 
    else:
        tt = {risk : wsint_f_vl * alpha_value[risk] * y_lme_s + wsint_f_sl * alpha_slope[risk] * y_lme_s_deriv for risk in surv_keys}
    
    #Survival part of eta
    eta_tw = {risk : np.matmul(w[risk][i,], gammas[risk])[np.newaxis].T for risk in surv_keys}

    if method == 'weibull-PH-GH':
        log_survival = np.array([-np.exp(eta_tw[risk]) * p * (wk * np.exp(np.log(sigma_t[risk]) + (sigma_t[risk] - 1) * np.log(st+1e-99) + tt[risk].flatten())).sum() for risk in surv_keys]).sum()

    return(np.exp(log_survival))