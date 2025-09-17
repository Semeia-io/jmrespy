"""
Function to minimize (for optimization) for survival and longitudinal parameters in EM algorithm
"""

####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import pandas as pd
import numpy as np

# Probability
from scipy.stats import norm

def Opt_long_surv_wb(thetas, *args):    

    #v_dict : Our dictionnary passed in *args
    v_dict = args[0]

    #Extract variables we need from dict
    ind_fixed = v_dict['ind_fixed']
    wgh = v_dict['wgh']
    parametrization = v_dict['parametrization']
    x_lme = v_dict['x_lme']
    x_lme_time = v_dict['x_lme_time']
    x_lme_time_deriv = v_dict['x_lme_time_deriv']
    x_lme_s = v_dict['x_lme_s']
    x_lme_s_deriv = v_dict['x_lme_s_deriv']
    y_lme = v_dict['y_lme']
    z_lme_b = v_dict['z_lme_b']
    z_lme_time_b = v_dict['z_lme_time_b']
    z_lme_time_b_deriv = v_dict['z_lme_time_b_deriv']
    z_lme_s_b = v_dict['z_lme_s_b']
    z_lme_s_b_deriv = v_dict['z_lme_s_b_deriv']
    p_b_yt = v_dict['p_b_yt']
    log_p_tb = v_dict['log_p_tb']
    scale_wb = v_dict['scale_wb']
    w1 = v_dict['w1']
    wk = v_dict['wk']
    id_l = v_dict['id_l']
    id_gk = v_dict['id_gk']
    wint_f_vl = v_dict['wint_f_vl']
    wint_f_sl = v_dict['wint_f_sl']
    wsint_f_vl = v_dict['wsint_f_vl']
    wsint_f_sl = v_dict['wsint_f_sl']
    log_time = v_dict['log_time']
    log_st = v_dict['log_st']
    d = v_dict['d']
    p = v_dict['p']
    scale_wb = v_dict['scale_wb']
    pos_betas = v_dict['pos_betas']
    pos_gammas = v_dict['pos_gammas']
    pos_alpha = v_dict['pos_alpha']
    pos_d_alpha = v_dict['pos_d_alpha']
    pos_sigma_t = v_dict['pos_sigma_t']
    surv_keys = v_dict['surv_keys']
    sigma = v_dict['sigma']

    #Extract thetas
    betas = thetas[pos_betas]
    gammas = {risk:thetas[pos_gammas[risk]] for risk in surv_keys}
    alpha = {risk:thetas[pos_alpha[risk]] for risk in surv_keys} if pos_alpha is not None else None
    d_alpha = {risk:thetas[pos_d_alpha[risk]] for risk in surv_keys} if pos_d_alpha is not None else None
    sigma_t = {risk:np.exp(thetas[pos_sigma_t[risk]]) for risk in surv_keys} if scale_wb is None else scale_wb

    #Linear predictors
    eta_yx = np.matmul(x_lme, betas)[np.newaxis].T #Estimated y by fixed effects of lme : eta_yx = beta0 + beta1*x1 + beta2*x2 + ... + betap*xp
    eta_tw = {risk : np.matmul(w1[risk], gammas[risk])[np.newaxis].T for risk in surv_keys} #Estimated value of exponential part of proportional risks model : h(t) = h(0) * exp(eta_ww) without link with longitundinal markor
    if parametrization in ['value', 'both']:
        y_hat_time = np.matmul(x_lme_time, betas)[np.newaxis].T + z_lme_time_b #Estimated y at times to event by lme (fixed effects and random effects)
        y_hat_s = np.matmul(x_lme_s, betas)[np.newaxis].T + z_lme_s_b #Estimated y at all stime_lag_0 (defined in jointModel.py) times to event by lme (fixed effects and random effects)
        eta_t = {risk : eta_tw[risk] + wint_f_vl * alpha[risk] * y_hat_time for risk in surv_keys} #Estimated eta with link with longitudinal markor if mi(t) only is the estimated value of longitudinal markor
        eta_s = {risk : wsint_f_vl * alpha[risk] * y_hat_s for risk in surv_keys} #Link beetween logitudinal markor's quadrature nodes (for the integral) and survival in eta
    if parametrization in ['slope', 'both']:
        y_hat_time_deriv = np.matmul(x_lme_time_deriv, betas[ind_fixed])[np.newaxis].T + z_lme_time_b_deriv
        y_hat_s_deriv = np.matmul(x_lme_s_deriv, betas[ind_fixed])[np.newaxis].T + z_lme_s_b_deriv
        if parametrization == 'both':
            eta_t = {risk : eta_t[risk] + wint_f_sl * d_alpha[risk] * y_hat_time_deriv for risk in surv_keys} #Estimated eta with link with longitudinal markor if mi(t) only is on the estimated derivation and current estimated value of longitudinal markor
            eta_s = {risk : eta_s[risk] + wsint_f_sl * d_alpha[risk] * y_hat_s_deriv for risk in surv_keys} #Link beetween logitudinal markor's quadrature nodes (for the integral) and survival in eta
        else:
            eta_t = {risk : eta_tw[risk] + wint_f_sl * d_alpha[risk] * y_hat_time_deriv for risk in surv_keys} #Estimated eta with link with longitudinal markor if mi(t) only is the estimated derivation of longitudinal markor
            eta_s = {risk : wsint_f_sl * d_alpha[risk] * y_hat_s_deriv for risk in surv_keys} #Link beetween logitudinal markor's quadrature nodes (for the integral) and survival in eta

    #Parts of logliklihoods (without log p(bi; θb))
    mu_y = eta_yx + z_lme_b #Estimated y by lme (fixed effects and random effects)
    log_norm = np.log(norm.pdf(y_lme, loc=np.array(mu_y), scale=sigma)+10e-100) #log likelihood of our y_lme following N(mu = mu_y, sigma = sigma) law
    log_p_y_b = np.array(pd.DataFrame(log_norm).groupby(id_l).sum()) #Sum of log_norm likelihoods by group : log p(yi | bi;θy)
    log_hazard = np.array([(np.log(sigma_t[risk]) + (sigma_t[risk] - 1) * log_time + eta_t[risk]) * d[risk] for risk in surv_keys]).sum(axis=0) #Instantaneous risk * d
    log_survival = np.array([-np.exp(eta_tw[risk]) * p * np.array(pd.DataFrame(wk * np.exp(np.log(sigma_t[risk]) + (sigma_t[risk] - 1) * log_st + eta_s[risk])).groupby(id_gk).sum()) for risk in surv_keys]).sum(axis=0) #Survival function
    log_p_tb = log_hazard + log_survival #Log likelihood of survival part in joint model : log p(Ti,δi | bi;θt,β)
    p_b_ytn = p_b_yt * (log_p_tb + log_p_y_b) #(log p(Ti,δi | bi;θt,β) + log p(yi | bi;θy))*p(bi | Ti, δi, yi; θ(it))

    #logliklihood
    log_lk = -np.matmul(p_b_ytn, wgh).sum()

    return(log_lk)