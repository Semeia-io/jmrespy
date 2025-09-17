"""
Function to minimize (for optimization) for survival parameters
"""

####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import pandas as pd
import numpy as np

def Opt_surv_wb(thetas, *args):    

    #v_dict : Our dictionnary passed in *args
    v_dict = args[0]

    #Extract variables we need from dict
    surv_keys = v_dict['surv_keys'][0]
    parametrization = v_dict['parametrization']
    wgh = v_dict['wgh']
    p_b_yt = v_dict['p_b_yt']
    log_p_tb = v_dict['log_p_tb']
    pos_gammas = v_dict['pos_gammas'][surv_keys]
    pos_alpha = v_dict['pos_alpha'][surv_keys] if parametrization in ['value', 'both'] else None
    pos_d_alpha = v_dict['pos_d_alpha'][surv_keys] if parametrization in ['slope', 'both'] else None
    pos_sigma_t = v_dict['pos_sigma_t'][surv_keys]
    scale_wb = v_dict['scale_wb']
    w1 = v_dict['w1'][surv_keys]
    wk = v_dict['wk']
    id_gk = v_dict['id_gk']
    wint_f_vl = v_dict['wint_f_vl']
    wint_f_sl = v_dict['wint_f_sl']
    wsint_f_vl = v_dict['wsint_f_vl']
    wsint_f_sl = v_dict['wsint_f_sl']
    y_hat_time = v_dict['y_hat_time']
    y_hat_time_deriv = v_dict['y_hat_time_deriv']
    y_hat_s = v_dict['y_hat_s']
    y_hat_s_deriv = v_dict['y_hat_s_deriv']
    log_time = v_dict['log_time']
    log_st = v_dict['log_st']
    d = v_dict['d'][surv_keys]
    p = v_dict['p']

    #Extract thetas
    gammas = thetas[pos_gammas]
    alpha = thetas[pos_alpha] if pos_alpha is not None else None
    d_alpha = thetas[pos_d_alpha] if pos_d_alpha is not None else None
    sigma_t = np.exp(thetas[pos_sigma_t]) if scale_wb is None else scale_wb

    #Linear predictors
    eta_tw = np.matmul(w1, gammas)[np.newaxis].T #Estimated value of exponential part of proportional risks model : h(t) = h(0) * exp(eta_ww)
    if parametrization == 'value':
        eta_t = eta_tw + (wint_f_vl * alpha) * y_hat_time #Estimated eta (see Rizoupoulos documentation) in our joint model
        eta_s = (wsint_f_vl * alpha) * y_hat_s #I can't even explin exactly this one, but it correspond to estimated eta at stime_lag_0 times but without the proportional risk part of eta
    elif parametrization == 'slope':
        eta_t = eta_tw + (wint_f_sl * d_alpha) * y_hat_time_deriv
        eta_s = (wsint_f_sl * d_alpha) * y_hat_s_deriv
    else:
        eta_t = eta_tw + (wint_f_vl * alpha) * y_hat_time + (wint_f_sl * d_alpha) * y_hat_time_deriv
        eta_s = (wsint_f_vl * alpha) * y_hat_s + (wsint_f_sl * d_alpha) * y_hat_s_deriv

    #Parts of logliklihoods
    log_hazard = np.log(sigma_t) + (sigma_t - 1) * log_time + eta_t
    log_survival = -np.exp(eta_tw) * p * np.array(pd.DataFrame(wk * np.exp(np.log(sigma_t) + (sigma_t - 1) * log_st + eta_s)).groupby(id_gk).sum())
    log_p_tb = d * log_hazard + log_survival #Log likelihood of survival part in joint model : log p(Ti,δi | bi;θt,β)
    p_b_ytn = p_b_yt * log_p_tb

    #logliklihood
    log_lk = -np.matmul(p_b_ytn, wgh).sum()

    return(log_lk)