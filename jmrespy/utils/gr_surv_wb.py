"""
Gradient (for optimization) for survival parameters (jac parameter for minimize fonction of scipy)
"""

####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import pandas as pd
import numpy as np

def Gr_surv_wb(thetas, *args):

    #v_dict : Our dictionnary passed in *args
    v_dict = args[0]

    #Extract variables we need from dict
    surv_keys = v_dict['surv_keys'][0]
    p = v_dict['p']
    parametrization = v_dict['parametrization']
    wgh = v_dict['wgh']
    wk = v_dict['wk']
    w1 = v_dict['w1'][surv_keys]
    log_st = v_dict['log_st']
    p_b_yt = v_dict['p_b_yt']
    id_gk = v_dict['id_gk']
    d = v_dict['d'][surv_keys]
    wint_f_vl = v_dict['wint_f_vl']
    wint_f_sl = v_dict['wint_f_sl']
    wsint_f_vl = v_dict['wsint_f_vl']
    wsint_f_sl = v_dict['wsint_f_sl']
    y_hat_s = v_dict['y_hat_s']
    y_hat_s_deriv = v_dict['y_hat_s_deriv']
    y_hat_time = v_dict['y_hat_time']
    y_hat_time_deriv = v_dict['y_hat_time_deriv']
    scale_wb = v_dict['scale_wb']
    st = v_dict['st'][np.newaxis].T
    log_time = v_dict['log_time']
    pos_gammas = v_dict['pos_gammas'][surv_keys]
    pos_alpha = v_dict['pos_alpha'][surv_keys] if parametrization in ['value', 'both'] else None
    pos_d_alpha = v_dict['pos_d_alpha'][surv_keys] if parametrization in ['slope', 'both'] else None
    pos_sigma_t = v_dict['pos_sigma_t'][surv_keys]

    #Extract thetas
    gammas = thetas[pos_gammas]
    alpha = thetas[pos_alpha] if pos_alpha is not None else None
    d_alpha = thetas[pos_d_alpha] if pos_d_alpha is not None else None
    sigma_t = np.exp(thetas[pos_sigma_t]) if scale_wb is None else scale_wb

    #Linear predictors
    eta_tw = np.matmul(w1, gammas)[np.newaxis].T #Estimated value of exponential part of proportional risks model : h(t) = h(0) * exp(eta_ww)
    if parametrization == 'value':
        eta_s = (wsint_f_vl * alpha) * y_hat_s #I can't even explin exactly this one, but it correspond to estimated eta at stime_lag_0 times but without the proportional risk part of eta
    elif parametrization == 'slope':
        eta_s = (wsint_f_sl * d_alpha) * y_hat_s_deriv
    else:
        eta_s = (wsint_f_vl * alpha) * y_hat_s + (wsint_f_sl * d_alpha) * y_hat_s_deriv    
    exp_eta_tw_p = np.exp(eta_tw) * p #Estimated values of proportional risks model : h(t) = h(0) * exp(eta_tw) with h(0) = p

    #scgammas : Value of S(γ) decomposed in each gamma (S(γ1), S(γ2), ..., S(γp))
    inte = wk * np.exp(np.log(sigma_t) + (sigma_t - 1) * log_st + eta_s)
    ki = exp_eta_tw_p * np.array(pd.DataFrame(inte).groupby(id_gk).sum()) #Integration of h0(s) exp α{x⊤i (s)β + zi⊤(s)bi}
    kii = np.matmul(p_b_yt * ki, wgh) #Integration of ki * p(bi | Ti,δi,yi;θ) dsdbi
    scgammas = -(w1 * (d - kii[np.newaxis].T)).sum(axis=0)

    #scalpha : S(α) on current marker
    rr = []
    #For each alpha
    if parametrization in ['value', 'both']:
        for i in range(wint_f_vl.shape[1]):
            rr.append(-(np.matmul((p_b_yt * (d * wint_f_vl[:,i][np.newaxis].T * y_hat_time - exp_eta_tw_p * np.array(pd.DataFrame(inte * wsint_f_vl[:,i][np.newaxis].T * y_hat_s).groupby(id_gk).sum()))), wgh)).sum())
        scalpha = np.array(rr)
    
    #scalpha_deriv : S(α) on evolution of marker
    rr = []
    #For each alpha
    if parametrization in ['slope', 'both']:
        for i in range(wint_f_sl.shape[1]):
            rr.append(-(np.matmul((p_b_yt * (d * wint_f_sl[:,i][np.newaxis].T * y_hat_time_deriv - exp_eta_tw_p * np.array(pd.DataFrame(inte * wsint_f_sl[:,i][np.newaxis].T * y_hat_s_deriv).groupby(id_gk).sum()))), wgh)).sum())
        scalpha_deriv = np.array(rr)

    #scsigma_t : S(θh0)
    if scale_wb is None: 
        inte2 = st**(sigma_t - 1) * (1 + sigma_t * log_st) * np.exp(eta_s)
        scsigma_t = - sigma_t * np.matmul((p_b_yt * (d * (1/sigma_t + log_time) - exp_eta_tw_p * np.array(pd.DataFrame(wk * inte2).groupby(id_gk).sum()))), wgh).sum()
    
    if parametrization == 'value':
        return(np.concatenate((np.array(scgammas), np.array(scalpha), np.array(scsigma_t)), axis=None))
    elif parametrization == 'slope':
        return(np.concatenate((np.array(scgammas), np.array(scalpha_deriv), np.array(scsigma_t)), axis=None))
    else:
        return(np.concatenate((np.array(scgammas), np.array(scalpha), np.array(scalpha_deriv), np.array(scsigma_t)), axis=None))