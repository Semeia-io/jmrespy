"""

"""

####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import pandas as pd
import numpy as np

def Gr_long_wb(betas, v_dict):

    #Extract variables we need from dict
    surv_keys = v_dict['surv_keys'][0]
    parametrization = v_dict['parametrization']
    ind_fixed = v_dict['ind_fixed']
    x_lme = v_dict['x_lme']
    y_lme = v_dict['y_lme']
    x_lme_s = v_dict['x_lme_s']
    x_lme_s_deriv = v_dict['x_lme_s_deriv']
    x_lme_time = v_dict['x_lme_time']
    x_lme_time_deriv = v_dict['x_lme_time_deriv']
    z_lme_s_b = v_dict['z_lme_s_b']
    z_lme_s_b_deriv = v_dict['z_lme_s_b_deriv']
    sigma = v_dict['sigma']
    wint_f_vl = v_dict['wint_f_vl']
    wint_f_sl = v_dict['wint_f_sl']
    wsint_f_vl = v_dict['wsint_f_vl']
    wsint_f_sl = v_dict['wsint_f_sl']
    alpha = v_dict['alpha'][surv_keys] if parametrization in ['value', 'both'] else None
    d_alpha = v_dict['d_alpha'][surv_keys] if parametrization in ['slope', 'both'] else None
    eta_tw = v_dict['eta_tw'][surv_keys]
    eta_s = v_dict['eta_s'][surv_keys]
    p = v_dict['p']
    wk = v_dict['wk']
    wgh = v_dict['wgh']
    sigma_t = v_dict['sigma_t'][surv_keys]
    log_st = v_dict['log_st']
    ncx = v_dict['ncx']
    id_gk = v_dict['id_gk']
    p_b_yt = v_dict['p_b_yt']
    z_lme_post_b = v_dict['z_lme_post_b']
    d = v_dict['d'][surv_keys]

    eta_yx = np.matmul(x_lme, betas) #Xiβ
    if parametrization in ['value', 'both']:
        y_lme_s = np.matmul(x_lme_s, betas)[np.newaxis].T + z_lme_s_b
        wint_f_vl_alph = np.matmul(wint_f_vl, alpha.flatten())[np.newaxis].T
        wsint_f_vl_alph = np.matmul(wsint_f_vl, alpha.flatten())[np.newaxis].T
        eta_s = wsint_f_vl_alph * y_lme_s
    if parametrization in ['slope', 'both']:
        y_lme_s_deriv = np.matmul(x_lme_s_deriv, betas[ind_fixed])[np.newaxis].T + z_lme_s_b_deriv
        wint_f_sl_alph = np.matmul(wint_f_sl, d_alpha.flatten())[np.newaxis].T
        wsint_f_sl_alph = np.matmul(wsint_f_sl, d_alpha.flatten())[np.newaxis].T
        if parametrization == 'both':
            eta_s = eta_s + wsint_f_sl_alph * y_lme_s_deriv
        else:
            eta_s = wsint_f_sl_alph * y_lme_s_deriv

    exp_eta_tw_p = np.exp(eta_tw) * p #Estimated values of proportional risks model : h(t) = h(0) * exp(eta_tw) with h(0) = p

    sc1 = - np.matmul(x_lme.T, y_lme - eta_yx[np.newaxis].T - z_lme_post_b[np.newaxis].T) / (sigma**2) #sc1 : first part of S(β) function
    inte = wk * np.exp(np.log(sigma_t) + (sigma_t - 1) * log_st + eta_s)
    sc2 = [] #sc2 : second part of S(β) function
    
    #Feeding of sc2 (part of hessian matrix)
    for i in range(ncx):
        if parametrization == 'value':
            xx_lme_s = wsint_f_vl_alph * x_lme_s[:,i][np.newaxis].T
            xx_lme = wint_f_vl_alph * x_lme_time[:,i][np.newaxis].T
            ki = exp_eta_tw_p * np.array(pd.DataFrame(inte * xx_lme_s).groupby(id_gk).sum()) #Integration of h0(s)αxi(s) exp α{x⊤i (s)β + zi⊤(s)bi}]
            kii = np.matmul(p_b_yt * ki, wgh) #Integration of ki * p(bi | Ti,δi,yi;θ) dsdbi
            val = (d * xx_lme - kii[np.newaxis].T).sum() #Value of suvival part of S(β) (beginning with αδixi(Ti))
        elif parametrization == 'slope':
            ii = np.where(i == ind_fixed)[0]
            if len(ii) > 0:
                xx_lme_s_deriv = wsint_f_sl_alph * x_lme_s_deriv[:,ii]
                ki = exp_eta_tw_p * np.array(pd.DataFrame(inte * xx_lme_s_deriv).groupby(id_gk).sum())
                kii = np.matmul(p_b_yt * ki, wgh)
                val = (d * wint_f_sl_alph * x_lme_time_deriv[:,ii] - kii[np.newaxis].T).sum()
            else:
                val = 0
        else:
            xx_lme_s = wsint_f_vl_alph * x_lme_s[:,i][np.newaxis].T
            xx_lme = wint_f_vl_alph * x_lme_time[:,i][np.newaxis].T
            ii = np.where(i == ind_fixed)[0]
            if len(ii) > 0:
                xx_lme_s_deriv = wsint_f_sl_alph * x_lme_s_deriv[:,ii]
                xx_lme_deriv = wint_f_sl_alph * x_lme_time_deriv[:,ii]
            else:
                xx_lme_s_deriv = 0
                xx_lme_deriv = 0
            ki = exp_eta_tw_p * np.array(pd.DataFrame(inte * (xx_lme_s + xx_lme_s_deriv)).groupby(id_gk).sum())
            kii = np.matmul(p_b_yt * ki, wgh)
            val = (d * (xx_lme + xx_lme_deriv) - kii[np.newaxis].T).sum()
        sc2.append(- val)
    return(sc1 + np.array(sc2)[np.newaxis].T)

