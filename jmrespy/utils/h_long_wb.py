"""

"""

####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import pandas as pd
import numpy as np

def H_long_wb(betas, v_dict):

    #Extract variables we need from dict
    surv_keys = v_dict['surv_keys'][0]
    ind_fixed = v_dict['ind_fixed']
    parametrization = v_dict['parametrization']
    x_lme = v_dict['x_lme']
    x_lme_s = v_dict['x_lme_s']
    x_lme_s_deriv = v_dict['x_lme_s_deriv']
    z_lme_s_b = v_dict['z_lme_s_b']
    z_lme_s_b_deriv = v_dict['z_lme_s_b_deriv']
    xt_x_lme = v_dict['xt_x_lme']
    sigma = v_dict['sigma']
    wsint_f_vl = v_dict['wsint_f_vl']
    wsint_f_sl = v_dict['wsint_f_sl']
    alpha = v_dict['alpha'][surv_keys] if parametrization in ['value', 'both'] else None
    d_alpha = v_dict['d_alpha'][surv_keys] if parametrization in ['slope', 'both'] else None
    eta_tw = v_dict['eta_tw'][surv_keys]
    p = v_dict['p']
    wgh = v_dict['wgh']
    wk = v_dict['wk']
    sigma_t = v_dict['sigma_t'][surv_keys]
    log_st = v_dict['log_st']
    ncx = v_dict['ncx']
    id_gk = v_dict['id_gk']
    p_b_yt = v_dict['p_b_yt']

    if parametrization in ['value', 'both']:
        y_lme_s = np.matmul(x_lme_s, betas)[np.newaxis].T + z_lme_s_b
        wsint_f_vl_alph = np.matmul(wsint_f_vl, alpha.flatten())[np.newaxis].T
        eta_s = wsint_f_vl_alph * y_lme_s
    if parametrization in ['slope', 'both']:
        y_lme_s_deriv = np.matmul(x_lme_s_deriv, betas[ind_fixed])[np.newaxis].T + z_lme_s_b_deriv
        wsint_f_sl_alph = np.matmul(wsint_f_sl, d_alpha.flatten())[np.newaxis].T
        if parametrization == 'both':
            eta_s = eta_s + wsint_f_sl_alph * y_lme_s_deriv
        else:
            eta_s = wsint_f_sl_alph * y_lme_s_deriv


    exp_eta_tw_p = np.exp(eta_tw) * p #Estimated values of proportional risks model : h(t) = h(0) * exp(eta_tw) with h(0) = p
    h1 = xt_x_lme / (sigma**2) #(part of hessian matrix)
    inte = wk * np.exp(np.log(sigma_t) + (sigma_t - 1) * log_st + eta_s)
    h2 = np.zeros((ncx, ncx))

    #Feeding of h2 (part of hessian matrix)
    for i in range(ncx):
        for j in range(ncx):
            if parametrization == 'value':
                xx_lme_s = (wsint_f_vl_alph**2) * x_lme_s[:,i][np.newaxis].T * x_lme_s[:,j][np.newaxis].T
            elif parametrization == 'slope':
                if i in ind_fixed and j in ind_fixed:
                    ii = np.where(i == ind_fixed)[0]
                    jj = np.where(j == ind_fixed)[0]
                    xx_lme_s = (wsint_f_sl_alph**2) * x_lme_s_deriv[:,ii] * x_lme_s_deriv[:,jj]
                else:
                    xx_lme_s = 0
            else:
                if i in ind_fixed and j in ind_fixed:
                    ii = np.where(i == ind_fixed)[0]
                    jj = np.where(j == ind_fixed)[0]
                    xx_lme_s = (wsint_f_vl_alph * x_lme_s[:,i][np.newaxis].T + wsint_f_sl_alph * x_lme_s_deriv[:,ii]) * (wsint_f_vl_alph * x_lme_s[:,j][np.newaxis].T + wsint_f_sl_alph * x_lme_s_deriv[:,jj])
                elif i in ind_fixed and j not in ind_fixed:
                    ii = np.where(i == ind_fixed)[0]
                    xx_lme_s = (wsint_f_vl_alph * x_lme_s[:,i][np.newaxis].T + wsint_f_sl_alph * x_lme_s_deriv[:,ii]) * wsint_f_vl_alph * x_lme_s[:,j][np.newaxis].T
                elif i not in ind_fixed and j in ind_fixed:
                    jj = np.where(j == ind_fixed)[0]
                    xx_lme_s = wsint_f_vl_alph * x_lme_s[:,i][np.newaxis].T * (wsint_f_vl_alph * x_lme_s[:,j][np.newaxis].T + wsint_f_sl_alph * x_lme_s_deriv[:,jj])
                else:
                    xx_lme_s = (wsint_f_vl_alph**2) * x_lme_s[:,i][np.newaxis].T * x_lme_s[:,j][np.newaxis].T
            ki = exp_eta_tw_p * np.array(pd.DataFrame(inte * xx_lme_s).groupby(id_gk).sum()) #Integration of h0(s) exp[α{x⊤i (s)β + zi⊤(s)bi}]
            kii = np.matmul(p_b_yt * ki, wgh) #Integration of ki * p(bi | Ti,δi,yi;θ) dsdbi
            h2[i,j] = kii.sum()
    
    return(h1 + h2) #Hessian matrix


    