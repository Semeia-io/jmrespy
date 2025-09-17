"""
Gradient (for optimization) for thetas parameters (jac parameter for minimize fonction of scipy) estimated by Quasi-Newton method
"""

####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import pandas as pd
import numpy as np

# Probability
from scipy.stats import norm

# Modules from our scripts
from .chol_trans import Chol_transf
from .dmvnorm import Dmvnorm
from .deriv_d import Deriv_d
from.jacobian2 import Jacobian2

def Score_weibullGH(thetas, *args):

    #v_dict : Our dictionnary passed in *args
    v_dict = args[0]

    ### Extract inputs

    # Keys
    surv_keys = v_dict['surv_keys'][0]
    long_keys = v_dict['long_keys'][0]

    # Control
    parametrization = v_dict['parametrization'][long_keys]
    tol4 = v_dict['tol4']

    # Design longitudinal
    y_lme = v_dict['y_lme'][long_keys]
    x_lme = v_dict['x_lme'][long_keys]
    x_lme_time = v_dict['x_lme_time'][long_keys]
    x_lme_time_deriv = v_dict['x_lme_time_deriv'][long_keys]
    x_lme_s = v_dict['x_lme_s'][long_keys]
    x_lme_s_deriv = v_dict['x_lme_s_deriv'][long_keys]
    z_lme = v_dict['z_lme'][long_keys]
    zt_z_lme = v_dict['zt_z_lme'][long_keys]
    z_lme_b = v_dict['z_lme_b'][long_keys]
    z_lme_time_b = v_dict['z_lme_time_b'][long_keys]
    z_lme_time_b_deriv = v_dict['z_lme_time_b_deriv'][long_keys]
    z_lme_s_b = v_dict['z_lme_s_b'][long_keys]
    z_lme_s_b_deriv = v_dict['z_lme_s_b_deriv'][long_keys]
    ind_fixed = v_dict['ind_fixed'][long_keys]
    ncx = v_dict['ncx_dict'][long_keys]
    ncz = v_dict['ncz_dict'][long_keys]
    n_long = v_dict['n_long'][long_keys]
    wgh = v_dict['wgh']

    # Design survival
    log_time = v_dict['log_time']
    log_st = v_dict['log_st']
    p = v_dict['p']
    d = v_dict['d'][surv_keys]
    w1 = v_dict['w1'][surv_keys]
    wint_f_vl = v_dict['wint_f_vl']
    wint_f_sl = v_dict['wint_f_sl']
    wsint_f_vl = v_dict['wsint_f_vl']
    wsint_f_sl = v_dict['wsint_f_sl']
    wk = v_dict['wk']
    n = v_dict['n']
    scale_wb = v_dict['scale_wb']

    # b
    b = v_dict['b']
    b2 = v_dict['b2']
    diag_d = v_dict['diag_d']

    # id
    id_l = v_dict['id_l'][long_keys]
    id_gk = v_dict['id_gk']

    # Thetas
    pos_betas = v_dict['pos_betas'][long_keys]
    pos_sigma = v_dict['pos_sigma'][long_keys]
    pos_gammas = v_dict['pos_gammas'][surv_keys]
    pos_alpha = v_dict['pos_alpha'][surv_keys][long_keys]
    pos_d_alpha = v_dict['pos_d_alpha'][surv_keys][long_keys]
    pos_sigma_t = v_dict['pos_sigma_t'][surv_keys]
    pos_d_vc = v_dict['pos_d_vc']
    betas = thetas[pos_betas]
    sigma = np.exp(thetas[pos_sigma])
    gammas = thetas[pos_gammas]
    alpha = thetas[pos_alpha] if pos_alpha is not None else None
    d_alpha = thetas[pos_d_alpha] if pos_d_alpha is not None else None
    sigma_t = np.exp(thetas[pos_sigma_t]) if scale_wb is None else scale_wb
    d_vc = np.exp(thetas[pos_d_vc]) if diag_d else Chol_transf(thetas[pos_d_vc])['res']

    #Linear predictors
    eta_yx = np.matmul(x_lme, betas)[np.newaxis].T #Estimated y by fixed effects of lme : eta_yx = beta0 + beta1*x1 + beta2*x2 + ... + betap*xp
    eta_tw = np.matmul(w1, gammas)[np.newaxis].T #Estimated value of exponential part of proportional risks model : h(t) = h(0) * exp(eta_ww)
    if parametrization in ['value', 'both']:
        y_hat_time = np.matmul(x_lme_time, betas)[np.newaxis].T + z_lme_time_b #Estimated y at times to event by lme (fixed effects and random effects)
        y_hat_s = np.matmul(x_lme_s, betas)[np.newaxis].T + z_lme_s_b #Estimated y at all stime_lag_0 (defined in jointModel.py) times to event by lme (fixed effects and random effects)
        wint_f_vl_alph = np.matmul(wint_f_vl, alpha)[np.newaxis].T
        wsint_f_vl_alph = np.matmul(wsint_f_vl, alpha)[np.newaxis].T
        eta_t = eta_tw + wint_f_vl_alph * y_hat_time #Estimated eta (see Rizoupoulos documentation) in our joint model
        eta_s = wsint_f_vl_alph * y_hat_s #I can't even explin exactly this one, but it correspond to estimated eta at stime_lag_0 times but without the proportional risk part of eta
    if parametrization in ['slope', 'both']:
        y_hat_time_deriv = np.matmul(x_lme_time_deriv, betas[ind_fixed])[np.newaxis].T + z_lme_time_b_deriv
        y_hat_s_deriv = np.matmul(x_lme_s_deriv, betas[ind_fixed])[np.newaxis].T + z_lme_s_b_deriv
        wint_f_sl_alph = np.matmul(wint_f_sl, d_alpha.flatten())[np.newaxis].T
        wsint_f_sl_alph = np.matmul(wsint_f_sl, d_alpha.flatten())[np.newaxis].T
        if parametrization == 'both':
            eta_t = eta_t + wint_f_sl_alph * y_hat_time_deriv
            eta_s = eta_s + wsint_f_sl_alph * y_hat_s_deriv
        else:
            eta_t = eta_tw + wint_f_sl_alph * y_hat_time_deriv
            eta_s = wsint_f_sl_alph * y_hat_s_deriv
    exp_eta_tw = np.exp(eta_tw)
    exp_eta_tw_p = exp_eta_tw * p

    #Likelihoods
    mu_y = eta_yx + z_lme_b #Estimated y by lme (fixed effects and random effects)
    log_norm = np.log(norm.pdf(y_lme, loc=np.array(mu_y), scale=sigma)+10e-100) #log likelihood of our y_lme following N(mu = mu_y, sigma = sigma) law
    log_p_y_b = np.array(pd.DataFrame(log_norm).groupby(id_l).sum()) #Sum of log_norm likelihoods by group : log p(yi | bi;θy)
    log_hazard = np.log(sigma_t) + (sigma_t - 1) * log_time + eta_t
    inte = wk * np.exp(np.log(sigma_t) + (sigma_t - 1) * log_st + eta_s)
    log_survival = - exp_eta_tw_p * np.array(pd.DataFrame(inte).groupby(id_gk).sum())
    log_p_tb = d * log_hazard + log_survival #Log likelihood of survival part in joint model : log p(Ti,δi | bi;θt,β)
    log_p_b = np.repeat(Dmvnorm(x=b, mu=np.repeat(0, ncz), varcov=d_vc, log=True), n) #Log likelihood of random effects : log p(bi | Vech(D))
    p_y_tb = np.exp(log_p_y_b + log_p_tb + log_p_b.reshape((log_p_y_b.shape[1], log_p_y_b.shape[0])).T)
    p_yt = np.matmul(p_y_tb, wgh) + tol4 #Integration of p(Ti, δi, yi, bi; θ) dbi
    p_b_yt = p_y_tb / p_yt[np.newaxis].T

    #Expectation for b
    post_b = np.matmul(p_b_yt, wgh[np.newaxis].T * b) #E(bi | Ti,δi,yi;θ)
    outer_post_b = np.array([np.outer(post_b[i], post_b[i]).flatten() for i in range(len(post_b))])
    post_vb = np.matmul(p_b_yt, wgh[np.newaxis].T * b2).flatten() - np.tile(post_b * post_b, 2).flatten() if ncz == 1 else np.matmul(p_b_yt, wgh[np.newaxis].T * b2) - outer_post_b #var(bi | Ti,δi,yi;θ)

    #Score y (Scores for θy : β, σ2)
    z_lme_post_b = post_b[id_l][np.newaxis].T if ncz == 1 else (z_lme * post_b[id_l]).sum(axis=1)[np.newaxis].T #Estimated random effects for each longitudinal marquor : Zibi
    mu = (y_lme - eta_yx).flatten() #Difference between longitudinal marquors and his estimation by fixed effects of lme only : yi − Xiβ
    tr_tz_z_vb = (zt_z_lme * post_vb).sum() #tr(t(Zi) Zivbi)
    sc1 = - np.matmul(x_lme.T, y_lme - eta_yx - z_lme_post_b) / (sigma ** 2) #sc1 : first part of S(β) function
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
    score_y = np.concatenate(((sc1 + np.array(sc2)[np.newaxis].T).flatten(), (-sigma * (-n_long / sigma + (np.matmul(mu, mu[np.newaxis].T - 2 * z_lme_post_b) + np.matmul(z_lme_post_b.T, z_lme_post_b) + tr_tz_z_vb) / (sigma**3))).flatten()), axis=None)

    #scgammas : Value of S(γ) decomposed in each gamma (S(γ1), S(γ2), ..., S(γp))
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
        inte2 = np.exp(log_st)**(sigma_t - 1) * (1 + sigma_t * log_st) * np.exp(eta_s)
        scsigma_t = - sigma_t * np.matmul((p_b_yt * (d * (1/sigma_t + log_time) - exp_eta_tw_p * np.array(pd.DataFrame(wk * inte2).groupby(id_gk).sum()))), wgh).sum()

    #Score t (Scores for θt : γ, α, θh0)
    if parametrization == 'value':
        score_t = np.concatenate((np.array(scgammas), np.array(scalpha), np.array(scsigma_t)), axis=None)
    elif parametrization == 'slope':
        score_t = np.concatenate((np.array(scgammas), np.array(scalpha_deriv), np.array(scsigma_t)), axis=None)
    else:
        score_t = np.concatenate((np.array(scgammas), np.array(scalpha), np.array(scalpha_deriv), np.array(scsigma_t)), axis=None)

    #Score b (Scores for θb)
    if diag_d:
        raise ValueError("Random effects must be on 2 or more variables today, we will soon update package to carry 1 random effect")
    else:
        sv_d = np.linalg.inv(d_vc) #Inverse of d
        d_d = Deriv_d(d_vc)
        #n_d_d = len(d_d)
        d1 = np.array([(sv_d * x).sum() for x in d_d])
        d2 = np.array([(np.matmul(np.matmul(sv_d, x), sv_d)).flatten() for x in d_d])
        cs_post_vb = post_vb.sum(axis=0)
        out = []
        for i in range(len(d_d)):
            d_mat = d2[i,].reshape((ncz, ncz)).T
            out.append((d2[i,] * cs_post_vb).sum() + (np.matmul(post_b, d_mat) * post_b).sum())
        jacob = Jacobian2(Chol_transf(thetas[pos_d_vc])['l'], ncz)
        score_b = np.matmul(0.5 * (n * d1 - out), jacob)

    return(np.concatenate((np.array(score_y), np.array(score_t), np.array(score_b)), axis=None))

