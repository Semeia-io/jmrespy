"""
EM algorithm
"""

####################### -------------------- Import libraries and modules -------------------- #######################

#Usual libraries
import numpy as np
import pandas as pd

# Probability
from scipy.stats import norm
from scipy.stats import multivariate_normal

# Optimisation
from scipy.optimize import minimize

# Modules from our scripts
from .toolbox import *
from .nearpd import Near_pd
from .h_long_wb import H_long_wb
from .gr_long_wb import Gr_long_wb
from .opt_surv_wb import Opt_surv_wb
from .gr_surv_wb import Gr_surv_wb
from .opt_long_surv_wb import Opt_long_surv_wb




####################### -------------------- EM algorithm for JM -------------------- #######################
def EM_optim(v_dict):

    ### Extract inputs

    # Keys
    surv_keys = v_dict['surv_keys']
    long_keys = v_dict['long_keys'][0] #This EM algorithm is only availaible for 1 longitudinal markor

    # Thetas
    betas = v_dict['betas'][long_keys]
    sigma = v_dict['sigma'][long_keys]
    d_vc = v_dict['d_vc']
    gammas = v_dict['gammas']
    alpha = {risk: v_dict['alpha'][risk][long_keys] for risk in surv_keys}
    d_alpha = {risk: v_dict['d_alpha'][risk][long_keys] for risk in surv_keys}
    sigma_t = v_dict['sigma_t']

    # Control
    solver_minimize_em = v_dict['solver_minimize_em']
    options_minimize_em = v_dict['options_minimize_em']
    jac_minimize_em = v_dict['jac_minimize_em']
    parametrization = v_dict['parametrization'][long_keys]
    competing_risks = v_dict['competing_risks']
    tol1 = v_dict['tol1']
    tol2 = v_dict['tol2']
    tol3 = v_dict['tol3']
    tol4 = v_dict['tol4']

    # Design longitudinal
    y_lme = v_dict['y_lme'][long_keys]
    x_lme = v_dict['x_lme'][long_keys]
    x_lme_time = v_dict['x_lme_time'][long_keys]
    xt_x_lme = v_dict['xt_x_lme'][long_keys]
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
    st = v_dict['st']
    log_st = v_dict['log_st']
    p = v_dict['p']
    d = v_dict['d']
    w1 = v_dict['w1']
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

    # id
    id_l = v_dict['id_l'][long_keys]
    id_gk = v_dict['id_gk']

    # Iteration parameters and matrixes used to follow estimations along EM iterations
    iter_em = v_dict['iter_em'] #Nb max of iterations for EM
    y_mat = [] #List (transformed in matrix at the end)  which will contain some parameters related to lme part in joint model : betas and sigma (one line = one iteration)
    #t_mat = [] #List (transformed in matrix at the end)  which will contain some parameters related to time to event part in joint model : gammas, alpha and sigma_t (one line = one iteration)
    t_mat = {risk : [] for risk in surv_keys} # Dictionnary of lists which will contain coefficients related to time to event part in joint model : gammas, alpha and sigma_t (one line = one iteration)
    b_mat = [] #List (transformed in matrix at the end)  which will contain estimated variance-covariance matrix of random effects (one line = one iteration)
    log_lik = [] #List which will contain logliklihood of our fitted model in each iteration
    conv = False #Will indicated if our algorithm converged or not

    #Loop of our algorithm
    for it in range(iter_em):
        print(it)

        #Adding current parameters in lists
        y_mat.append(np.concatenate((betas, sigma), axis=None))
        b_mat.append(d_vc)
        if parametrization == 'value':
            for risk in surv_keys:
                t_mat[risk].append(np.concatenate((gammas[risk], alpha[risk], sigma_t[risk]), axis=None))
        elif parametrization == 'slope':
            for risk in surv_keys:
                t_mat[risk].append(np.concatenate((gammas[risk], d_alpha[risk], sigma_t[risk]), axis=None))
        else:
            for risk in surv_keys:
                t_mat[risk].append(np.concatenate((gammas[risk], alpha[risk], d_alpha[risk], sigma_t[risk]), axis=None))
        
        #Linear predictors
        eta_yx = np.matmul(x_lme, betas)[np.newaxis].T #Estimated y by fixed effects of lme : eta_yx = beta0 + beta1*x1 + beta2*x2 + ... + betap*xp
        eta_tw = {risk : np.matmul(w1[risk], gammas[risk])[np.newaxis].T for risk in surv_keys} #Estimated value of exponential part of proportional risks model : h(t) = h(0) * exp(eta_ww) without link with longitundinal markor
        if parametrization in ['value', 'both']:
            y_hat_time = np.matmul(x_lme_time, betas)[np.newaxis].T + z_lme_time_b #Estimated y at times to event by lme (fixed effects and random effects)
            y_hat_s = np.matmul(x_lme_s, betas)[np.newaxis].T + z_lme_s_b #Estimated y at all quadrature nodes of times to event by lme (fixed effects and random effects)
            eta_t = {risk : eta_tw[risk] + wint_f_vl * alpha[risk] * y_hat_time for risk in surv_keys} #Estimated eta with link with longitudinal markor if mi(t) only is the estimated value of longitudinal markor
            eta_s = {risk : wsint_f_vl * alpha[risk] * y_hat_s for risk in surv_keys} #Link beetween logitudinal markor's quadrature nodes (for the integral) and survival in eta
        if parametrization in ['slope', 'both']:
            y_hat_time_deriv = np.matmul(x_lme_time_deriv, betas[ind_fixed])[np.newaxis].T + z_lme_time_b_deriv #Estimated y at times to event by lme (fixed effects and random effects)
            y_hat_s_deriv = np.matmul(x_lme_s_deriv, betas[ind_fixed])[np.newaxis].T + z_lme_s_b_deriv #Estimated y at all stime_lag_0 (defined in jointModel.py) times to event by lme (fixed effects and random effects)
            if parametrization == 'both':
                eta_t = {risk : eta_t[risk] + wint_f_sl * d_alpha[risk] * y_hat_time_deriv for risk in surv_keys} #Estimated eta with link with longitudinal markor if mi(t) only is on the estimated derivation and current estimated value of longitudinal markor
                eta_s = {risk : eta_s[risk] + wsint_f_sl * d_alpha[risk] * y_hat_s_deriv for risk in surv_keys} #Link beetween logitudinal markor's quadrature nodes (for the integral) and survival in eta
            else:
                eta_t = {risk : eta_tw[risk] + wint_f_sl * d_alpha[risk] * y_hat_time_deriv for risk in surv_keys} #Estimated eta with link with longitudinal markor if mi(t) only is the estimated derivation of longitudinal markor
                eta_s = {risk : wsint_f_sl * d_alpha[risk] * y_hat_s_deriv for risk in surv_keys} #Link beetween logitudinal markor's quadrature nodes (for the integral) and survival in eta
        
        #E-step : likelihoods
        mu_y = eta_yx + z_lme_b #Estimated y by lme (fixed effects and random effects)
        log_norm = np.log(norm.pdf(y_lme, loc=mu_y, scale=sigma)+10e-100) #log likelihood of our y_lme following N(mu = mu_y, sigma = sigma) law, we added 10e-100 to avoid 
        log_p_y_b = np.array(pd.DataFrame(log_norm).groupby(id_l).sum()) #Sum of log_norm likelihoods by group : log p(yi | bi;θy)
        log_hazard = np.array([(np.log(sigma_t[risk]) + (sigma_t[risk] - 1) * log_time + eta_t[risk]) * d[risk] for risk in surv_keys]).sum(axis=0) #Instantaneous risk * d
        log_survival = np.array([-np.exp(eta_tw[risk]) * p * np.array(pd.DataFrame(wk * np.exp(np.log(sigma_t[risk]) + (sigma_t[risk] - 1) * log_st + eta_s[risk])).groupby(id_gk).sum()) for risk in surv_keys]).sum(axis=0) #Survival function
        log_p_tb = log_hazard + log_survival #Log likelihood of survival part in joint model : log p(Ti,δi | bi;θt,β)
        log_p_b = np.repeat(np.log(multivariate_normal.pdf(x=b, mean=np.repeat(0, ncz), cov=d_vc) + 10e-100), n) #Log likelihood of random effects : log p(bi | Vech(D))
        #log_p_b = np.repeat(Dmvnorm(x=b, mu=np.repeat(0, ncz), varcov=d_vc, log=True), n) #Log likelihood of random effects : log p(bi | Vech(D))
        p_y_tb = np.exp(log_p_y_b + log_p_tb + log_p_b.reshape((log_p_y_b.shape[1], log_p_y_b.shape[0])).T) #p(Ti, δi, yi, bi; θ)
        p_yt = np.matmul(p_y_tb, wgh) + tol4 #Integration of p(Ti, δi, yi, bi; θ) dbi, We add tol4 to avoid 0 in p_yt
        p_b_yt = p_y_tb / p_yt[np.newaxis].T

        #E-step : Expectation
        post_b = np.matmul(p_b_yt, wgh[np.newaxis].T * b) #E(bi | Ti,δi,yi;θ)
        outer_post_b = np.array([np.outer(post_b[i], post_b[i]).flatten() for i in range(len(post_b))])
        post_vb = np.matmul(p_b_yt, wgh[np.newaxis].T * b2) - outer_post_b #var(bi | Ti,δi,yi;θ)

        #E-step : Compute log-likelihood
        log_p_yt = np.log(p_yt)
        log_lik.append(log_p_yt[np.isfinite(log_p_yt)].sum())

        #Check convergence
        if it > 5 and log_lik[it] > log_lik[it-1]:
            thets1 = np.concatenate((y_mat[it-1], np.concatenate([t_mat[risk][it-1] for risk in surv_keys], axis=None), b_mat[it-1].flatten()), axis=None)
            thets2 = np.concatenate((y_mat[it], np.concatenate([t_mat[risk][it] for risk in surv_keys], axis=None), b_mat[it].flatten()), axis=None)
            check1 = max(abs(thets2 - thets1) / (abs(thets1) + tol1)) < tol2
            check2 = (log_lik[it] - log_lik[it-1]) < tol3 * (abs(log_lik[it-1]) + tol3)

            if check1 or check2:
                conv = True
                print('EM algorithm converged! \n calculating Hessian...')
                break

        #M-step : estimated new σ
        z_lme_post_b = (z_lme * post_b[id_l]).sum(axis=1) #Estimated random effects for each longitudinal marquor : Zibi
        mu = (y_lme - eta_yx).flatten() #Difference between longitudinal marquors and his estimation by fixed effects of lme only : yi − Xiβ
        tr_tz_z_vb = (zt_z_lme * post_vb).sum() #tr(t(Zi) Zivbi)
        sigma_hat = np.sqrt((np.matmul(mu, mu-2*z_lme_post_b) + tr_tz_z_vb + np.matmul(z_lme_post_b.T, z_lme_post_b)) / n_long) #Estimated new σ

        #M-step : estimated new D
        dn = np.matmul(p_b_yt, (b2 * wgh[np.newaxis].T)).mean(axis=0).reshape(ncz, ncz)
        d_hat = 0.5 * (dn + dn.T)

        if competing_risks is None:

            #M-step : thetas skeleton (only for gammas, alpha, d_alpha and log_sigma_t)
            pos_betas = None
            pos_sigma = None
            if parametrization == 'value':
                thetas_optim, pos_gammas, pos_alpha, pos_sigma_t = Dict_skel(dict(gammas = gammas, alpha = alpha, sigma_t = {risk:np.log(sigma_t[risk]) for risk in surv_keys}))
                pos_d_alpha = None
            elif parametrization == 'slope':
                thetas_optim, pos_gammas, pos_d_alpha, pos_sigma_t = Dict_skel(dict(gammas = gammas, d_alpha = d_alpha, sigma_t = {risk:np.log(sigma_t[risk]) for risk in surv_keys}))
                pos_alpha = None
            else:
                thetas_optim, pos_gammas, pos_alpha, pos_d_alpha, pos_sigma_t = Dict_skel(dict(gammas = gammas, alpha = alpha, d_alpha = d_alpha, sigma_t = {risk:np.log(sigma_t[risk]) for risk in surv_keys}))
        
        else:
            
            #M-step : thetas skeleton (only for betas, gammas, alpha, d_alpha and log_sigma_t)
            pos_sigma = None
            if parametrization == 'value':
                thetas_optim, pos_betas, pos_gammas, pos_alpha, pos_sigma_t = Dict_skel(dict(betas = betas, gammas = gammas, alpha = alpha, sigma_t = {risk:np.log(sigma_t[risk]) for risk in surv_keys}))
                pos_d_alpha = None
            elif parametrization == 'slope':
                thetas_optim, pos_betas, pos_gammas, pos_d_alpha, pos_sigma_t = Dict_skel(dict(betas = betas, gammas = gammas, d_alpha = d_alpha, sigma_t = {risk:np.log(sigma_t[risk]) for risk in surv_keys}))
                pos_alpha = None
            else:
               thetas_optim, pos_betas, pos_gammas, pos_alpha, pos_d_alpha, pos_sigma_t = Dict_skel(dict(betas = betas, gammas = gammas, alpha = alpha, d_alpha = d_alpha, sigma_t = {risk:np.log(sigma_t[risk]) for risk in surv_keys}))

        #M-step : Dict of variables we will need in function we will call to estimate
        dict_m_step = dict(

            parametrization = parametrization,

            surv_keys = surv_keys,

            y_lme = y_lme,
            y_hat_time = y_hat_time if parametrization in ['value', 'both'] else None,
            y_hat_time_deriv = y_hat_time_deriv if parametrization in ['slope', 'both'] else None,
            y_hat_s = y_hat_s if parametrization in ['value', 'both'] else None,
            y_hat_s_deriv = y_hat_s_deriv if parametrization in ['slope', 'both'] else None,

            x_lme = x_lme,
            x_lme_time = x_lme_time if parametrization in ['value', 'both'] else None,
            x_lme_time_deriv = x_lme_time_deriv if parametrization in ['slope', 'both'] else None,
            x_lme_s = x_lme_s if parametrization in ['value', 'both'] else None,
            x_lme_s_deriv = x_lme_s_deriv if parametrization in ['slope', 'both'] else None,
            xt_x_lme = xt_x_lme,

            z_lme_b = z_lme_b,
            z_lme_time_b = z_lme_time_b if parametrization in ['value', 'both'] else None,
            z_lme_time_b_deriv = z_lme_time_b_deriv if parametrization in ['slope', 'both'] else None,
            z_lme_s_b = z_lme_s_b if parametrization in ['value', 'both'] else None,
            z_lme_s_b_deriv = z_lme_s_b_deriv if parametrization in ['slope', 'both'] else None,
            z_lme_post_b = z_lme_post_b,

            w1 = w1,

            log_time = log_time,
            log_st = log_st,
            st = st,
            p = p,
            d = d,

            eta_tw = eta_tw,
            eta_s = eta_s,
            eta_t = eta_t,

            wint_f_vl = wint_f_vl if parametrization in ['value', 'both'] else None,
            wint_f_sl = wint_f_sl if parametrization in ['slope', 'both'] else None,
            wsint_f_vl = wsint_f_vl if parametrization in ['value', 'both'] else None,
            wsint_f_sl = wsint_f_sl if parametrization in ['slope', 'both'] else None,

            alpha = alpha if parametrization in ['value', 'both'] else None,
            d_alpha = d_alpha if parametrization in ['slope', 'both'] else None,
            sigma_t = sigma_t,
            sigma = sigma,

            pos_betas = pos_betas,
            pos_gammas = pos_gammas,
            pos_alpha = pos_alpha if parametrization in ['value', 'both'] else None,
            pos_d_alpha = pos_d_alpha if parametrization in ['slope', 'both'] else None,
            pos_sigma_t = pos_sigma_t,

            p_b_yt = p_b_yt,
            log_p_tb = log_p_tb,

            wgh = wgh,
            wk = wk,
            id_gk = id_gk,

            ind_fixed = ind_fixed if parametrization in ['slope', 'both'] else None,
            id_l = id_l,
            ncx = ncx,
            scale_wb = scale_wb,
        )

        if competing_risks is None:

            #M-step : estimated new betas
            sc_betas = Gr_long_wb(betas, dict_m_step)
            h_long_betas = H_long_wb(betas, dict_m_step)
            h_betas = Near_pd(h_long_betas)
            betas_hat = betas - np.linalg.solve(h_betas, sc_betas).flatten() #solves the equation np.matmul(a, x) = b for x

            #M-step : estimated other new parameters (by optimisation)
            print('optimisation for surv parameters')
            opt = minimize(
                fun = Opt_surv_wb,
                x0 = thetas_optim,
                method = solver_minimize_em,
                args = (dict_m_step),
                jac = Gr_surv_wb,
                options = options_minimize_em
            )

        else:

            #M-step : estimated other new parameters (by optimisation)
            print('optimisation for long and surv parameters')
            opt = minimize(
                fun = Opt_long_surv_wb,
                x0 = thetas_optim,
                method = solver_minimize_em,
                args = (dict_m_step),
                jac = jac_minimize_em,
                options = options_minimize_em
            )

        #M-step : estimated new thetas
        thetas_optim_hat = opt.x

        #Message about iteration infos (parameters at begining of iter)
        print('log likelihood : ' + str(log_lik[it]))
        print('fixed effects : ' + str(betas))
        print('sigma : ' + str(sigma))
        print('random effects : ' + str(b_mat[it]))
        print('gammas : ' + str(gammas))
        if parametrization in ['value', 'both']:
            print('alpha : ' + str(alpha))
        if parametrization in ['slope', 'both']:
            print('d_alpha : ' + str(d_alpha))
        print('sigma_t : ' + str(sigma_t))

        if competing_risks is None:

            #M_step : Actualisation of all our thetas
            betas = betas_hat
            sigma = sigma_hat
            d_vc = d_hat
            gammas = {risk: thetas_optim_hat[pos_gammas[risk]] for risk in surv_keys}
            alpha = {risk: thetas_optim_hat[pos_alpha[risk]] for risk in surv_keys} if pos_alpha is not None else {risk: None for risk in surv_keys}
            d_alpha = {risk: thetas_optim_hat[pos_d_alpha[risk]] for risk in surv_keys} if pos_d_alpha is not None else {risk: None for risk in surv_keys}
            sigma_t = {risk: np.exp(thetas_optim_hat[pos_sigma_t[risk]]) for risk in surv_keys}
        
        else:

            #M_step : Actualisation of all our thetas
            betas = thetas_optim_hat[pos_betas]
            sigma = sigma_hat
            d_vc = d_hat
            gammas = {risk:thetas_optim_hat[pos_gammas[risk]] for risk in surv_keys}
            alpha = {risk:thetas_optim_hat[pos_alpha[risk]] for risk in surv_keys} if pos_alpha is not None else None
            d_alpha = {risk:thetas_optim_hat[pos_d_alpha[risk]] for risk in surv_keys} if pos_d_alpha is not None else None
            sigma_t = {risk:np.exp(thetas_optim_hat[pos_sigma_t[risk]]) for risk in surv_keys}
    
    #Dictionnary returned by function
    out_dict = {
        'thetas': (
            Alone_to_dict(betas, long_keys), #Redictionnarize longitudinal thetas to be compatible with the following of script
            Alone_to_dict(sigma, long_keys), #Redictionnarize longitudinal thetas to be compatible with the following of script
            d_vc,
            gammas,
            Dapply(alpha, Alone_to_dict, surv_keys, long_keys), #Redictionnarize longitudinal thetas to be compatible with the following of script
            Dapply(d_alpha, Alone_to_dict, surv_keys, long_keys), #Redictionnarize longitudinal thetas to be compatible with the following of script
            sigma_t
        ),
        'conv': conv,
        'z_lme_post_b': {long_keys: z_lme_post_b},
        'post_b': {long_keys: post_b},
        'post_vb': {long_keys: post_vb}
        }
        
    return out_dict