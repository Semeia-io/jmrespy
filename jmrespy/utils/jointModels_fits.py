"""
This file contains all differences method we handle to fit joint models in fit() in ../jointModel.py
"""

####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

# Probability
from scipy.stats import norm
from scipy.stats import qmc
from scipy.stats import multivariate_normal
from scipy.special import expit

# Quadratures
from .quadratures import Gauss_Hermite

# Optimisation
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.optimize import OptimizeResult
from .ntqn import bfgs_e

# Modules from our scripts
from .toolbox import *
from .dmvnorm import Dmvnorm
from .nearpd import Near_pd
from .h_long_wb import H_long_wb
from .gr_long_wb import Gr_long_wb
from .opt_surv_wb import Opt_surv_wb
from .opt_long_surv_wb import Opt_long_surv_wb
from .gr_surv_wb import Gr_surv_wb
from .chol_trans import Chol_transf
from .loglik_weibull import LgLik_weibull
from .loglik_weibull import LgLik_weibull_grad
from .score_weibull_GH import Score_weibullGH
from .fd_vec import Fd_vec
from .cd_vec import Cd_vec
from .cd_hess import Cd_hess
from .expectation_maximization import EM_optim
from .noise_level import Noise_level



####################### -------------------- WeibullPHGH fitting function -------------------- #######################
def WeibullPH_fit(params_matrixes, int_method):
    """
        Function which estimates parameters of the Joint models if the user assume time to event following a Weibull distribution
    Args:
        params_matrixes (dict): Dictionnary containing design matrices and parameters variables we need for estimation of the model
        int_method (str): Specify the method to use for the estimation of random effects integrals, GH for Gauss-Hermite and QMC for Quasi Monte-Carlo
    """

    #Keys of surv objects
    surv_keys = list(params_matrixes['surv_keys'])

    #Competing risk
    competing_risks = params_matrixes['competing_risks']

    #Keys of longitudinal objects
    long_keys = params_matrixes['long_keys']

    # Response vectors
    log_time = params_matrixes['log_time']
    d = params_matrixes['d']
    y_lme = params_matrixes['y_lme']

    # Design matrixes
    x_lme = Dapply(params_matrixes['x_lme'], np.array, long_keys)
    x_lme_time = params_matrixes['x_lme_time']
    x_lme_time_deriv = params_matrixes['x_lme_time_deriv']
    x_lme_s = params_matrixes['x_lme_s']
    x_lme_s_deriv = params_matrixes['x_lme_s_deriv']
    z_lme = params_matrixes['z_lme']
    z_lme_time = params_matrixes['z_lme_time']
    z_lme_time_deriv = params_matrixes['z_lme_time_deriv']
    z_lme_s = params_matrixes['z_lme_s']
    z_lme_s_deriv = params_matrixes['z_lme_s_deriv']
    
    # w1 = w in input of WeibullPHGH_fit but adding for each risk design matrices a column of 1 before first column (intercept) and transformed these matrices in numpy array
    w1 = {}
    for risk in surv_keys:
        w = np.array(params_matrixes['w'][risk])
        w1[risk] = np.c_[np.repeat(1, len(w)), w]

    wint_f_vl = params_matrixes['wint_f_vl']
    wint_f_sl = params_matrixes['wint_f_sl']
    wsint_f_vl = params_matrixes['wsint_f_vl']
    wsint_f_sl = params_matrixes['wsint_f_sl']

    #Random effect variance-covariance matrix associated values
    inv_chol_vc = np.array(params_matrixes['con']['inv_chol_vc'])
    det_inv_chol_vc = params_matrixes['con']['det_inv_chol_vc']

    # Identificators of groups
    id_l = params_matrixes['id_l'] #Each patient id repeted as much as the number of longitudinal statement

    # Family name and link of GLMM
    family_link = params_matrixes['family_link']
    family_name = params_matrixes['family_name']

    #Scale wb
    scale_wb = params_matrixes['scale_wb']

    #Others control parameters
    parametrization = params_matrixes['parametrization']
    tol1 = params_matrixes['con']['tol1']
    tol2 = params_matrixes['con']['tol2']
    tol3 = params_matrixes['con']['tol3']
    tol4 = params_matrixes['con']['tol4']
    eps_hes = params_matrixes['con']['eps_hes']
    numeri_deriv = params_matrixes['con']['numeri_deriv']
    solver_minimize_em = params_matrixes['con']['solver_minimize_em']
    solver_minimize = params_matrixes['con']['solver_minimize']
    options_minimize_em = params_matrixes['con']['options_minimize_em']
    options_minimize = params_matrixes['con']['options_minimize']
    options_noise_minimize = params_matrixes['con']['options_noise_minimize']
    jac_minimize = params_matrixes['con']['jac_minimize']
    jac_minimize_em = params_matrixes['con']['jac_minimize_em']
    eps_f = params_matrixes['con']['eps_f']
    eps_g = params_matrixes['con']['eps_g']

    #Deriv form
    ind_fixed = {key: np.array(params_matrixes['deriv_form'][key]['ind_fixed']) if parametrization[key] in ['slope', 'both'] else None for key in long_keys}
    ind_random = {key: np.array(params_matrixes['deriv_form'][key]['ind_random']) if parametrization[key] in ['slope', 'both'] else None for key in long_keys}

    # Sample size settings
    n_shape = lambda x: x.shape[1]
    ncx_dict = Dapply(x_lme, n_shape, long_keys)
    ncx = sum(ncx_dict.values())
    ncz_dict = Dapply(z_lme, n_shape, long_keys)
    ncz = sum(ncz_dict.values())
    n = len(log_time)
    n_long = Dapply(y_lme, len, long_keys)

    # Crossproducts
    xt_x_lme = Dapply(x_lme, lambda x: np.matmul(x.T, x), long_keys)
    z_splited_lme = {key: np.split(z_lme[key], np.unique(id_l[key], return_index=True)[1][1:]) for key in long_keys} #Random effects of each patient splitted
    zt_z_lme = Dapply(z_splited_lme, lambda z: np.array([np.matmul(zi.T, zi).flatten() for zi in z]), long_keys) #Each line correspond to flatten crossprod of random effects for each group

    if int_method == 'GH':
        # Gauss-Hermite quadrature rule components (used to approximate integrals)
        ghk = params_matrixes['con']['ghk']
        gh_weights, gh_nodes = Gauss_Hermite(ghk)
        input_grid_points = {'Var{}'.format(i+1): gh_nodes for i in range(ncz)}
        b = np.array(Expandgrid(input_grid_points)) #each gh node associated with the ghk gh nodes (the current gh node included, so we have ghk^ncz associations)
        k = b.shape[0]
        input_grid_weights = {'Var{}'.format(i+1):gh_weights for i in range(ncz)}
        wgh = np.array(Expandgrid(input_grid_weights))
        wgh = 2**(ncz/2) * np.prod(wgh, axis=1) * np.exp(np.sum(b*b, axis=1)) #Why D.Rizopoulos does this transformation on wgh, might have a report with the density function of our model
        wgh = wgh * det_inv_chol_vc
        b = np.sqrt(2) * (np.matmul(inv_chol_vc, b.T)).T
        b2 = b*b if ncz==1 else np.array([np.outer(b[i], b[i]).flatten() for i in range(len(b))])

        # Computing matricial products between different random design matrices (Z) and random effects (b) we need for log-likelihood
        z_lme_b, z_lme_time_b, z_lme_time_b_deriv, z_lme_s_b, z_lme_s_b_deriv = Z_b(
            long_keys = long_keys,
            ncz_dict = ncz_dict,
            z_lme = z_lme,
            z_lme_time = z_lme_time,
            z_lme_s = z_lme_s,
            z_lme_s_deriv = z_lme_s_deriv,
            z_lme_time_deriv = z_lme_time_deriv,
            b = b,
            k = k,
            ind_random = ind_random,
            parametrization = parametrization
        )

    # Gauss-Kronrod quadrature rule components
    st = params_matrixes['st']
    log_st = np.log(st)[np.newaxis].T
    wk = np.tile(params_matrixes['wk'], len(log_time))[np.newaxis].T
    p = params_matrixes['p'][np.newaxis].T
    id_gk = np.repeat(range(len(log_time)), params_matrixes['con']['gkk'])

    # Initial values
    betas = params_matrixes['initial_values']['betas']
    sigma = params_matrixes['initial_values']['sigma']
    d_vc = params_matrixes['initial_values']['d_vc']
    diag_d = d_vc.ndim == 1 #Indicates if d_vc is a vector (case where we only take the vc diagonal when vc is diagonal matrix) or not (so, a matrix)
    gammas = {risk : params_matrixes['initial_values'][risk]['gammas'] for risk in surv_keys}
    alpha = {risk : params_matrixes['initial_values'][risk]['alpha'] for risk in surv_keys}
    d_alpha = {risk : params_matrixes['initial_values'][risk]['d_alpha'] for risk in surv_keys}
    sigma_t = {risk : params_matrixes['initial_values'][risk]['sigma_t'] for risk in surv_keys}

    # Constant values
    betas_c = params_matrixes['constant_values']['betas']
    sigma_c = params_matrixes['constant_values']['sigma']
    d_vc_c = params_matrixes['constant_values']['d_vc']
    gammas_c = {risk : params_matrixes['constant_values'][risk]['gammas'] for risk in surv_keys}
    alpha_c = {risk : params_matrixes['constant_values'][risk]['alpha'] for risk in surv_keys}
    d_alpha_c = {risk : params_matrixes['constant_values'][risk]['d_alpha'] for risk in surv_keys}
    sigma_t_c = {risk : params_matrixes['constant_values'][risk]['sigma_t'] for risk in surv_keys}

    # conv : convergence indicator
    conv = False



    ### EM algorithm ---
    
    # Dictionnary in input for EM
    iter_em = params_matrixes['con']['iter_em']
    if iter_em > 0:
        dict_EM = {

            # Thetas vars
            'betas': betas,
            'sigma': sigma,
            'd_vc': d_vc,
            'gammas': gammas,
            'alpha': alpha,
            'd_alpha': d_alpha,
            'sigma_t': sigma_t,

            # Control vars
            'iter_em': iter_em,
            'solver_minimize_em': solver_minimize_em,
            'options_minimize_em': options_minimize_em,
            'jac_minimize_em': jac_minimize_em,
            'parametrization': parametrization,
            'competing_risks': competing_risks,
            'tol1': tol1,
            'tol2': tol2,
            'tol3': tol3,
            'tol4': tol4,

            # Survival vars
            'surv_keys': surv_keys,
            'log_time': log_time,
            'st': st,
            'log_st': log_st,
            'p': p,
            'd': d,
            'w1': w1,
            'wint_f_vl': wint_f_vl,
            'wint_f_sl': wint_f_sl,
            'wsint_f_vl': wsint_f_vl,
            'wsint_f_sl': wsint_f_sl,
            'wk': wk,
            'id_gk': id_gk,
            'n': n,
            'scale_wb': scale_wb,

            # Longitudinal vars
            'long_keys': long_keys,
            'y_lme': y_lme,
            'x_lme': x_lme,
            'xt_x_lme': xt_x_lme,
            'x_lme_time': x_lme_time,
            'x_lme_time_deriv': x_lme_time_deriv,
            'x_lme_s': x_lme_s,
            'x_lme_s_deriv': x_lme_s_deriv,
            'z_lme': z_lme,
            'zt_z_lme': zt_z_lme,
            'z_lme_b': z_lme_b,
            'z_lme_time_b': z_lme_time_b,
            'z_lme_time_b_deriv': z_lme_time_b_deriv,
            'z_lme_s_b': z_lme_s_b,
            'z_lme_s_b_deriv': z_lme_s_b_deriv,
            'ind_fixed': ind_fixed,
            'id_l': id_l,
            'ncx_dict': ncx_dict,
            'ncz_dict': ncz_dict,
            'n_long': n_long,
            'wgh': wgh,

            #b vars
            'b': b,
            'b2': b2            
        }

        #Estimation of parameters
        out_em = EM_optim(dict_EM)
        betas, sigma, d_vc, gammas, alpha, d_alpha, sigma_t = out_em['thetas']
        conv = out_em['conv']
        z_lme_post_b = out_em['z_lme_post_b']
        post_b = out_em['post_b']
        post_vb = out_em['post_vb']

    #log(only on diagonal) chol of d_vc
    d_vc_log = np.log(d_vc) if diag_d else Chol_transf(d_vc)

    #New thetas skeleton (including all thetas parameters)
    dict_skel = {'betas': betas, 'sigma': Dapply(sigma, lambda x: np.log(x) if x is not None else None, long_keys), 'gammas': gammas, 'alpha': alpha, 'd_alpha': d_alpha, 'sigma_t': {risk:np.log(sigma_t[risk]) for risk in surv_keys}, 'd_vc': d_vc_log}
    thetas, pos_betas, pos_sigma, pos_gammas, pos_alpha, pos_d_alpha, pos_sigma_t, pos_d_vc = Dict_skel(dict_skel)
    
    #Thetas skeleton of constant parameters
    dict_skel_c = {'betas': betas_c, 'sigma': Dapply(sigma_c, lambda x: np.log(x) if x is not None else None, long_keys), 'gammas': gammas_c, 'alpha': alpha_c, 'd_alpha': d_alpha_c, 'sigma_t': {risk:np.log(sigma_t_c[risk]) for risk in surv_keys}, 'd_vc': d_vc_c}
    thetas_c, pos_betas_c, pos_sigma_c, pos_gammas_c, pos_alpha_c, pos_d_alpha_c, pos_sigma_t_c, pos_d_vc_c = Dict_skel(dict_skel_c)
    
    #Vector of thetas without fixed parameters
    thetas_var = thetas[np.isnan(thetas_c)]

    #Dict of variables we will need in function we will call to estimate log likelihood in optimization functions (only variable remaining the same between QMC and GH integration)
    optim_dict = {

        'parametrization': parametrization,
        'family_name': family_name,

        'surv_keys': surv_keys,
        'long_keys': long_keys,

        'y_lme': y_lme,

        'x_lme': x_lme,
        'x_lme_time': x_lme_time,
        'x_lme_time_deriv': x_lme_time_deriv,
        'x_lme_s': x_lme_s,
        'x_lme_s_deriv': x_lme_s_deriv,

        'z_lme': z_lme,
        'zt_z_lme': zt_z_lme,

        'inv_chol_vc': inv_chol_vc,
        'det_inv_chol_vc': det_inv_chol_vc,
        'diag_d': diag_d,
        'pos_d_vc': pos_d_vc,

        'w1': w1,
        
        'log_time': log_time,
        'log_st': log_st,
        'p': p,
        'd': d,

        'wint_f_vl': wint_f_vl,
        'wint_f_sl': wint_f_sl,
        'wsint_f_vl': wsint_f_vl,
        'wsint_f_sl': wsint_f_sl,

        'init_thetas': thetas,

        'pos_betas': pos_betas,
        'pos_sigma': pos_sigma,
        'pos_gammas': pos_gammas,
        'pos_alpha': pos_alpha,
        'pos_d_alpha': pos_d_alpha,
        'pos_sigma_t': pos_sigma_t,

        'wk': wk,
        'id_gk': id_gk,

        'ind_fixed': ind_fixed,
        'id_l': id_l,
        'ncz': ncz,
        'n': n,
        'n_long': n_long,
        'scale_wb': scale_wb,
        'tol4': tol4,
        'int_method': int_method,

        'thetas_c': thetas_c,
        'rescale': True,
        'n_cores_grad': params_matrixes['con']['n_cores_grad']
    }

    #Adding to optim_dict supplementary variables we need only for GH integration
    if int_method == 'GH':
        optim_dict['b'] = b
        optim_dict['z_lme_b'] = z_lme_b
        optim_dict['z_lme_time_b'] = z_lme_time_b
        optim_dict['z_lme_time_b_deriv'] = z_lme_time_b_deriv
        optim_dict['z_lme_s_b'] = z_lme_s_b
        optim_dict['z_lme_s_b_deriv'] = z_lme_s_b_deriv
        optim_dict['wgh'] = wgh

    #Adding to optim_dict supplementary variables we need only for QMC integration
    if int_method == 'QMC':
        optim_dict['ncz_dict'] = ncz_dict
        optim_dict['z_lme'] = z_lme
        optim_dict['z_lme_time'] = z_lme_time
        optim_dict['z_lme_time_deriv'] = z_lme_time_deriv
        optim_dict['z_lme_s'] = z_lme_s
        optim_dict['z_lme_s_deriv'] = z_lme_s_deriv
        optim_dict['ind_random'] = ind_random
        optim_dict['nbmc'] = params_matrixes['con']['nbmc']
        optim_dict['seed_MC'] = params_matrixes['con']['seed_MC']

    #Computing log likelihood
    log_lik = - LgLik_weibull(thetas_var, optim_dict)

    
    ### Minimization of Log likelihood without EM (If EM didn't converged) ---
    if not conv:

        #Callback for numerical optimization which records values of function and thetas at each iteration
        def callback_display(thetas_iter):
            optim_dict['iter'] += 1

            lglik_record = optim_dict['lglik_record']
            thetas_record = optim_dict['thetas_record']

            #lglik_iter = - LgLik_weibull(thetas_iter, optim_dict)
            lglik_iter = 0

            #Adding current thetas and loglik in recording vectors
            lglik_record = np.concatenate((lglik_record, lglik_iter), axis=None)
            thetas_record = np.concatenate((thetas_record, [Thetas_feed(thetas_iter, optim_dict['thetas_c'])]), axis=0)

            optim_dict['lglik_record'] = lglik_record
            optim_dict['thetas_record'] = thetas_record

            #Displaying state of minimization
            if optim_dict['int_method'] == 'QMC':
                print('\nIter : {}\nThetas : {} \nFun : {} \nnbmc : {}'.format(optim_dict['iter'], thetas_iter, lglik_iter, optim_dict['nbmc']))
            elif optim_dict['int_method'] == 'GH':
                print('\nIter : {}\nThetas : {} \nFun : {}'.format(optim_dict['iter'], thetas_iter, lglik_iter))

        #Optimization of thetas with defined jacobian function (only for one gaussian longitudinal marker without competing risks)
        if competing_risks is None and len(long_keys) == 0 and family_name[long_keys[0]] == 'gaussian':

            #Estimation of thetas by one of scipy.optimize.minimize solvers
            out = minimize(
                fun = LgLik_weibull,
                x0 = thetas_var,
                method = solver_minimize,
                args = (optim_dict),
                jac = Score_weibullGH,
                options = options_minimize
            )
        
        #Optimisation of thetas for other cases
        else:

            #For QMC estimation of integral : Replacing classical finite derivate estimation of gradient by a similar function which adpated for QMC
            if jac_minimize is None and int_method == 'QMC':
                jac_minimize = LgLik_weibull_grad

            optim_dict['iter'] = 0
            optim_dict['lglik_record'] = np.array([- LgLik_weibull(thetas_var, optim_dict)])
            optim_dict['thetas_record'] = np.array([Thetas_feed(thetas_var, thetas_c)])

            #Estimation of thetas by one of scipy.optimize.minimize solvers
            out = minimize(
                fun = LgLik_weibull,
                x0 = thetas_var,
                method = solver_minimize,
                args = (optim_dict),
                jac = jac_minimize,
                options = options_minimize,
                callback = callback_display
            )

            #Enhancement of thetas optimization for noisy functions with a noisy-tolerant BFGS algorithm
            if int_method == 'QMC':

                #Flag for succes of line search
                ls_success = False
                
                #Count of repetion of bfgs_e due to line search fail
                ls_try = 0
                ls_try_max = 30

                while ls_success is False and ls_try < ls_try_max:

                    #Extraction of new set of initial paramters for minimization
                    x_init = out.x if isinstance(out, OptimizeResult) else out[0]

                    #Estimation of noise level
                    eps_g, eps_f = Noise_level(x=x_init, optim_dict=optim_dict, n_rep=30, prob=0.99)

                    nb_try = 3
                    try_count = 0

                    #Sometimes a bad sampling could lead to error at first iteration
                    while try_count < nb_try:
                        try:
                            out = bfgs_e(
                                    func = LgLik_weibull,
                                    grad = jac_minimize,
                                    x0 = x_init,
                                    eps_f = eps_f,
                                    eps_g = eps_g,
                                    args = (optim_dict),
                                    options = options_noise_minimize,
                                    callback = callback_display
                                )
                            break
                        except:
                            print("An error occured during e-bfgs, we retry")
                            try_count += 1
                    
                    #Avoiding infinite while loop if max tries was reached for bfgs_e
                    if try_count == 3:
                        break

                    #Checking if bfgs_e was terminated because of line search fail
                    if out[5] != 8:
                        ls_success = True
                    #Increment the count of line search fails if the update of noise level didn't permited noise tolerant L-BFGS to progress
                    elif (out[0] == x_init).all():
                        ls_try += 1
                    #Reset the count of line search fails if noise tolerant L-BFGS progressed
                    else:
                        ls_try = 0


        #Actualisation of all our thetas
        thetas_hat = Thetas_feed(out.x, thetas_c) if isinstance(out, OptimizeResult) else Thetas_feed(out[0], thetas_c)
        extract_pos = lambda x, thetas_hat: thetas_hat[x] if x is not None else None
        extract_pos_exp = lambda x, thetas_hat: np.exp(thetas_hat[x]) if x is not None else None
        extract_pos_alpha = lambda x, thetas_hat: Dapply(x, extract_pos, long_keys, thetas_hat)
        betas = Dapply(pos_betas, extract_pos, long_keys, thetas_hat)
        sigma = Dapply(pos_sigma, extract_pos_exp, long_keys, thetas_hat)
        d_vc = thetas_hat[pos_d_vc]
        d_vc = np.exp(d_vc) if diag_d else Chol_transf(d_vc)['res'] #chol of d_vc : Vec ----> Matrix
        gammas = Dapply(pos_gammas, extract_pos, surv_keys, thetas_hat)
        alpha = Dapply(pos_alpha, extract_pos_alpha, surv_keys, thetas_hat)
        d_alpha = Dapply(pos_d_alpha, extract_pos_alpha, surv_keys, thetas_hat)
        sigma_t = Dapply(pos_sigma_t, extract_pos_exp, surv_keys, thetas_hat)

        #Compute posterior moments for thetas after quasi-Newton
        if isinstance(out, OptimizeResult): #Condition if out is an OptimizeResult
            condition = out.success or - out.fun > log_lik
        else:
            condition = out[5] in np.array([0, 4, 5]) or - out[1] > log_lik
        if condition:

            log_lik = - out.fun if isinstance(out, OptimizeResult) else - out[1]

            #Simulation of b for QMC integration
            nbmc = params_matrixes['con']['nbmc']
            seed_MC = params_matrixes['con']['seed_MC']
            if int_method == 'QMC':
                dist = qmc.MultivariateNormalQMC(mean=np.zeros(ncz), cov=d_vc, seed=seed_MC)
                b = dist.random(nbmc)
                k = b.shape[0]
                z_lme_b, z_lme_time_b, z_lme_time_b_deriv, z_lme_s_b, z_lme_s_b_deriv = Z_b(
                    long_keys = long_keys,
                    ncz_dict = ncz_dict,
                    z_lme = z_lme,
                    z_lme_time = z_lme_time,
                    z_lme_s = z_lme_s,
                    z_lme_s_deriv = z_lme_s_deriv,
                    z_lme_time_deriv = z_lme_time_deriv,
                    b = b,
                    k = k,
                    ind_random = ind_random,
                    parametrization = parametrization
                )
            
            #Probability density function of our random effects b
            b_pdf = multivariate_normal.pdf(x=b, mean=np.repeat(0, ncz), cov=d_vc) + 10e-30

            #Linear predictors
            eta_yx = {} #Estimated y or logit(p(y=1)) by fixed effects of lme : eta_yx = beta0 + beta1*x1 + beta2*x2 + ... + betap*xp
            eta_tw = {risk : np.matmul(w1[risk], gammas[risk])[np.newaxis].T for risk in surv_keys} #Estimated value of exponential part of proportional risks model : h(t) = h(0) * exp(eta_tw) without link with longitundinal markor
            eta_t = eta_tw.copy() #Estimated eta including g(b, alpha, t) the link between longitudinal markor and survival
            eta_s = {risk: 0 for risk in surv_keys} #g(b, alpha, t) values of quadrature nodes for gauss kronrod quadrature to compute S(t)

            for key in long_keys:
                eta_yx[key] = np.matmul(x_lme[key], betas[key])[np.newaxis].T #Estimated y or logit(p(y=1)) by fixed effects of lme : eta_yx = beta0 + beta1*x1 + beta2*x2 + ... + betap*xp
                if family_name[key] == 'gaussian':
                    if parametrization[key] in ['value', 'both']:
                        y_hat_time = np.matmul(x_lme_time[key], betas[key])[np.newaxis].T + z_lme_time_b[key] #Estimated y at times to event by lme (fixed effects and random effects) : y_i(ti)
                        y_hat_s = np.matmul(x_lme_s[key], betas[key])[np.newaxis].T + z_lme_s_b[key] #Estimated y at all stime_lag_0 (defined in jointModel.py) times to event by lme (fixed effects and random effects)
                        for risk in surv_keys:
                            eta_t[risk] = eta_t[risk] + wint_f_vl * alpha[risk][key] * y_hat_time #Adding alpha*mi(t) to eta_t only if mi(t) the estimated value of longitudinal markor, is in g(b, alpha, t)
                            eta_s[risk] = eta_s[risk] + wsint_f_vl * alpha[risk][key] * y_hat_s #Adding alpha*mi(t) to eta_s only if mi(t) the estimated value of longitudinal markor, is in g(b, alpha, t)        
                    if parametrization[key] in ['slope', 'both']:
                        y_hat_time_deriv = np.matmul(x_lme_time_deriv[key], betas[key][ind_fixed[key]])[np.newaxis].T + z_lme_time_b_deriv[key]
                        y_hat_s_deriv = np.matmul(x_lme_s_deriv[key], betas[key][ind_fixed[key]])[np.newaxis].T + z_lme_s_b_deriv[key]
                        for risk in surv_keys:
                            eta_t[risk] = eta_t[risk] + wint_f_sl * d_alpha[risk][key] * y_hat_time_deriv #Adding alpha*m'i(t) to eta_t only if mi(t) the estimated derivation of longitudinal markor, is in g(b, alpha, t)
                            eta_s[risk] = eta_s[risk] + wsint_f_sl * d_alpha[risk][key] * y_hat_s_deriv #Adding alpha*m'i(t) to eta_s only if m'i(t) the estimated derivation of longitudinal markor, is in g(b, alpha, t)
                elif family_name[key] == 'bernoulli':
                    if parametrization[key] in ['value', 'both']:
                        y_hat_time = expit(np.matmul(x_lme_time[key], betas[key])[np.newaxis].T + z_lme_time_b[key]) #Estimated y at times to event by lme (fixed effects and random effects) : y_i(ti)
                        y_hat_s = expit(np.matmul(x_lme_s[key], betas[key])[np.newaxis].T + z_lme_s_b[key]) #Estimated y at all stime_lag_0 (defined in jointModel.py) times to event by lme (fixed effects and random effects)
                        for risk in surv_keys:
                            eta_t[risk] = eta_t[risk] + wint_f_vl * alpha[risk][key] * y_hat_time #Adding alpha*mi(t) to eta_t only if mi(t) the estimated value of longitudinal markor, is in g(b, alpha, t)
                            eta_s[risk] = eta_s[risk] + wsint_f_vl * alpha[risk][key] * y_hat_s #Adding alpha*mi(t) to eta_s only if mi(t) the estimated value of longitudinal markor, is in g(b, alpha, t)        
                    if parametrization[key] in ['slope', 'both']:
                        y_hat_time_deriv = expit(np.matmul(x_lme_time_deriv[key], betas[key][ind_fixed[key]])[np.newaxis].T + z_lme_time_b_deriv[key])
                        y_hat_s_deriv = expit(np.matmul(x_lme_s_deriv[key], betas[key][ind_fixed[key]])[np.newaxis].T + z_lme_s_b_deriv[key])
                        for risk in surv_keys:
                            eta_t[risk] = eta_t[risk] + wint_f_sl * d_alpha[risk][key] * y_hat_time_deriv #Adding alpha*m'i(t) to eta_t only if mi(t) the estimated derivation of longitudinal markor, is in g(b, alpha, t)
                            eta_s[risk] = eta_s[risk] + wsint_f_sl * d_alpha[risk][key] * y_hat_s_deriv #Adding alpha*m'i(t) to eta_s only if m'i(t) the estimated derivation of longitudinal markor, is in g(b, alpha, t)

            #Likelihoods
            log_p_y_b = 0 #log p(yi | bi;θy)
            for key in long_keys:
                if family_name[key] == 'gaussian':
                    mu_y = eta_yx[key] + z_lme_b[key] #Estimated y by lme (fixed effects and random effects)
                    log_norm = np.log(norm.pdf(y_lme[key], loc=np.array(mu_y), scale=sigma[key])+10e-30) #log likelihood of our y_lme following N(mu = mu_y, sigma = sigma) law
                    log_p_y_b = log_p_y_b + np.array(pd.DataFrame(log_norm).groupby(id_l[key]).sum()) #Sum of log_norm likelihoods by group : log p(yi | bi;θy)
                elif family_name[key] == 'bernoulli':
                    logit_p_y = eta_yx[key] + z_lme_b[key] #Estimated logit(p(y=1)) by lme (fixed effects and random effects)
                    log_p_y = np.log((expit(logit_p_y)**y_lme[key]) * ((1-expit(logit_p_y))**(1-y_lme[key])) + 10e-30) #log_probability of observing real y with our model 
                    log_p_y_b = log_p_y_b + np.array(pd.DataFrame(log_p_y).groupby(id_l[key]).sum()) #Sum of log_probability by group : log p(yi | bi;θy)
            log_hazard = np.array([(np.log(sigma_t[risk]) + (sigma_t[risk] - 1) * log_time + eta_t[risk]) * d[risk] for risk in surv_keys]).sum(axis=0) #Instantaneous risk * d
            log_survival = np.array([-np.exp(eta_tw[risk]) * p * np.array(pd.DataFrame(wk * np.exp(np.log(sigma_t[risk]) + (sigma_t[risk] - 1) * log_st + eta_s[risk])).groupby(id_gk).sum()) for risk in surv_keys]).sum(axis=0) #Survival function
            log_p_tb = log_hazard + log_survival #Log likelihood of survival part in joint model : log p(Ti,δi | bi;θt,β)
            log_p_b = np.repeat(np.log(multivariate_normal.pdf(x=b, mean=np.repeat(0, ncz), cov=d_vc) + 10e-30), n) #Log likelihood of random effects : log p(bi | Vech(D))
            p_y_tb = np.exp(log_p_y_b + log_p_tb + log_p_b.reshape((log_p_y_b.shape[1], log_p_y_b.shape[0])).T) #p(Ti, δi, yi, bi; θ)
            p_yt = np.matmul(p_y_tb, 1/b_pdf)/nbmc + tol4 if int_method == 'QMC' else np.matmul(p_y_tb, wgh) + tol4 #Integration of p(Ti, δi, yi, bi; θ) dbi
            #p_yt = np.matmul(p_y_tb, wgh) + tol4 #Integration of p(Ti, δi, yi, bi; θ) dbi
            p_b_yt = p_y_tb / p_yt[np.newaxis].T
            
            #Expectation for b
            if int_method == 'GH':
                post_b = np.matmul(p_b_yt, wgh[np.newaxis].T * b) #E(bi | Ti,δi,yi;θ)
                outer_post_b = np.array([np.outer(post_b[i], post_b[i]).flatten() for i in range(len(post_b))])
                post_vb = np.matmul(p_b_yt, wgh[np.newaxis].T * b2) - outer_post_b #var(bi | Ti,δi,yi;θ)
                z_lme_post_b = {} #Estimated random effects for each longitudinal marquor : Zibi
                i = 0
                for key in long_keys:
                    ncz_key = ncz_dict[key]
                    i_end = i+ncz_key
                    z_lme_b[key] = np.matmul(z_lme[key], b[:, i:i_end].T)
                    z_lme_post_b[key] = z_lme[key] * post_b[id_l[key], i:i_end] if ncz_key == 1 else (z_lme[key] * post_b[id_l[key], i:i_end]).sum(axis=1)
                    i = i_end
            else:
                post_b = None
                post_vb = None
                z_lme_post_b = None

    else:
        out = None
        thetas_hat = thetas


    ### Computing output results ---

    #Suppression of constant thetas after optimisation of parameters. If we don't suppress them, they will continue to overwrite thetas in hessian estimation. (We want all thetas parameters to variate to estimates hessian matrix)
    optim_dict['thetas_c'] = np.repeat(np.nan, len(thetas_hat))

    #Thetas names
    thetas_names = np.repeat('name', thetas.shape).astype('<U256')

    for key in long_keys:
        #betas
        thetas_names[pos_betas[key]] = params_matrixes['x_lme'][key].columns + '_' + key
        #sigma
        if pos_sigma[key] is not None:
            thetas_names[pos_sigma[key]] = 'sigma_{}'.format(key)
    for risk in surv_keys:
        #gammas
        thetas_names[pos_gammas[risk]] = np.concatenate((np.array(['Intercept_'+risk]), np.array([col+'_'+risk for col in params_matrixes['w'][risk].columns])),axis=None)
        #sigma_t
        thetas_names[pos_sigma_t[risk]] = 'log(sigma_t)_{}'.format(risk)
        for key in long_keys:
            if parametrization[key] in ['value', 'both']:
                #alpha
                thetas_names[pos_alpha[risk][key]] = 'alpha_value_{}_{}'.format(risk, key)
            if parametrization[key] in ['slope', 'both']:
                #d_alpha
                thetas_names[pos_d_alpha[risk][key]] = 'alpha_slope_{}_{}'.format(risk, key)
    #d_vc
    thetas_names[pos_d_vc] = np.array(['b_'+str(i) for i in range(len(d_vc_log))])

    #thetas in a dataframe
    thetas_out = pd.DataFrame(thetas_hat[np.newaxis], columns=thetas_names)

    #thetas skeleton (pos of each thetas by name)
    thetas_skel = {
        'pos_betas': pos_betas,
        'pos_sigma': pos_sigma,
        'pos_gammas': pos_gammas,
        'pos_alpha': pos_alpha,
        'pos_d_alpha': pos_d_alpha,
        'pos_sigma_t': pos_sigma_t,
        'pos_d_vc': pos_d_vc
    }

    #Log-likelihood of our final model
    log_lik = - LgLik_weibull(thetas_hat, optim_dict)

    #Gradients of our log-likelihood function with our thetas
    if competing_risks is None and len(long_keys) == 0 and family_name[long_keys[0]] == 'gaussian':
        score = Score_weibullGH(thetas_hat, optim_dict)
    else:
        score = None

    #Hessian
    if int_method == 'QMC':
        #Avoiding rescaling before computing Hessian by central finite difference
        optim_dict['rescale'] = False

    if competing_risks is None and len(long_keys) == 0 and family_name[long_keys[0]] == 'gaussian':
        if numeri_deriv == 'fd':
            hessian = Fd_vec(thetas_hat, Score_weibullGH, eps_hes, optim_dict)
        elif numeri_deriv == 'cd_mat':
            hessian = Cd_hess(thetas_hat, LgLik_weibull, eps_hes, optim_dict)
        elif numeri_deriv == 'optimizer_hess_inv':
            #Sometimes, inverse hessian matrix is a scipy.sparse.linalg.LinearOperator who needs to use .todense() method to be converted to numpy array
            if 'todense' in dir(out.hess_inv):
                hessian = np.linalg.inv(out.hess_inv.todense())
            else:
                hessian =  np.linalg.inv(out.hess_inv)
        else:
            hessian = Cd_vec(thetas_hat, Score_weibullGH, eps_hes, optim_dict)
    else:
        if numeri_deriv == 'optimizer_hess_inv':
            
            if 'todense' in dir(out.hess_inv):
                hessian =  np.linalg.inv(out.hess_inv.todense())
            else:
                hessian =  np.linalg.inv(out.hess_inv)
        else:
            hessian = Cd_hess(thetas_hat, LgLik_weibull, eps_hes, optim_dict)

    hessian_out = pd.DataFrame(hessian, index=thetas_names, columns=thetas_names)



    ### Return estimated parameters ---

    #Params to return
    params_return = {

        #Coefficients
        'coefficients':{
            'thetas': thetas_out,
            'thetas_skel': thetas_skel,
            'thetas_names': thetas_names,
            'betas': betas,
            'sigma': sigma,
            'gammas': gammas,
            'alpha': alpha,
            'd_alpha': d_alpha,
            'sigma_t': sigma_t,
            'd_vc': d_vc,
        },

        #Random effects
        'ranef':{
            'post_b': post_b,
            'post_vb': post_vb,
            'z_lme_post_b': z_lme_post_b
        },

        #Infos about performance
        'score': score,
        'hessian': hessian_out,
        'log_lik': log_lik,
        'log_lik_optim': optim_dict['lglik_record'],
        'thetas_optim': pd.DataFrame(optim_dict['thetas_record'], columns=thetas_names),
        'success': out.success if 'success' in dir(out) else None,
        'conv_message': out.message if 'message' in dir(out) else None,
        

        #Control parameters
        'control': params_matrixes['con']
    }
        
    return(params_return)