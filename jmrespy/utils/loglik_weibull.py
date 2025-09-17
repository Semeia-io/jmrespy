"""
Function to minimize (for optimization) for survival parameters in Quasi Newton
"""

####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import numpy as np

# Probability
from scipy.stats import multivariate_normal
from scipy.optimize import approx_fprime

# Parralellization
import mpire
from tqdm import tqdm

# Modules from our scripts
from .chol_trans import Chol_transf
from .toolbox import *
from .indiv_densities import Indiv_densities
from .rescale_b_QMC import Rescale_b_QMC
from .fd_vec import Diff_f_term

#from memory_profiler import profile
#fp=open('memory_profiler.log', 'w+')
#@profile(stream=fp)
#@profile
def LgLik_weibull(thetas, *args):
    """
        Function which computes the log-likelihood of a joint model for longitudinal and survival data with a time-to-event assumed distributed whith a Weibull distribution
    Args:
        thetas (np.array) : Vector of theta parameters (without fixed thetas which are inputed in *args)
        *args (dict): Dictionnary containing design matrices and parameters variables we need for computation of log-likelihood
    """

    #v_dict : Our dictionnary passed in *args
    v_dict = args[0]

    #Extract variables we need from dict (common between QMC and GH integration)
    ind_fixed = v_dict['ind_fixed']
    parametrization = v_dict['parametrization']
    x_lme = v_dict['x_lme']
    x_lme_time = v_dict['x_lme_time']
    x_lme_time_deriv = v_dict['x_lme_time_deriv']
    x_lme_s = v_dict['x_lme_s']
    x_lme_s_deriv = v_dict['x_lme_s_deriv']
    w1 = v_dict['w1']
    y_lme = v_dict['y_lme']
    wint_f_vl = v_dict['wint_f_vl']
    wint_f_sl = v_dict['wint_f_sl']
    wsint_f_vl = v_dict['wsint_f_vl']
    wsint_f_sl = v_dict['wsint_f_sl']
    id_l = v_dict['id_l']
    id_gk = v_dict['id_gk']
    wk = v_dict['wk']
    log_time = v_dict['log_time']
    log_st = v_dict['log_st']
    p = v_dict['p']
    d = v_dict['d']
    ncz = v_dict['ncz']
    n = v_dict['n']
    det_inv_chol_vc = v_dict['det_inv_chol_vc']
    diag_d = v_dict['diag_d']
    scale_wb = v_dict['scale_wb']
    pos_betas = v_dict['pos_betas']
    pos_sigma = v_dict['pos_sigma']
    pos_gammas = v_dict['pos_gammas']
    pos_alpha = v_dict['pos_alpha']
    pos_d_alpha = v_dict['pos_d_alpha']
    pos_sigma_t = v_dict['pos_sigma_t']
    pos_d_vc = v_dict['pos_d_vc']
    tol4 = v_dict['tol4']
    surv_keys = v_dict['surv_keys']
    long_keys = v_dict['long_keys']
    family_name = v_dict['family_name']
    int_method = v_dict['int_method']
    thetas_c = v_dict['thetas_c']

    #Extract variables we need from dict (only GH integration)
    if int_method == 'GH':
        b = v_dict['b']
        z_lme_b = v_dict['z_lme_b']
        z_lme_time_b = v_dict['z_lme_time_b']
        z_lme_time_b_deriv = v_dict['z_lme_time_b_deriv']
        z_lme_s_b = v_dict['z_lme_s_b']
        z_lme_s_b_deriv = v_dict['z_lme_s_b_deriv']
        wgh = v_dict['wgh']

    #Extract variables we need from dict (only QMC integration)
    if int_method == 'QMC':
        ncz_dict = v_dict['ncz_dict']
        z_lme = v_dict['z_lme']
        z_lme_time = v_dict['z_lme_time']
        z_lme_time_deriv = v_dict['z_lme_time_deriv']
        z_lme_s = v_dict['z_lme_s']
        z_lme_s_deriv = v_dict['z_lme_s_deriv']
        ind_random = v_dict['ind_random']
        nbmc = v_dict['nbmc']
        seed_MC = v_dict['seed_MC']
    
    #Insertion of fixed thetas inside variable thetas vector (we insert if fixed thetas[i] is not na)
    thetas = Thetas_feed(thetas, thetas_c)

    #Extract thetas
    betas = {key: thetas[pos_betas[key]] for key in long_keys}
    sigma = {key: np.exp(thetas[pos_sigma[key]]) if family_name[key] == 'gaussian' else None for key in long_keys}
    gammas = {risk: thetas[pos_gammas[risk]] for risk in surv_keys}
    alpha = {risk: {key: thetas[pos_alpha[risk][key]] if pos_alpha[risk][key] is not None else None for key in long_keys} for risk in surv_keys}
    d_alpha = {risk: {key: thetas[pos_d_alpha[risk][key]] if pos_d_alpha[risk][key] is not None else None for key in long_keys} for risk in surv_keys}
    sigma_t = {risk: np.exp(thetas[pos_sigma_t[risk]]) for risk in surv_keys} if scale_wb is None else scale_wb
    d_vc = np.exp(thetas[pos_d_vc]) if diag_d else Chol_transf(thetas[pos_d_vc])['res']

    #Computation of individual likelihood through GH estimation of integral over random effects
    if int_method == 'GH':

        #p(Ti, δi, yi, bi; θ)
        p_y_tb = Indiv_densities(
            surv_keys = surv_keys,
            long_keys = long_keys,
            family_name = family_name,
            parametrization = parametrization,
            y_lme = y_lme,
            b = b,
            d = d,
            p = p,
            log_time = log_time,
            log_st = log_st,
            w1 = w1,
            x_lme = x_lme,
            x_lme_time = x_lme_time,
            x_lme_time_deriv = x_lme_time_deriv,
            x_lme_s = x_lme_s,
            x_lme_s_deriv = x_lme_s_deriv,
            z_lme_b = z_lme_b,
            z_lme_time_b = z_lme_time_b,
            z_lme_time_b_deriv = z_lme_time_b_deriv,
            z_lme_s_b = z_lme_s_b,
            z_lme_s_b_deriv = z_lme_s_b_deriv,
            gammas = gammas,
            sigma_t = sigma_t,
            betas = betas,
            alpha = alpha,
            d_alpha = d_alpha,
            sigma = sigma,
            d_vc = d_vc,
            wint_f_vl = wint_f_vl,
            wint_f_sl = wint_f_sl,
            wsint_f_vl = wsint_f_vl,
            wsint_f_sl = wsint_f_sl,
            ind_fixed = ind_fixed,
            n = n,
            ncz = ncz,
            id_l = id_l,
            id_gk = id_gk,
            wk = wk
        )

        p_yt = np.matmul(p_y_tb, wgh) + tol4

    #Computation of individual likelihood through QMC estimation of integral over random effects
    if int_method == 'QMC':

        #Rescaling of random effects only if it is specified
        if v_dict['rescale']:

            #Empirical rescale of b distribution for each patient
            b_list, b_mean_list, b_cov_list, z_lme_b_list, z_lme_time_b_list, z_lme_time_b_deriv_list, z_lme_s_b_list, z_lme_s_b_deriv_list = Rescale_b_QMC(thetas, nbmc, v_dict)

            #Saving of current random effects to reuse them in estimation of derivate of log-likelihood
            v_dict['b'] = b_list
            v_dict['z_lme_b'] = z_lme_b_list
            v_dict['z_lme_time_b'] = z_lme_time_b_list
            v_dict['z_lme_time_b_deriv'] = z_lme_time_b_deriv_list
            v_dict['z_lme_s_b'] = z_lme_s_b_list
            v_dict['z_lme_s_b_deriv'] = z_lme_s_b_deriv_list
            v_dict['b_mean'] = b_mean_list
            v_dict['b_cov'] = b_cov_list
        
        #Use already scaled random effects if it is specified
        else:

            b_list = v_dict['b']
            z_lme_b_list = v_dict['z_lme_b']
            z_lme_time_b_list = v_dict['z_lme_time_b']
            z_lme_time_b_deriv_list = v_dict['z_lme_time_b_deriv']
            z_lme_s_b_list = v_dict['z_lme_s_b']
            z_lme_s_b_deriv_list = v_dict['z_lme_s_b_deriv']
            b_mean_list = v_dict['b_mean']
            b_cov_list = v_dict['b_cov']

        p_yt = np.array([])

        for i in range(n):

            #Recomputation of p(Ti, δi, yi, bi; θ) with patient-specific drawing of bi
            p_y_tb_i = Indiv_densities(
                surv_keys = surv_keys,
                long_keys = long_keys,
                family_name = family_name,
                parametrization = parametrization,
                y_lme = {key: y_lme[key][np.where(id_l[key]==i)] for key in long_keys},
                b = b_list[i],
                d = Dapply(d, lambda x:x[[i]], surv_keys),
                p = p[[i]],
                log_time = log_time[[i]],
                log_st = log_st[np.where(id_gk==i)],
                w1 = Dapply(w1, lambda x:x[[i]], surv_keys),
                x_lme = {key: x_lme[key][np.where(id_l[key]==i)] for key in long_keys},
                x_lme_time = Dapply(x_lme_time, lambda x:x[[i]] if x is not None else None, long_keys),
                x_lme_time_deriv = Dapply(x_lme_time_deriv, lambda x:x[[i]] if x is not None else None, long_keys),
                x_lme_s = {key: x_lme_s[key][np.where(id_gk==i)] if x_lme_s[key] is not None else None for key in long_keys},
                x_lme_s_deriv = {key: x_lme_s_deriv[key][np.where(id_gk==i)] if x_lme_s_deriv[key] is not None else None for key in long_keys},
                z_lme_b = z_lme_b_list[i],
                z_lme_time_b = z_lme_time_b_list[i],
                z_lme_time_b_deriv = z_lme_time_b_deriv_list[i],
                z_lme_s_b = z_lme_s_b_list[i],
                z_lme_s_b_deriv = z_lme_s_b_deriv_list[i],
                gammas = gammas,
                sigma_t = sigma_t,
                betas = betas,
                alpha = alpha,
                d_alpha = d_alpha,
                sigma = sigma,
                d_vc = d_vc,
                wint_f_vl = wint_f_vl[[i]],
                wint_f_sl = wint_f_sl[[i]],
                wsint_f_vl = wsint_f_vl[np.where(id_gk==i)],
                wsint_f_sl = wsint_f_sl[np.where(id_gk==i)],
                ind_fixed = ind_fixed,
                n = 1,
                ncz = ncz,
                id_l = {key: id_l[key][np.where(id_l[key]==i)] for key in long_keys},
                id_gk = id_gk[np.where(id_gk==i)],
                wk = wk[np.where(id_gk==i)]
            )

            #Probability density function of our random effects b
            b_pdf = multivariate_normal.pdf(x=b_list[i][:,np.arange(ncz)], mean=b_mean_list[i], cov=b_cov_list[i][:ncz, :ncz], allow_singular = True) + 1e-30

            p_yt = np.append(p_yt, np.matmul(p_y_tb_i, 1/b_pdf)/nbmc + tol4)

    #Log-liklihood to optimize
    log_p_yt = np.log(p_yt)

    #Value to minimize
    val = -log_p_yt[np.isfinite(log_p_yt)].sum()

    return(val)

def LgLik_weibull_grad(thetas, *args):
    """
        Function which computes the jacobian of log-likelihood function by central finite differentiation
    Args:
        thetas (np.array) : Vector of theta parameters (without fixed thetas which are inputed in *args)
        *args (dict): Dictionnary containing design matrices and parameters variables we need for computation of log-likelihood
    """

    #v_dict : Our dictionnary passed in *args
    v_dict = args[0]

    #n_cores for gradient computation
    n_cores = v_dict['n_cores_grad']

    #Avoiding rescaling when computing log-likelihood function inside this function
    v_dict['rescale'] = False

    if n_cores == 1:
        jac = approx_fprime(thetas, LgLik_weibull, 1.4901161193847656e-08, v_dict)
    else:
        n = len(thetas)
        ex = np.maximum(thetas, 1)
        f0 = LgLik_weibull(thetas, v_dict)

        jac = []

        #Warning, start_method 'fork' isn't available for windows
        with mpire.WorkerPool(n_jobs=n_cores, shared_objects=v_dict, start_method='fork') as pool:
            val = pool.map(
                #Shared objects are passed on as the second argument after the worker ID (worker ID insn't i)
                Diff_f_term,
                [
                    (
                        i,
                        thetas,
                        LgLik_weibull,
                        f0,
                        1.4901161193847656e-08,
                        ex
                    )
                    for i in range(n)
                ],
                progress_bar=True
            )
            jac = jac + val
        
        jac = np.array(jac)
        
        

    


    #Resetting Rescale option to True ( Avoiding rescaling comes only for specific case, like approximation of derivate of loglikelihood)
    v_dict['rescale'] = True

    return jac