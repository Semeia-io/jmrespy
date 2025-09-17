####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import numpy as np

# Probability
from scipy.stats import qmc, multivariate_normal

# Linalg
from scipy.linalg import eigh
from scipy.stats._multivariate import _eigvalsh_to_eps

#Warnings
import warnings

# Our modules
from .toolbox import *
from .indiv_densities import Indiv_densities
from .chol_trans import Chol_transf





####################### -------------------- Rescaling of random effects -------------------- #######################
def Rescale_b_QMC(thetas, nbmc, *args):
    """
        Function which computes random effects for QMC integration whith a better distribution for importance sampling technique. Returns also the drawn random effect and their matricial products whith random effect design matrices. 
    Args:
        thetas (np.array) : Vector of theta parameters (with fixed thetas)
        nbmc (integer) : Number of QMC points
        *args (dict): Dictionnary containing design matrices and parameters variables we need for computation of log-likelihood
    """

    ### Extract inputs from dict

    #v_dict : Our dictionnary passed in *args
    v_dict = args[0]

    # Dimension information
    n = v_dict['n']
    ncz = v_dict['ncz']
    ncz_dict = v_dict['ncz_dict']
    ind_random = v_dict['ind_random']
    ind_fixed = v_dict['ind_fixed']
    id_l = v_dict['id_l']
    id_gk = v_dict['id_gk']
    diag_d = v_dict['diag_d']

    # Gauss-Kronrod
    wk = v_dict['wk']

    # Keys
    long_keys = v_dict['long_keys']
    surv_keys = v_dict['surv_keys']

    # Response
    y_lme = v_dict['y_lme']
    d = v_dict['d']
    p = v_dict['p']
    log_time = v_dict['log_time']
    log_st = v_dict['log_st']

    # Fixed effect design matrices
    w1 = v_dict['w1']
    x_lme = v_dict['x_lme']
    x_lme_time = v_dict['x_lme_time']
    x_lme_time_deriv = v_dict['x_lme_time_deriv']
    x_lme_s = v_dict['x_lme_s']
    x_lme_s_deriv = v_dict['x_lme_s_deriv']

    # Random effect design matrices
    z_lme = v_dict['z_lme']
    z_lme_time = v_dict['z_lme_time']
    z_lme_time_deriv = v_dict['z_lme_time_deriv']
    z_lme_s = v_dict['z_lme_s']
    z_lme_s_deriv = v_dict['z_lme_s_deriv']

    # Thetas positions
    pos_betas = v_dict['pos_betas']
    pos_sigma = v_dict['pos_sigma']
    pos_d_vc = v_dict['pos_d_vc']
    pos_gammas = v_dict['pos_gammas']
    pos_alpha = v_dict['pos_alpha']
    pos_d_alpha = v_dict['pos_d_alpha']
    pos_sigma_t = v_dict['pos_sigma_t']

    # Weights
    wint_f_vl = v_dict['wint_f_vl']
    wint_f_sl = v_dict['wint_f_sl']
    wsint_f_vl = v_dict['wsint_f_vl']
    wsint_f_sl = v_dict['wsint_f_sl']

    # Other informations
    parametrization = v_dict['parametrization']
    family_name = v_dict['family_name']

    # Extract thetas
    extract_pos = lambda x, thetas_hat: thetas_hat[x] if x is not None else None
    extract_pos_exp = lambda x, thetas_hat: np.exp(thetas_hat[x]) if x is not None else None
    extract_pos_alpha = lambda x, thetas_hat: Dapply(x, extract_pos, long_keys, thetas_hat)
    betas = Dapply(pos_betas, extract_pos, long_keys, thetas)
    sigma = Dapply(pos_sigma, extract_pos_exp, long_keys, thetas)
    gammas = Dapply(pos_gammas, extract_pos, surv_keys, thetas)
    sigma_t = Dapply(pos_sigma_t, extract_pos_exp, surv_keys, thetas)
    alpha = Dapply(pos_alpha, extract_pos_alpha, surv_keys, thetas)
    d_alpha = Dapply(pos_d_alpha, extract_pos_alpha, surv_keys, thetas)
    d_vc = thetas[pos_d_vc]
    d_vc = np.exp(d_vc) if diag_d else Chol_transf(d_vc)['res'] #chol of d_vc : Vec ----> Matrix
    #d_vc = d_vc*np.identity(len(d_vc.diagonal()))

    #First simulation of bi for QMC integration following general distribution of b
    try:
        #cov multiplied by 4 to be sure to have high-densities zone of extreme patients in the first sampling
        dist = qmc.MultivariateNormalQMC(mean=np.zeros(ncz), cov=d_vc*4)
        b = dist.random(nbmc)
    #If qmc.MultivariateNormalQMC didn't succeed, this is beacause cov is singular
    except:
        #cov multiplied by 4 to be sure to have high-densities zone of extreme patients in the first sampling
        dist = multivariate_normal(mean=np.zeros(ncz), cov=d_vc*4, allow_singular=True)
        b = dist.rvs(size=nbmc)

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

    #Empirical rescale of b distribution for each patient
    b_list = []
    b_mean_list = []
    b_cov_list = []
    z_lme_b_list = np.array([])
    z_lme_time_b_list = np.array([])
    z_lme_time_b_deriv_list = np.array([])
    z_lme_s_b_list = np.array([])
    z_lme_s_b_deriv_list = np.array([])

    for i in range(p_y_tb.shape[0]):

        #Plot_screen(b, p_y_tb, i, '/Users/lucaschabeau/Documents/GitHub/Data_Corner/Projects/JM_python/Adaptation_Python/screen_rescale/first_sample/')
        #Position of density max observed in first sample
        pos_max_density = np.where(p_y_tb[i,:] == p_y_tb[i,:].max())

        #Case where we have multiple max pos
        if len(pos_max_density[0]) > 1:
            pos_max_density = pos_max_density[0][0]

        #Resampling centered at b corresponding to maximum densities at first sampling
        try:
            dist = qmc.MultivariateNormalQMC(mean=b[pos_max_density].squeeze(), cov=d_vc*2)
            b_centered = dist.random(nbmc)
        except:
            dist = multivariate_normal(mean=b[pos_max_density].squeeze(), cov=d_vc*2, allow_singular=True)
            b_centered = dist.rvs(size=nbmc)
        k = b_centered.shape[0]
        z_lme_b, z_lme_time_b, z_lme_time_b_deriv, z_lme_s_b, z_lme_s_b_deriv = Z_b(
            long_keys = long_keys,
            ncz_dict = ncz_dict,
            z_lme = {key: z_lme[key][np.where(id_l[key]==i)] for key in long_keys},
            z_lme_time = {key: z_lme_time[key][i] if z_lme_time[key] is not None else None for key in long_keys},
            z_lme_s = {key: z_lme_s[key][np.where(id_gk==i)] if z_lme_s[key] is not None else None for key in long_keys},
            z_lme_s_deriv = {key: z_lme_s_deriv[key][np.where(id_gk==i)] if z_lme_s_deriv[key] is not None else None for key in long_keys},
            z_lme_time_deriv = {key: z_lme_time_deriv[key][i] if z_lme_time_deriv[key] is not None else None for key in long_keys},
            b = b_centered,
            k = k,
            ind_random = ind_random,
            parametrization = parametrization
        )

        p_y_tb_i = Indiv_densities(
            surv_keys = surv_keys,
            long_keys = long_keys,
            family_name = family_name,
            parametrization = parametrization,
            y_lme = {key: y_lme[key][np.where(id_l[key]==i)] for key in long_keys},
            b = b_centered,
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

        #Estimation of patient-specific mu of its random effects distribution N(mu, sigma2) by empirical ponderate mean (ponderated by p(Ti, δi, yi, bi; θ) observed on first drawing)
        try:
            #b_mean = np.average(b, axis=0, weights=p_y_tb[i,:])
            b_mean = np.average(b_centered, axis=0, weights=p_y_tb_i.squeeze())
        except:
            #b_mean = np.average(b, axis=0, weights=p_y_tb[i,:]+1e100)
            b_mean = np.average(b_centered, axis=0, weights=p_y_tb_i.squeeze()+1e100)

        #Estimation of patient-specific sigma2 of its random effects distribution N(mu, sigma2) by empirical ponderate covariance (ponderated by p(Ti, δi, yi, bi; θ) observed on first drawing)
        #b_cov multiplied by 2 to also take some information around high interest zone empiricaly defined
        singular_cov = True
        count_singular = 0
        while singular_cov:
            scale_cov = 2
            try:
                b_cov = np.cov(b_centered, rowvar=0, aweights=p_y_tb_i.squeeze()) * scale_cov
                if not np.isfinite(b_cov).all():
                    b_cov = np.cov(b_centered, rowvar=0, aweights=p_y_tb_i.squeeze()+1e100) * scale_cov
            except:
                b_cov = np.cov(b_centered, rowvar=0, aweights=p_y_tb_i.squeeze()+1e100) * scale_cov

            #If b_cov isn't PSD, we fix this issue brought by numerical approximation, by increasing diagonal with a small delta
            eigvals_b_cov = eigh(b_cov, eigvals_only=True)
            delta_eps = 0 #Count of epsilon added to diagonal to solve PSD matrix estimation
            while np.min(eigvals_b_cov) < -_eigvalsh_to_eps(eigvals_b_cov):
                delta_eps += 1
                b_cov = b_cov + np.eye(ncz) * np.finfo(b_cov.dtype).eps
                warnings.warn("Warning...... Estimated cov matrix of b for subject {} wasn't PSD, {} was added to cov diagonal".format(i, delta_eps*np.finfo(b_cov.dtype).eps))
                eigvals_b_cov = eigh(b_cov, eigvals_only=True)
                
            singular_cov = np.linalg.det(b_cov) == 0
            count_singular += 1
            if count_singular > 5:
                #Stopping regenerated cov, singular cov matrix will be allowed
                break

        #Resampling of random effects to catch high-density zone of each patient
        try:
            dist = qmc.MultivariateNormalQMC(mean=b_mean, cov=b_cov)
            b_scaled =  dist.random(nbmc)
        except:
            dist = multivariate_normal(mean=b_mean, cov=b_cov, allow_singular=True)
            b_scaled = dist.rvs(size=nbmc)

        k = b_scaled.shape[0]
        z_lme_b, z_lme_time_b, z_lme_time_b_deriv, z_lme_s_b, z_lme_s_b_deriv = Z_b(
            long_keys = long_keys,
            ncz_dict = ncz_dict,
            z_lme = {key: z_lme[key][np.where(id_l[key]==i)] for key in long_keys},
            z_lme_time = {key: z_lme_time[key][i] if z_lme_time[key] is not None else None for key in long_keys},
            z_lme_s = {key: z_lme_s[key][np.where(id_gk==i)] if z_lme_s[key] is not None else None for key in long_keys},
            z_lme_s_deriv = {key: z_lme_s_deriv[key][np.where(id_gk==i)] if z_lme_s_deriv[key] is not None else None for key in long_keys},
            z_lme_time_deriv = {key: z_lme_time_deriv[key][i] if z_lme_time_deriv[key] is not None else None for key in long_keys},
            b = b_scaled,
            k = k,
            ind_random = ind_random,
            parametrization = parametrization
        )

        b_list.append(b_scaled)
        b_mean_list.append(b_mean)
        b_cov_list.append(b_cov)
        z_lme_b_list = np.append(z_lme_b_list, z_lme_b)
        z_lme_time_b_list = np.append(z_lme_time_b_list, z_lme_time_b)
        z_lme_time_b_deriv_list = np.append(z_lme_time_b_deriv_list, z_lme_time_b_deriv)
        z_lme_s_b_list = np.append(z_lme_s_b_list, z_lme_s_b)
        z_lme_s_b_deriv_list = np.append(z_lme_s_b_deriv_list, z_lme_s_b_deriv)

    return b_list, b_mean_list, b_cov_list, z_lme_b_list, z_lme_time_b_list, z_lme_time_b_deriv_list, z_lme_s_b_list, z_lme_s_b_deriv_list