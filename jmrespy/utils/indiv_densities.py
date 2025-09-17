# Usual libraries
import numpy as np

# Probability
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.special import expit

# Toolbox
from .toolbox import Group_by_sum

def Indiv_densities(
        surv_keys,
        long_keys,
        family_name,
        parametrization,
        y_lme,
        b,
        d,
        p,
        log_time,
        log_st,
        w1,
        x_lme,
        x_lme_time,
        x_lme_time_deriv,
        x_lme_s,
        x_lme_s_deriv,
        z_lme_b,
        z_lme_time_b,
        z_lme_time_b_deriv,
        z_lme_s_b,
        z_lme_s_b_deriv,
        gammas,
        sigma_t,
        betas,
        alpha,
        d_alpha,
        sigma,
        d_vc,
        wint_f_vl,
        wint_f_sl,
        wsint_f_vl,
        wsint_f_sl,
        ind_fixed,
        n,
        ncz,
        id_l,
        id_gk,
        wk
    ):
    """
        Function which computes for each observation of sample, the densities p(Ti, δi, yi, bi; θ) for a set of random effects bi.
        The fonction return a matrix of dimension (n_indiv, n_bi), each line corresponds to densities for a patient i and his set of bi for which we wants to evaluate this density
    Args:
        surv_keys (list): List of labels of competing risks
        long_keys (list): List of labels of longitudinal markers
        family_name (dict): Dictionnary containing for each longitudinal marker, the nature of longitudinal marker
        parametrization (dict): Dictionnary containing for each longitudinal marker, the nature link with time-to-event
            * 'value' : alpha * Estimated value of longitudinal marker at a time t
            * 'slope' : alpha * Estimated slope of longitudinal marker at a time t
            * 'both' : alpha1 * Estimated value of longitudinal marker at a time t + alpha2 * Estimated slope of longitudinal marker at a time t
        y_lme (dict): List of longitudinal response observed over follow-up
        b (numpy.array): Array of random effects for which individual density have to be computed
        d (dict): Indicator of event or censoring
        p (dict): Time to event / 2
        log_time (dict): Log(Time to event)
        log_st (dict): Log of matricial product between p and Gauss-Kronrod quadrature nodes
        w1 (dict): Dictionnary containing for each competing risk, the design matrix of baseline covariates (including the 1 column for intercept)
        x_lme (dict): Dictionnary containing for each longitudinal marker, the fixed effect design matrix of longitudinal marker (including the 1 column for intercept)
        x_lme_time (dict): Dictionnary containing for each longitudinal marker, the fixed effect design matrix of longitudinal marker (including the 1 column for intercept) at time-to-event
        x_lme_time_deriv (dict): Design matrix we need to compute derivation of matricial product between x_lme_time and Beta over time
        x_lme_s (dict): Same as x_lme but with matricial product between p and Gauss-Kronrod quadrature nodes replacing time covariate of design matrix
        x_lme_s_deriv (dict): Same as x_lme_deriv but with matricial product between p and Gauss-Kronrod quadrature nodes replacing time covariate of design matrix
        z_lme_b (dict): Matricial product between random effect design matrix of longitudinal marker (for each longitudinal marker) and associated b
        z_lme_time_b (dict): Matricial product between equivalent of x_lme_time for random effects and associated b
        z_lme_time_b_deriv (dict): Matricial product between equivalent of x_lme_time_deriv for random effects and associated b
        z_lme_s_b (dict): Matricial product between equivalent of x_lme_s for random effects and associated b
        z_lme_s_b_deriv (dict): Matricial product between equivalent of x_lme_s_deriv for random effects and associated b
        gammas (dict): Dictionnary containing for each competing risk, the coefficients associated to baseline covariates for a Time-to-event Ti assumed distributed following a Weibull law
        sigma_t (dict): Shape value of Weibull distribution
        betas (dict): Dictionnary containing for each longitudinal marker, the coefficients associated to longitudinal covariates
        alpha (dict): Coefficients corresponding to link between longitudinal responses and time-to-event (Must be None parametrization = 'slope')
        d_alpha (dict): Coefficients corresponding to link between estimated slope of longitudinal responses and time-to-event (Must be None parametrization = 'value')
        sigma (dict): Estimation of variance of longitudinal response (Must be None if longitudinal response is distributed following a Binomial distribution)
        d_vc (dict): Estimation of Variance/Covariance matrix of random effects
        wint_f_vl (dict): Weights applied to alpha for each patient
        wint_f_sl (dict): Weights applied to d_alpha for each patient
        wsint_f_vl (dict): Same as wint_f_vl but with dimension corresponding x_lme_s number of observations
        wsint_f_sl (dict): Same as wint_f_sl but with dimension corresponding x_lme_s number of observations
        ind_fixed (dict): Indication for each longitudinal response, of coefficients associated with time
        n (dict): Number of patients
        ncz (int): Random effect dimensionality
        id_l (dict): Id of patients over longitudinal response
        id_gk (numpy.array): Id of patients over matricial product between longitudinal response and Gauss-Kronrod nodes
        wk (numpy.array): Gauss-Kronrod weights associated with each value of matricial product between longitudinal response and Gauss-Kronrod nodes
        
    """

    #Linear predictors
    eta_yx = {} #Estimated y or logit(p(y=1)) by fixed effects of lme : eta_yx = beta0 + beta1*x1 + beta2*x2 + ... + betap*xp
    eta_tw = {risk : np.matmul(w1[risk], gammas[risk])[np.newaxis].T for risk in surv_keys} #Estimated value of exponential part of proportional risks model : h(t) = h(0) * exp(eta_tw) without link with longitundinal markor
    eta_t = eta_tw.copy() #Estimated eta including intercept (from Weibull) * 1 and g(b, alpha, t) the link between longitudinal markor and survival
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
            #log_norm = np.log(norm.pdf(y_lme[key], loc=np.array(mu_y), scale=sigma[key])+10e-30) #log likelihood of our y_lme following N(mu = mu_y, sigma = sigma) law
            log_norm = norm.logpdf(y_lme[key], loc=np.array(mu_y), scale=sigma[key])+10e-30 #log likelihood of our y_lme following N(mu = mu_y, sigma = sigma) law
            log_p_y_b = log_p_y_b + Group_by_sum(log_norm, id_l[key], axis=0) #Sum of log_norm likelihoods by group : log p(yi | bi;θy)
        elif family_name[key] == 'bernoulli':
            logit_p_y = eta_yx[key] + z_lme_b[key] #Estimated logit(p(y=1)) by lme (fixed effects and random effects)
            log_p_y = np.log((expit(logit_p_y)**y_lme[key]) * ((1-expit(logit_p_y))**(1-y_lme[key])) + 10e-30) #log_probability of observing real y with our model 
            log_p_y_b = log_p_y_b + Group_by_sum(log_p_y, id_l[key], axis=0) #Sum of log_probability by group : log p(yi | bi;θy)
    log_hazard = np.array([(np.log(sigma_t[risk]) + (sigma_t[risk] - 1) * log_time + eta_t[risk]) * d[risk] for risk in surv_keys]).sum(axis=0) #Instantaneous risk * d
    log_survival = np.nan_to_num(np.array([-np.exp(eta_tw[risk]) * p * Group_by_sum(wk * np.exp(np.log(sigma_t[risk]) + (sigma_t[risk] - 1) * log_st + eta_s[risk]), id_gk, axis=0) for risk in surv_keys]).sum(axis=0)) #Survival function. For some extreme cases, can conatains -inf or nan, respectively replaced by lowest possible value and 0 by nan_to_num
    log_p_tb = log_hazard + log_survival #Log likelihood of survival part in joint model : log p(Ti,δi | bi;θt,β)
    log_p_b = np.repeat(np.log(multivariate_normal.pdf(x=b[:,np.arange(ncz)], mean=np.repeat(0, ncz), cov=d_vc[:ncz, :ncz], allow_singular=True) + 10e-30), n) #Log likelihood of random effects : log p(bi | Vech(D))
    #log_p_b = np.repeat(Dmvnorm(x=b, mu=np.repeat(0, ncz), varcov=d_vc, log=True), n) #Log likelihood of random effects : log p(bi | Vech(D))
    p_y_tb = np.exp(log_p_y_b + log_p_tb + log_p_b.reshape((log_p_y_b.shape[1], log_p_y_b.shape[0])).T) #p(Ti, δi, yi, bi; θ)
    return p_y_tb