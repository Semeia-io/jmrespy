import numpy as np
from scipy.integrate import quad_vec
from scipy.special import expit

def CIF(t, u, w, gammas, sigma_t, i, surv_keys, risk, g_args):
    """Compute Cumulative incidence function of event of interest between times t and u, works with or without competing envents

    Args:
        t (np.float): Time of begining for the CIF window
        u (np.array): Times of ending for the CIF window
        w (dict): containing Design matrix of baseline data for each competing event
        gammas (dict): Containing coefficients vectors associated to baseline design matrix, for each competing event (by a matrix product in a cause-specific Cox proportionnal hazard model)
        sigma_t (dict): Containing sigma_t parameter of Weibull distribution for each competing event
        i (scalar): indicator of surv_time for wich CIF is computed. IF surv_times contains only one time, i = 0
        surv_keys (np.array): Keys correspondig to each competing events, this keys are used to extract elements in all dictionnaries of this function
        risk (str): Event of interest for which cIF is computed (must be one of the surv_keys)
        g_args (tuple): Containing :
            * 1st element : betas coefficients with subject random effects added
            * 2nd elelment : self.parametrization, the structure of the association between time-to) event and longitudinal data
            * 3rd element : Coefficient associated to the current level of longitudinal marker
            * 4th : Coefficient associated to the slope of longitudinal marker
            * 5th element : ind_fixed, Pos of betas associated with time, works only for derivation of mi(t) = B0 + b0i + (B1 + b1i) * t
            * 6th element : family_long, the type of longitudinal biomarkers ('gaussian' or 'binary')

    Returns:
        np.array: CIF
    """

    #eta_tw = lambda*exp(gammas*w) for cause specific event to predict
    eta_tw = np.matmul(w[risk][i,], gammas[risk])

    args_integrand = (w, gammas, sigma_t, i, surv_keys, risk, g_args)

    #Cumulative incidence function
    cif = np.array([np.exp(eta_tw) * sigma_t[risk] * quad_vec(Integrand_CIF, t, u_scalar, quadrature='gk15', norm='2', limit=1, args=args_integrand)[0] for u_scalar in u]).flatten()

    return cif

def Integrand_CIF(t, *args):
    """Fonction to integrate when computing CIF

    Args:
        t (np.float): Time for computation of integrand of CIF (h(t)*S(t))

    Returns:
        scalar: Integrand of CIF at time t
    """

    #Extract *args
    w, gammas, sigma_t, i, surv_keys, risk, g_args = args

    betas_b, parametrization, alpha_value, alpha_slope, ind_fixed, family_long = g_args
    exp_g = np.exp(g(t,risk, betas_b, parametrization, alpha_value, alpha_slope, ind_fixed, family_long))
    s_t = Survival(t, w, gammas, sigma_t, i, surv_keys, g_args)

    integrand = t**(sigma_t[risk]-1) * exp_g * s_t

    return integrand

def Survival(
        t,
        w,
        gammas,
        sigma_t,
        i,
        surv_keys,
        g_args
    ):
    """Function computing overall survival at time t (whatever is the event of interest, this is the probability to be all-events free at time t)

    Args:
        t (np.float): Time for computation of overall
        w (dict): containing Design matrix of baseline data for each competing event
        gammas (dict): Containing coefficients vectors associated to baseline design matrix, for each competing event (by a matrix product in a cause-specific Cox proportionnal hazard model)
        sigma_t (dict): Containing sigma_t parameter of Weibull distribution for each competing event
        i (scalar): indicator of surv_time for wich Survival is computed. IF surv_times contains only one time, i = 0
        surv_keys (np.array): Keys correspondig to each competing events, this keys are used to extract elements in all dictionnaries of this function
        g_args (tuple): Containing :
            * 1st element : betas coefficients with subject random effects added
            * 2nd elelment : self.parametrization, the structure of the association between time-to) event and longitudinal data
            * 3rd element : Coefficient associated to the current level of longitudinal marker
            * 4th : Coefficient associated to the slope of longitudinal marker
            * 5th element : ind_fixed, Pos of betas associated with time, works only for derivation of mi(t) = B0 + b0i + (B1 + b1i) * t
            * 6th element : family_long, the type of longitudinal biomarkers ('gaussian' or 'binary')

    Returns:
        np.array: Probability of beeing event-free at time t
    """

    #Overall cumulative risk
    cr = Cumulative_risk(t, w, gammas, sigma_t, i, surv_keys, g_args)

    #Overall survival function
    S_t = np.exp(-cr)

    return S_t

def Cumulative_risk(
        t,
        w,
        gammas,
        sigma_t,
        i,
        surv_keys,
        g_args
    ):
    """Function computing overall cumulative risk of events at time t

    Args:
        t (np.float): Time for computation of overall
        w (dict): containing Design matrix of baseline data for each competing event
        gammas (dict): Containing coefficients vectors associated to baseline design matrix, for each competing event (by a matrix product in a cause-specific Cox proportionnal hazard model)
        sigma_t (dict): Containing sigma_t parameter of Weibull distribution for each competing event
        i (scalar): indicator of surv_time for wich Survival is computed. IF surv_times contains only one time, i = 0
        surv_keys (np.array): Keys correspondig to each competing events, this keys are used to extract elements in all dictionnaries of this function
        g_args (tuple): Containing :
            * 1st element : betas coefficients with subject random effects added
            * 2nd elelment : self.parametrization, the structure of the association between time-to) event and longitudinal data
            * 3rd element : Coefficient associated to the current level of longitudinal marker
            * 4th : Coefficient associated to the slope of longitudinal marker
            * 5th element : ind_fixed, Pos of betas associated with time, works only for derivation of mi(t) = B0 + b0i + (B1 + b1i) * t
            * 6th element : family_long, the type of longitudinal biomarkers ('gaussian' or 'binary')

    Returns:
        np.array: Overall cumulative risk of events at time t
    """

    if t == 0:
        cr = 0
    else:
        #eta_tw = lambda*exp(gammas*w)
        eta_tw = {risk : np.matmul(w[risk][i,], gammas[risk]) for risk in surv_keys}

        #Overall cumulative risk computaion cr = integral(hazard_1 + ... + hazard_k) for k competing risks
        cr = np.array([np.exp(eta_tw[risk]) * sigma_t[risk] * quad_vec(Integrand_CR, 0, t, quadrature='gk15', norm='2', limit=1, args=(sigma_t, risk) + g_args)[0] for risk in surv_keys]).sum()
    
    return cr

def Integrand_CR(t, *args):
    """
        Compute remaining integrand for cause-specific cumulative hazard. Must be multiplied by elements out of integral to compute cumulative hazard
    Args:
        t (np.float): time
    """

    #Extract *args
    sigma_t, risk, betas_b, parametrization, alpha_value, alpha_slope, ind_fixed, family_long = args

    #Function to integrate cumulative hazards (hazard ratio excluding non time dependant terms)
    integrand = t**(sigma_t[risk]-1) * np.exp(g(t,risk, betas_b, parametrization, alpha_value, alpha_slope, ind_fixed, family_long))

    return integrand

def g(t, risk, betas_b, parametrization, alpha_value, alpha_slope, ind_fixed, family_long):
    """Function computing association between longitudinal biomarker and time-to-event

    Args:
        t (np.float): time
        risk ('str'): Key of event
        betas_b (dict): betas coefficients with subject random effects added
        parametrization (dict): self.parametrization, the structure of the association between time-to) event and longitudinal data
        alpha_value (dict): Coefficient associated to the current level of longitudinal marker
        alpha_slope (dict): Coefficient associated to the slope of longitudinal marker
        ind_fixed (int): ind_fixed, Pos of betas associated with time, works only for derivation of mi(t) = B0 + b0i + (B1 + b1i) * t
        family_long (dict): family_long, the type of longitudinal biomarkers ('gaussian' or 'binary')

    Returns:
        np.float:
            * If parametrization == 'value' : mi(t) = alpha_value * (B0 + b0i + (B1 + b1i) * t)
            * If parametrization == 'slope' : mi'(t) = alpha_slope * (B1 + b1i)
            * If parametrization == 'both' : mi(t) + mi'(t) = alpha_value * (B0 + b0i + (B1 + b1i) * t) + alpha_slope * (B1 + b1i)
    """
    g_out = 0
    for key in betas_b.keys():
        if parametrization[key] in ['value', 'both']:
            mu_t = np.matmul(np.array([1, t]), betas_b[key]) if family_long[key] == 'gaussian' else expit(np.matmul(np.array([1, t]), betas_b[key]))
            g_out += alpha_value[risk][key] * mu_t
        if parametrization[key] in ['slope', 'both']:
            g_out += alpha_slope[risk][key] * betas_b[key][ind_fixed]
    return g_out