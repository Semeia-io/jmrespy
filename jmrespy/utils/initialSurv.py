"""
Function to set initial survival values
"""

####################### -------------------- Import libraries and modules -------------------- #######################

#Usual libraries
import pandas as pd
import numpy as np
import sys

#lifelines utils
from lifelines.utils import add_covariate_to_timeline
from lifelines.utils import to_long_format

# Models libraries
from lifelines.fitters.cox_time_varying_fitter import CoxTimeVaryingFitter
from lifelines import WeibullAFTFitter

def Initial_surv(
    time_s,
    d,
    w,
    id_l,
    time_l,
    method,
    parametrization,
    long_cur = None,
    long_deriv = None,
):

    #Extract longitudinal keys (we can find them in all longitudinal dictionnaries, they are the same keys):
    long_keys = list(parametrization.keys())

    #Last longitudinal marker for each group
    long_id = {key: pd.DataFrame(long_cur[key]).groupby(id_l[key]).last() if long_cur[key] is not None else None for key in long_keys}

    #Last derived longitudinal marker for each group
    long_d_id = {key: pd.DataFrame(long_deriv[key]).groupby(id_l[key]).last() if long_deriv[key] is not None else None for key in long_keys}    
    
    #Complete survival design matrix with last longitudinal marker of each group
    ww = w.copy()
    for key in long_keys:
        if long_id[key] is not None:
            ww['{}_cur'.format(key)] = long_id[key]
        if long_d_id[key] is not None:
            ww['{}_deriv'.format(key)] = long_d_id[key]

    ###Initialisation of survival parameters with cox time variying fitter and baseline hazard parameters with specialized method ---
    if method in ['Cox-PH-GH', 'weibull-PH-GH', 'weibull-PH-QMC', 'piecewise-PH-GH', 'spline-PH-GH', 'spline-PH-Laplace']:
        
        #dlf : Design matrix in long format compatible with Cox time variying fitter
        dd = w.copy()
        dd['id'] = dd.index
        dd['time_s'] = time_s
        dd['d'] = d
        dlf = to_long_format(dd, duration_col='time_s')

        #Adding longitudinal estimations of markors in dlf
        for key in long_keys:
            if long_cur[key] is not None:
                cv = pd.DataFrame({'id': id_l[key], 'time_l': time_l[key], '{}_cur'.format(key): long_cur[key]})
                dlf = add_covariate_to_timeline(dlf, cv, duration_col='time_l', id_col='id', event_col='d')
            if long_deriv[key] is not None:
                cv = pd.DataFrame({'id': id_l[key], 'time_l': time_l[key], '{}_deriv'.format(key): long_deriv[key]})   
                dlf = add_covariate_to_timeline(dlf, cv, duration_col='time_l', id_col='id', event_col='d')

        #Fill Nas with next valid observation of markor
        dlf = dlf.fillna(method='bfill')

        #Cox Time Varying fitter
        col_surv = dlf.columns[dlf.columns != 'id']
        try:
            ctv = CoxTimeVaryingFitter()
            ctv.fit(dlf[col_surv], start_col='start', stop_col='stop', event_col='d')
        except:
            ctv = CoxTimeVaryingFitter(penalizer=0.001)
            ctv.fit(dlf[col_surv], start_col='start', stop_col='stop', event_col='d')
        coefs = ctv.params_

        #out : Dictionnary of initial parameters returned by Initial_surv()
        out = {}

        #gammas_init : Estimated baseline coefficients
        gammas_init = coefs[list(w.columns)]

        #alpha_init : Estimated links between curents longitudinal markors and survival
        alpha_init = {key: coefs['{}_cur'.format(key)] if long_cur[key] is not None else None for key in long_keys}
        out['alpha'] = alpha_init

        #d_alpha_init : Estimated links between derivated longitudinal markors and survival
        d_alpha_init = {key: coefs['{}_deriv'.format(key)] if long_deriv[key] is not None else None for key in long_keys}
        out['d_alpha'] = d_alpha_init
        
        ###Weibull-PH-GH ---
        if method in ['weibull-PH-GH', 'weibull-PH-QMC']:
            ww['time_s'] = time_s
            ww['d'] = d

            #Accelerated failure time fitter to estimates baseline hazard parameters
            init_fit = WeibullAFTFitter()
            init_fit.fit(ww, duration_col='time_s', event_col='d')

            #sigma_t : Estimated baseline hazard shape
            sigma_t = np.exp(init_fit.params_['rho_'])[0]
            out['sigma_t'] = sigma_t

            #gammas : Vector of estimated baseline hazard scale and estimated baseline covariates coefficients
            coefs_aft = - init_fit.params_['lambda_'] * sigma_t
            out_gammas = np.concatenate((np.array([coefs_aft['Intercept']]), np.array(gammas_init)), axis=0)
            out['gammas'] = out_gammas

            return(out)

        else:
            print('Only handle weibull-PH-GH method for instant, new methods will come along nexts updates')
            sys.exit()

    else:
        print("method must be in ['Cox-PH-GH', 'weibull-PH-GH', 'weibull-PH-QMC', 'piecewise-PH-GH', 'spline-PH-GH', 'spline-PH-Laplace']")
        sys.exit()