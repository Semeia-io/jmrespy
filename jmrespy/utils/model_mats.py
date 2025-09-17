"""
Called in Survfit(), generate design matrices we need to estimate survival likelihoods
"""
####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import numpy as np
import pandas as pd

#Formulas
import patsy

# Quadratures
from .quadratures import Gauss_Kronrod




####################### -------------------- Model_mats function -------------------- #######################

def Model_mats(
    object_JM,
    data_id,
    time,
    ii
):
    #Control parameters from fitted JM
    con = object_JM.prop['control']

    #wk and sk : Gauss-Kronrod quadrature weights and nodes
    wk, sk = Gauss_Kronrod(con['gkk'])

    #p : time divided by 2
    p = time/2

    #st : Matrix i*j containing each of i values of p multiplied by each of j Gauss-Kronrod quadrature nodes + 1
    st = p * (sk+1)

    #out : dictionnary returned by function
    out = {
        'st': st,
        'wk': wk,
        'p': p,
        'x_lme_s': {},
        'z_lme_s': {},
        'x_lme_s_deriv': {},
        'z_lme_s_deriv': {}
    }

    id_gk = np.repeat(ii, con['gkk'])
    stime_lag_0 = np.maximum(st.flatten() - object_JM.lag, np.zeros(len(st.flatten()))) #Max beetween flatten st - lag and 0
    for key in data_id.keys():
        #data_id2 : Dataframe of flatten st associated with last longitudinal marquor collected for each group
        data_id2 = data_id[key].iloc[id_gk,:].reset_index(drop=True) #reset index to avoid duplicated index
        data_id2[object_JM.time_var[key]] = stime_lag_0

        #For curent marker
        if object_JM.parametrization[key] in ['value', 'both']:

            #Model matrixes for fixed and random effects from associated formulas
            design_fe_s = patsy.dmatrices(object_JM.lme_formula[key], data_id2)
            design_re_s = patsy.dmatrix(object_JM.lme_re_formula[key], data_id2)

            #x_lme_s : fixed effects design matrix in linear mixed model with stime_lag_0 replacing times of longitudinal marquors
            out['x_lme_s'][key] = np.array(design_fe_s[1])

            #z_lme_s : random effects design matrix in linear mixed model with stime_lag_0 replacing times of longitudinal marquors
            out['z_lme_s'][key] = np.array(design_re_s)

        #For derivated marker
        if object_JM.parametrization in ['slope', 'both']:

            #Model matrixes for fixed and random effects from associated formulas
            design_fe_s_deriv = patsy.dmatrix(object_JM.derivForm['fixed'][key], data_id2)
            design_re_s_deriv = patsy.dmatrix(object_JM.derivForm['random'][key], data_id2)

            #x_lme_s : fixed effects design matrix in linear mixed model with stime_lag_0 replacing times of longitudinal marquors
            out['x_lme_s_deriv'][key] = np.array(design_fe_s_deriv)

            #z_lme_s : random effects design matrix in linear mixed model with stime_lag_0 replacing times of longitudinal marquors
            out['z_lme_s_deriv'][key] = np.array(design_re_s_deriv)

    
    return(out)