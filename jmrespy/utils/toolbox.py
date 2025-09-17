"""
This file contains all functions we could use to simplify the code
"""
####################### -------------------- Modules and libraries -------------------- #######################
from itertools import product
import pandas as pd
import numpy as np

#Deepcopy
from copy import deepcopy




####################### -------------------- Alone_to_dict -------------------- #######################
"""
Find if an object is a dictionnary and put in inside a dictionnary if this is not the case. Usefull to generalize code for competing risks and multiple markors
"""
def Alone_to_dict(obj, key):
    if not isinstance(obj, dict):
        return {key: obj}
    else:
        return obj





####################### -------------------- Alone_dict_to_dict -------------------- #######################
"""
Find if a dictionnary contains another dictionnaries and put all items inside one dictionnary in the dictionnary if this is not the case. Usefull to generalize code for competing risks and multiple markors
"""
def Alone_dict_to_dict(obj, key):
    if not any(isinstance(dict_val, dict) for dict_val in obj.values()):
        return {key: obj}
    else:
        return obj





####################### -------------------- Alone_slice_dict_to_dict -------------------- #######################
"""
Find if a slice of a dictionnary contains another dictionnaries and put all items inside one dictionnary in the dictionnary if this is not the case. Usefull to generalize code for competing risks and multiple markors, only for init parameter
"""
def Alone_slice_dict_to_dict(obj, slice, key):
    if len(slice) > 0:
        slice_dict = Alone_dict_to_dict({key: obj[key] for key in slice}, key)
    else:
        slice_dict = {}
    return slice_dict





####################### -------------------- Dapply -------------------- #######################
"""
Apply a function to all or part of a dictionnary
"""
def Dapply(obj, f, keys, *args):
    out_dict = {key: f(obj[key], *args) for key in keys}
    return out_dict





####################### -------------------- Dapply_method -------------------- #######################
"""
Apply a method to all or part of a dictionnary
"""
def Dapply_method(obj, method, keys):
    out_dict = {key: getattr(obj[key], method) for key in keys}
    return out_dict





####################### -------------------- Expandgrid -------------------- #######################
"""
Adaptation of expand.grid R's function for python
"""
def Expandgrid(dictionary):
   return pd.DataFrame([row for row in product(*dictionary.values())], columns=dictionary.keys())





####################### -------------------- Dict_skel -------------------- #######################
"""
This function allows to keep a structure in our dictionnaries transformed in lists.
This function converts one or many dictionnaries in one flatten list and save the structure of these dictionaries in another dictionary containing
Position of each element of the dictionary in the flatten list.

In case of nested dictionnary, this function works only for an input dict of depth 3

This function is used to pass dictionaries of model parameters in scipy.optimize.optim
"""
def Dict_skel(input_dict):

    pos = 0
    thetas = [] # List of thetas return by function
    pos_thetas = [] # Position of each theta in thetas

    # Filling thetas ans pos_thetas
    for input_key in input_dict.keys():
        # If a theta is a dict
        if isinstance(input_dict[input_key], dict):
            pos_theta = {}
            for key in input_dict[input_key].keys():
                #If a theta is a dict containing a dict
                if isinstance(input_dict[input_key][key], dict):
                    pos_theta[key] = {}
                    for dict_key in input_dict[input_key][key].keys():
                        pos_theta[key][dict_key] = []
                        if input_dict[input_key][key][dict_key] is not None:
                            for i in range(len(np.array(input_dict[input_key][key][dict_key]).flatten())):
                                thetas.append(np.array(input_dict[input_key][key][dict_key]).flatten()[i])
                                pos_theta[key][dict_key].append(pos)
                                pos = pos+1
                        else:
                            pos_theta[key][dict_key] = None
                #If a theta is a dict not containing a dict
                else:
                    pos_theta[key] = []
                    if input_dict[input_key][key] is not None:
                        for i in range(len(np.array(input_dict[input_key][key]).flatten())):
                            thetas.append(np.array(input_dict[input_key][key]).flatten()[i])
                            pos_theta[key].append(pos)
                            pos = pos+1
                    else:
                        pos_theta[key] = None
            pos_thetas.append(pos_theta)
        #If a theta is not a dict
        else:
            pos_theta = []
            if input_dict[input_key] is not None:
                for i in range(len(np.array(input_dict[input_key]).flatten())):
                    thetas.append(np.array(input_dict[input_key]).flatten()[i])
                    pos_theta.append(pos)
                    pos = pos+1
            else:
                pos_theta = None
            pos_thetas.append(pos_theta)
    return(tuple([np.array(thetas)] + pos_thetas))




####################### -------------------- Z_b -------------------- #######################
"""
This function computes all matricial products we need between different random design matrices (Z) and random effects (b)
This function become relevant since we have multiple ways to compute random effects (b) with invariant design matrices (Z)
The objective of this function is to simplify the code, avoiding the same code multiple times in different part of our library
"""
def Z_b(
    long_keys,
    ncz_dict,
    z_lme,
    z_lme_time,
    z_lme_s,
    z_lme_s_deriv,
    z_lme_time_deriv,
    b,
    k,
    ind_random,
    parametrization,
    id_l = None,
    id_gk = None
    ):
    z_lme_b = {} #z_lme matricialy multiplied by transposed b
    z_lme_time_b = {} #z_lme_time matricialy multiplied by transposed b
    z_lme_time_b_deriv = {}
    z_lme_s_b = {} #z_lme_s matricialy multiplied by transposed b
    z_lme_s_b_deriv = {}
    i = 0
    
    #Computing all matricial products of Z with b for quadrature
    for key in long_keys:

        ncz_key = ncz_dict[key]
        i_end = i+ncz_key

        #If b inputed insn't a numpy array of 2 dimensions but an array of 3 dimensions, that means user chosen to draw patient-specific b
        global_b_draw = b.ndim == 2

        z_lme_b[key] = np.matmul(z_lme[key], b[:, i:i_end].T) if global_b_draw else np.matmul(np.expand_dims(z_lme[key], axis=1), np.transpose(b[id_l[key], :, i:i_end], axes=(0,2,1))).squeeze()

        if parametrization[key] in ['value', 'both']:
            z_lme_time_b[key] = np.matmul(z_lme_time[key], b[:, i:i_end].T) if global_b_draw else np.matmul(np.expand_dims(z_lme_time[key], axis=1), np.transpose(b[:, :, i:i_end], axes=(0,2,1))).squeeze()
            z_lme_s_b[key] = np.matmul(z_lme_s[key], b[:, i:i_end].T) if global_b_draw else np.matmul(np.expand_dims(z_lme_s[key], axis=1), np.transpose(b[:, :, i:i_end][id_gk], axes=(0,2,1))).squeeze()
        else:
            z_lme_time_b[key] = None
            z_lme_s_b[key] = None
        
        if parametrization[key] in ['slope', 'both']:
            if ind_random[key] is not None:
                z_lme_time_b_deriv[key] = np.matmul(z_lme_time_deriv[key], b[:,i + ind_random[key]].T) if global_b_draw else np.matmul(np.expand_dims(z_lme_time_deriv[key], axis=1), np.transpose(b[:, :, i + ind_random[key]], axes=(0,2,1))).squeeze()
                z_lme_s_b_deriv[key] = np.matmul(z_lme_s_deriv[key], b[:,i + ind_random[key]].T) if global_b_draw else np.matmul(np.expand_dims(z_lme_s_deriv[key], axis=1), np.transpose(b[:, :, i + ind_random[key]][id_gk], axes=(0,2,1))).squeeze()
            else:
                z_lme_time_b_deriv[key] = np.zeros((z_lme_time_deriv[key].shape[0], k))
                z_lme_s_b_deriv[key] = np.zeros((z_lme_s_deriv[key].shape[0], k))

        else:
            z_lme_time_b_deriv[key] = None
            z_lme_s_b_deriv[key] = None

        i = i_end
    
    return z_lme_b, z_lme_time_b, z_lme_time_b_deriv, z_lme_s_b, z_lme_s_b_deriv





####################### -------------------- Init_check_type -------------------- #######################
"""
Verify if parameters in init dictionnary have expected types, raise an error if this isn't the case
"""
def Init_check_type(obj, slice):
    for k in obj.keys():
        for init_input in set(slice.keys()).intersection(set(obj[k].keys())):
            accepted_init_objects = slice[init_input]
            if not isinstance(obj[k][init_input], accepted_init_objects):
                raise ValueError("{} type in 'init' must be in {}".format(init_input, list(accepted_init_objects)))





####################### -------------------- Init_verifs -------------------- #######################
"""
    Function which verify the good declaration of initial parameters inputed by user and harmonize their format to allow user to input dictionnaries or not
Args:
    init (dict): Dictionnary containing initial parameters inputed by user
    surv_object (ll.CoxPHFitter): Fitted Cox proportional hazard model
    lme_object (statsmodels.regression.mixed_linear_model.MixedLM, bambi.models.Model): Non fitted generalized linear mixed model
"""
def Init_verifs(init, surv_object, lme_object):

    #init expected parameters :
    init_long = {
        'betas': tuple([np.ndarray]),
        'sigma': tuple([int, float])
    }
    init_survival = {
        'gammas': tuple([np.ndarray]),
        'sigma_t': tuple([int, float]),
        'alpha': tuple([int, float, dict]),
        'd_alpha': tuple([int, float, dict])
    }
    init_d_vc = {'d_vc': tuple([np.ndarray])}

    #For case when an empty dictionnary is passed in init
    if init is not None:
        if len(init) == 0:
            init = None
    
    if init is not None:

        #Verifying if init is a dictionnary
        accepted_init_objects = tuple([dict])
        if not isinstance(init, accepted_init_objects):
            raise ValueError("'init' type must be in {}".format(list(accepted_init_objects)))
        
        #Convert d_vc to numpy array if user input a list instead of a numpy array (useful if user use json configuration files)
        if 'd_vc' in init.keys():
            if isinstance(init['d_vc'], list):
                init['d_vc'] = np.array(init['d_vc'])

        #Build a dictionnary with all elements inside init converted to dict
        init_dict = {} #init with only dictionnaries inside (we add to init_dict, non dictionnarized data inside a dictionnary)
        long_alone_init_dict = {} #longitudinal non dictionnarized data will be put inside this dictionnary
        surv_alone_init_dict = {} #survival non dictionnarized data will be put inside this dictionnary
        d_vc_alone_init_dict = {} #variance - covariance matrix components, non dictionnarized will be put inside this dictionnary
        all_init_keys = dict(list(init_long.items()) + list(init_survival.items()) + list(init_d_vc.items())).keys() # List of all possible inputs
        for key in init.keys():
            #We put all dictionnaries inside init_dict
            if isinstance(init[key], tuple([dict])):
                #Special case of alpha and d_alpha
                if key in ['alpha', 'd_alpha']:
                    surv_alone_init_dict[key] = init[key]
                else:
                    init_dict[key] = init[key]
            #We put all non dictionnaries input in init inside a survival or longitudinal dict
            else:
                if key in init_long.keys():
                    long_alone_init_dict[key] = init[key]
                elif key in init_survival.keys():
                    surv_alone_init_dict[key] = init[key]
                elif key in init_d_vc.keys():
                    d_vc_alone_init_dict['d_vc'] = init[key]
                else:
                    raise ValueError("'{}' init parameter is not expected, must be in {}".format(key, list(all_init_keys)))
        
        #We add survival, longitudinal, d_vc dictionnaries in init_dict (if they aren't empty)
        if len(long_alone_init_dict) > 0:
            init_dict['long'] = long_alone_init_dict
        if len(surv_alone_init_dict) > 0:
            init_dict['survival'] = surv_alone_init_dict
        if len(d_vc_alone_init_dict) > 0:
            init_dict['d_vc'] = d_vc_alone_init_dict

        #Discriminating survival init from longitudinal and d_vc init in init_dict
        long_init_keys = []
        surv_init_keys = []
        d_vc_init_keys = []
        for key in init_dict.keys():
            if set(init_dict[key].keys()).intersection(set(init_d_vc.keys())) == set(init_dict[key].keys()):
                d_vc_init_keys.append(key)
            elif set(init_dict[key].keys()).intersection(set(init_long.keys())) == set(init_dict[key].keys()):
                long_init_keys.append(key)
            elif set(init_dict[key].keys()).intersection(set(init_survival.keys())) == set(init_dict[key].keys()):
                surv_init_keys.append(key)
            else:
                #Raising error if unexpecting parameters are inputed in init
                raise ValueError("'{}' dictionnary in init contains errors, possible init parameters are {} for longitudinal parameters and {} for survival parameters".format(key, list(init_long.keys()), list(init_survival.keys())))
        
        #Verifying if init_dict(survival part) and surv_object have same keys
        if len(surv_init_keys) > 0:
            if set(surv_init_keys) != set(surv_object.keys()):
                raise ValueError("'surv_object' and dictionnaries relative to cause-specific risks in 'init' must have same keys")
        
        #Verifying if init_dict(longitudinal part) and other longitudinals parameters have same keys (we only test lme_object, because lme_object have arleady been tested to have the same keys than other longitudinals parameters)
        if len(long_init_keys) > 0:
            if set(long_init_keys) != set(lme_object.keys()):
                raise ValueError("Dictionnaries relative to longitudinal parameters in 'init' must have same keys than 'lme_object', 'lme_object_fitted', 'lme_data', 'lme_formula', 'lme_re_formula' and 'derivForm'. \n Expeced keys are {}".format(list(lme_object.keys())))
        
        #Verifying types of values inputed in longitudinal init_dict and survival init_dict
        Init_check_type({key: init_dict[key] for key in long_init_keys}, init_long)
        Init_check_type({key: init_dict[key] for key in surv_init_keys}, init_survival)
        Init_check_type({key: init_dict[key] for key in d_vc_init_keys}, init_d_vc)

    else:
        init_dict = None
    
    return(init_dict)





####################### -------------------- Init_replace -------------------- #######################
"""
    Function which replace default inital parameters of the model by initial values inputed by user
Args:
    init (dict): Dictionnary containing initial parameters inputed by user
    initial_values (dict): Complete default dictionnaray of parameters used to initialize estimation of joint model parameters
    surv_object (ll.CoxPHFitter): Fitted Cox proportional hazard model
"""
def Init_replace(init, initial_values, surv_object):

    if init is not None:

        #Survival keys of init
        surv_keys_init = set(init.keys()).intersection(set(surv_object.keys()))
        for risk in surv_keys_init:
            # Verifying if there are unexpected names initial parameters in init (survival part)
            input_init_keys = init[risk].keys()
            estim_init_keys = initial_values[risk].keys()
            if len(set(input_init_keys) - set(estim_init_keys)) != 0:
                raise ValueError("Unkown names in 'init' {} risk: {} must be in {}".format(risk, list(set(input_init_keys) - set(estim_init_keys)) ,list(set(estim_init_keys))))
            else:
                #Replacing initial values
                for key in input_init_keys:
                    initial_values[risk][key] = init[risk][key]
                    #Case where key is a dictionnary
                    if isinstance(initial_values[risk][key], tuple([dict])):
                        input_sub_keys = init[risk][key].keys()
                        estim_sub_keys = initial_values[risk][key].keys()
                        if len(set(input_sub_keys) - set(estim_sub_keys)) != 0:
                            raise ValueError("Unkown names in 'init' {} risk, {} parameter : {} must be in {}".format(risk, key, list(set(input_sub_keys) - set(estim_sub_keys)) ,list(set(estim_sub_keys))))
                        else:
                            for sub_key in initial_values[risk][key].keys():
                                initial_values[risk][key][sub_key] = init[risk][key][sub_key]

        #Other keys of init
        long_keys_init = set(init.keys()) - set(surv_object.keys())
        for key in long_keys_init:
            input_init_keys = set(init[key].keys()) - set(surv_object.keys())
            estim_init_keys = set(initial_values.keys()) - set(surv_object.keys())

            # Verifying if there are unknown names initial parameters in init
            if len(set(input_init_keys) - set(estim_init_keys)) != 0:
                raise ValueError("Unkown names in 'init' : {}. Must be in {}".format(list(set(input_init_keys) - set(estim_init_keys)) ,list(set(estim_init_keys))))
            else:
                #Replacing initial values
                for input_key in input_init_keys:
                    if input_key == 'd_vc': #Special case for d_vc which isn't stored inside a dictionnary because, we only have one covariance random effect matrix
                        initial_values[input_key] = init[key][input_key]
                    else:
                        initial_values[input_key][key] = init[key][input_key]

    return(initial_values)





####################### -------------------- Thetas_feed -------------------- #######################
"""
    Function useful only to insert constant thetas inside thetas passed to optimization solver. It insert constant thetas inside thetas vector
Args:
    thetas (np.array): Numpy array containing thetas parameters set used to minimize log-likelihood function
    thetas_c (np.array): Numpy array containing thetas parameters remaining constant during log-likelihood minimization. Each values of thetas_c take value np.nan if the corresponding theta parameter don't remain constant and the constant value if the theta parameter remain constant
"""
def Thetas_feed(thetas, thetas_c):
    for i in range(len(thetas_c)):
        if not np.isnan(thetas_c[i]):
            thetas = np.insert(thetas, i, thetas_c[i])
    
    return(thetas)





####################### -------------------- Thetas_feed -------------------- #######################
"""
    Replace all deepest values of nested dictionnary by numpy nan
Args:
    dictionnary (np.array): Dictionnary from which we wants to replace values by numpay nn
"""
def Replace_values_by_na(dictionnary):
            dictionnary_out = deepcopy(dictionnary)
            for key in dictionnary_out :
                if isinstance(dictionnary_out[key],dict) :
                    dictionnary_out[key] = Replace_values_by_na(dictionnary_out[key])
                elif isinstance(dictionnary_out[key],np.ndarray) :
                    dictionnary_out[key][:] = np.nan
                else:
                    #We need None values (alpha, d_alpha) to stay None values
                    if dictionnary_out[key] is not None:
                        dictionnary_out[key] = np.nan
            return dictionnary_out





####################### -------------------- Group_by_sum -------------------- #######################
"""
    Do same as pandas.groupby().sum() but faster
Args:
    matrix (np.array): Array from which we wants to makes sums by group
    groups (np.array): Array containing groups (groups must be sorted)
    axis (int): axis on which we wants to compute group sum on matrix
"""
def Group_by_sum(matrix, groups, axis):

    group_sum = np.array([matrix[groups == i,:].sum(axis=axis) for i in np.unique(groups)])

    return group_sum