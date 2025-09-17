####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import pandas as pd
import numpy as np

#Formulas
import patsy

#Others
import arviz as az





####################### -------------------- coxph extraction function -------------------- #######################
"""
Function which extract all values we need from coxph model in input
"""
def Extract_coxph(
    surv_object,
    surv_data,
    risk
    ):

    #w : Baseline datas in cox model
    design_w = patsy.dmatrix(surv_object[risk].formula, surv_data)
    w = pd.DataFrame(design_w, columns = design_w.design_info.term_names).iloc[:,1:] #Removing intercept
                
    #d : Indicator of observation of event during the follow-up
    evt_col = surv_object[risk].event_col
    d = np.array(surv_data[evt_col])[np.newaxis].T

    return(w, d)





####################### -------------------- statsmodels MixedLM extraction function -------------------- #######################
"""
Function which extract all values we need from statsmodels MixedLM model in input
"""
def Extract_sm_MixedLM(
    lme_object,
    lme_object_fitted
    ):

    #id_l : numerical group id associated with each longitunal value
    groups = lme_object.groups
    id_l = pd.factorize(groups)[0]

    #fe : Fixed effects
    fe = lme_object_fitted.fe_params.values

    #b : random effects vector
    b = pd.DataFrame.from_dict(lme_object_fitted.random_effects, orient='index')

    #vc : Variance-Covariance matrix for random effects
    vc = lme_object_fitted.cov_re

    #sigma : Residual standard deviation
    sigma = np.sqrt(lme_object_fitted.scale)

    #All informations extracted from model are returned in a dictionnary
    out_extract = dict(
        id_l = id_l,
        fe = fe,
        b = b,
        vc = vc,
        sigma = sigma,
        family_link = 'identity',
        family_name = 'gaussian'
    )

    return(out_extract)





####################### -------------------- bambi GLMM extraction function -------------------- #######################
"""
Function which extract all values we need from bambi glmm model in input
"""
def Extract_bambi_glmm(
    lme_object,
    lme_object_fitted
    ):

    #Name of parameter representing E(Y|X) glmm response ('mu' for gaussian, 'p' for bernouilli etc)
    parent_param = lme_object.family.likelihood.parent

    #Groups for random effects are the same for each group-specific term while we only have one column of groups for random effects
    random_keys = lme_object.response_component.group_specific_terms.keys()
    key_random_1 = list(random_keys)[0]

    #id_l : numerical group id associated with each longitunal value
    groups = np.asarray(lme_object.response_component.group_specific_terms[key_random_1].groups) #Sorted groups id
    groups_idx = lme_object.response_component.group_specific_terms[key_random_1].group_index #Index of all longitudinal responses associated group in sorted groups id 
    id_l = pd.factorize(groups[groups_idx])[0]
    
    #fe : Fixed effects
    fixed_terms = [lme_object.response_component.intercept_term.name] + [lme_object.response_component.common_terms[key].name for key in lme_object.response_component.common_terms.keys()] #Fixed terms names    
    fe = np.array(az.summary(lme_object_fitted, var_names=fixed_terms, group = 'posterior', kind='stats', stat_focus='mean')['mean']) #Extraction of estimation of estimated fixed effects

    #b : estimated random effects
    dfs_random = []
    for random_key in list(random_keys):

        #Extraction of estimation of random effect on variable concerned by random_key
        df_r = az.summary(lme_object_fitted, var_names=random_key, group='posterior', kind='stats', stat_focus='mean')['mean']

        #Replacement of index by group patient key
        df_r.index = lme_object.response_component.group_specific_terms[random_key].groups

        #Change name to replace "mean" by random_key
        df_r.name = random_key

        #Adding fitted random effects in list of pd.Series to concatenate
        dfs_random.append(df_r)
    
    #Concatenation of all pd.Series containing random effects
    b = pd.concat(dfs_random, axis=1)
    
    #vc : Variance-Covariance matrix for random effects
    vc = np.array(b.cov()) #bambi glmm don't provides covariates betwteen random effects, so we give an imperfect estimation instead
    b_sigma_terms = ['{}_sigma'.format(key) for key in random_keys] #Random effect Variance terms names
    b_sigma = np.array(az.summary(lme_object_fitted, var_names=b_sigma_terms, group = 'posterior', kind='stats', stat_focus='mean')['mean'] ** 2) #Estimated variances of random effects
    np.fill_diagonal(vc, b_sigma) #Replacement of cov matrix diagonal by variances of random effects
    vc_pd = pd.DataFrame(vc, index=random_keys, columns=random_keys)

    #sigma : Residual standard deviation
    # For gaussian response vector
    if lme_object.family.name == 'gaussian' and lme_object.family.link[parent_param].name == 'identity':
        sigma_term = '{}_sigma'.format(lme_object.response_name) #Sigma term names
        sigma = np.array(az.summary(lme_object_fitted, var_names=sigma_term, group = 'posterior', kind='stats', stat_focus='mean')['mean'])
    
    # For bernoulli response vector
    if lme_object.family.name == 'bernoulli':
        sigma = None
    
    #All informations extracted from model are returned in a dictionnary
    out_extract = dict(
        id_l = id_l,
        fe = fe,
        b = b,
        vc = vc_pd,
        sigma = sigma,
        family_link = lme_object.family.link[parent_param].name,
        family_name = lme_object.family.name
    )

    return(out_extract)