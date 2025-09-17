####################### -------------------- Import libraries and modules -------------------- #######################

# Usual libraries
import numpy as np
import pandas as pd

# Models
import lifelines as ll
import statsmodels
import bambi
import arviz
from statsmodels.formula.api import mixedlm

#Formulas
import patsy

#Linear algebra
from scipy.linalg import block_diag

# Optimisation
from scipy.optimize import minimize

#Probabilities
from scipy.stats import norm, multivariate_normal, uniform, qmc

#Estimations
from sklearn.covariance import empirical_covariance

#For outputs
from tabulate import tabulate

#Loading bar
from tqdm import tqdm

#Warnings
import warnings

#Deepcopy
from copy import deepcopy

#Quadratures
from .utils.quadratures import Gauss_Kronrod

# Modules from our scripts
from .utils import jointModels_fits as jmf
from .utils import toolbox as tb
from .utils.extract_models import Extract_coxph
from .utils.extract_models import Extract_sm_MixedLM
from .utils.extract_models import Extract_bambi_glmm
from .utils.initialSurv import Initial_surv
from .utils.model_mats import Model_mats
from .utils.log_posterior_b import Log_posterior_b
from .utils.s_b import S_b
from .utils.cd import Cd
from .utils.chol_trans import Chol_transf
from .utils.rmvt import Rmvt
from .utils.dmvt import Dmvt
from .utils.toolbox import Dict_skel, Dapply
from .utils.survivals import CIF, Survival




####################### -------------------- JointModels's class -------------------- #######################

class JointModel:
    
    def __init__(
        self,
        lme_object,
        lme_object_fitted,
        lme_data,
        lme_formula,
        lme_re_formula,
        surv_object,
        surv_data,
        time_var,
        parametrization,
        method = "weibull-PH-GH",
        interfact = None,
        derivForm = None,
        lag = 0,
        scale_wb = None,
        competing_risks = None,
        control = {},
        init = None,
        constant = None,
        **kwargs
    ):



        ### Check parameters compatibilities ---

        #method : For instant we only carry "weibull-PH-GH" and "weibull-PH-QMC" method for parameter "method"
        accepted_method = ["weibull-PH-GH", "weibull-PH-QMC"]
        if method not in accepted_method:
            raise ValueError("'method' must be in {}".format(accepted_method))
        
        #surv_object :
        accepted_surv_objects = tuple([ll.CoxPHFitter])
        if not isinstance(surv_object, dict):
            if competing_risks == 'cs':
                raise ValueError("Models inputed in 'surv_object' must be stocked inside a dictionnary when competing_risks = 'cs'") #Raise error if the user given parametrized jm function for competing risks, but don't give a dictionnary in input
        surv_object = tb.Alone_to_dict(surv_object, 'survival')
        for risk in surv_object.keys():
            if not isinstance(surv_object[risk], accepted_surv_objects):
                raise ValueError("Models type in 'surv_object' must be in {}".format(list(accepted_surv_objects)))
            if surv_object[risk].formula is None:
                raise ValueError("Formula argument of 'surv_object' must be specified")
        
        #Competing risk :
        if competing_risks is None and len(set(surv_object.keys())) > 1:
            raise ValueError("Only one risk is expected when competing risk is None, you gave {} in surv_object parameter".format(len(set(surv_object.keys()))))

        #surv_data :
        accepted_surv_data = tuple([pd.DataFrame])
        if not isinstance(surv_data, accepted_surv_data):
            raise ValueError("'surv_data' type must be in {}".format(list(accepted_surv_data)))
        for risk in surv_object.keys():
            missing_cols = [col for col in surv_object[risk].params_.index if col not in surv_data.columns]
            if len(missing_cols) > 0:
                raise ValueError("surv data miss {}".format(list(missing_cols)))
            if surv_object[risk].event_col not in surv_data.columns:
                raise ValueError("surv data miss {}".format(list(surv_object[risk].event_col)))
        
        #parametrization :
        accepted_parametrization = ['value', 'slope', 'both']
        parametrization = tb.Alone_to_dict(parametrization, 'long')
        for mark in parametrization.keys():
            if parametrization[mark] not in accepted_parametrization:
                raise ValueError("'parametrization' value must be in {}".format(list(accepted_parametrization)))
        
        #lme_object :
        accepted_lme_objects = tuple([statsmodels.regression.mixed_linear_model.MixedLM, bambi.models.Model])
        lme_object = tb.Alone_to_dict(lme_object, 'long')
        for mark in lme_object.keys():
            if not isinstance(lme_object[mark], accepted_lme_objects):
                raise ValueError("'lme_object' type must be in {}".format(list(accepted_lme_objects)))

        #lme_object_fitted :
        accepted_lme_fitted_objects = tuple([statsmodels.regression.mixed_linear_model.MixedLMResultsWrapper, arviz.data.inference_data.InferenceData])
        lme_object_fitted = tb.Alone_to_dict(lme_object_fitted, 'long')
        for mark in lme_object_fitted.keys():
            if not isinstance(lme_object_fitted[mark], accepted_lme_fitted_objects):
                raise ValueError("'lme_object_fitted' type must be in {}".format(list(accepted_lme_fitted_objects)))
        
        #lme_data :
        accepted_lme_data = tuple([pd.core.frame.DataFrame])
        lme_data = tb.Alone_to_dict(lme_data, 'long')
        for mark in lme_data.keys():
            if not isinstance(lme_data[mark], accepted_lme_data):
                raise ValueError("'lme_data' type must be in {}".format(list(accepted_lme_data)))
        
        #lme_formula :
        lme_formula = tb.Alone_to_dict(lme_formula, 'long')

        #lme_re_formula :
        lme_re_formula = tb.Alone_to_dict(lme_re_formula, 'long')

        #time_var
        time_var = tb.Alone_to_dict(time_var, 'long')
        
        #derivForm :
        derivForm = tb.Alone_dict_to_dict(derivForm, 'long')
        for mark in derivForm.keys():
            if not isinstance(derivForm[mark], dict):
                raise ValueError("Derivate formulations inputed in 'derivForm' must be dictionnaries")
            if set(derivForm[mark].keys()) != {'fixed', 'ind_fixed', 'random', 'ind_random'}:
                raise ValueError("Derivate formulations inputed in 'derivForm' must be dictionnaries with keys {'fixed', 'ind_fixed', 'random', 'ind_random'}")

        #Check if all longitudinals parameters inputed in dictionnaries have same keys
        if not set(parametrization.keys()) == set(lme_object.keys()) == set(lme_object_fitted.keys()) == set(lme_data.keys()) == set(lme_formula.keys()) == set(lme_re_formula.keys()) == set(time_var.keys()) == set(derivForm.keys()):
            raise ValueError("'lme_object', 'lme_object_fitted', 'lme_data', 'lme_formula', 'lme_re_formula', 'time_var' and 'derivForm' must have the same keys")

        #solvers for optimization :
        accepted_solvers = ["BFGS", "L-BFGS-B", "Nelder-Mead", "Powell", "SLSQP", "COBYLA", "trust-constr"] #Accepted solvers
        accepted_solvers_hess = ["BFGS", "L-BFGS-B"] #Accepted solvers for hessian inversion given by solver use as cov matrix
        if 'solver_minimize' in control.keys():
            if control['solver_minimize'] not in accepted_solvers:
                raise ValueError("'solver_minimize' must be in {}".format(accepted_solvers))
            #Case where user use 'optimizer_hess_inv' with a incompatible solver
            if 'numeri_deriv' in control.keys():
                if control['numeri_deriv'] == 'optimizer_hess_inv' and control['solver_minimize'] not in accepted_solvers_hess:
                    raise ValueError("'numeri_deriv = optimizer_hess_inv' only available for {} solvers".format(accepted_solvers_hess))
        
        #init parameters inputed by user :
        init_dict = tb.Init_verifs(init, surv_object, lme_object)

        #constant parameters inputed by user
        constant_dict = tb.Init_verifs(constant, surv_object, lme_object)
        



        ### Creation of useful attributes for the follow of script ---

        #Longitudinal keys
        long_keys = list(lme_object.keys())

        #Survival keys : think to replace all self.surv_object.keys() by surv_keys later
        surv_keys = list(surv_object.keys())

        #Discriminitaion of groups variables for longitudinal models
        groups_col = {}

        for key in long_keys:
            if isinstance(lme_object[key], statsmodels.regression.mixed_linear_model.MixedLM):
                # (The only way i found to discriminate the group column from statsmodels, need to find better)
                groups_col[key] = np.array([col for col in lme_data[key].columns if sum(lme_object[key].groups != lme_data[key][col].values) == 0])
            elif isinstance(lme_object[key], bambi.models.Model):
                groups_col[key] = np.array(list(lme_object[key].response_component.group_specific_groups.keys())).flatten()
            #Becareful, the class is not implemented to handle more than one group column
            if len(groups_col[key]) != 1:
                raise ValueError("'lme_object' must have one and only one group column for random effects for each longitudinal model. JointModel is not yet implemented to handle different cases. You inputed {} groups columns for {} longitudinal model".format(len(groups_col[key]), key))

        ### Affectation of input parameters ---
        self.long_keys = long_keys
        self.lme_object = lme_object
        self.lme_object_fitted = lme_object_fitted
        self.lme_data = lme_data
        self.lme_formula = lme_formula
        self.lme_re_formula = lme_re_formula
        self.surv_keys = surv_keys
        self.surv_object = surv_object
        self.surv_data = surv_data
        self.time_var = time_var
        self.parametrization = parametrization
        self.method = method
        self.interfact = interfact
        self.derivForm = derivForm
        self.lag = lag
        self.scale_wb = scale_wb
        self.groups_col = groups_col
        self.competing_risks = competing_risks
        self.control = control
        self.init = init_dict
        self.constant = constant_dict
        self.kwargs = kwargs
        self.prop = None





    def _get_model_params(
        self
    ):

        ### Survival Process ---

        #time : Time to event or censoring
        key_1 = self.surv_keys[0] #We have same times to 'events' or 'censoring' in cs, so we take any cause-specific key to compute time variable
        duration_col = self.surv_object[key_1].duration_col
        time = np.array(self.surv_data[duration_col])

        #id_t : id for each time to event or censoring
        id_t = np.array(range(len(time)))

        #nt : number of time to event or censoring
        nt = len(id_t)

        #n_risks : Number of risks in competition
        n_risks = len(self.surv_keys)

        #Weights wint_f_vl and wint_f_sl for each patients applied to value and slope
        mat_1 = np.repeat(np.array([1]), len(time))
        wint_f_vl = mat_1.copy()
        wint_f_sl = mat_1.copy()

        #Variables relating to survival data who can differs in cause-specific competing risk
        w = {} #Baseline data in propotional hazard model
        d = {} #Indicator of observation of event during the follow

        #Extraction of baseline data and evt indicator for each risk
        for risk in self.surv_keys:
            if self.surv_object[risk]._class_name == "CoxPHFitter":
                w[risk], d[risk] = Extract_coxph(self.surv_object, self.surv_data, risk)
        



        ### Longitudinal process ---

        #Generalization for extraction of parameters from different sources of longitudinal models
        generalized_extract_long = {}
        for key in self.long_keys:
            if isinstance(self.lme_object[key], statsmodels.regression.mixed_linear_model.MixedLM):
                generalized_extract_long[key] = Extract_sm_MixedLM(self.lme_object[key], self.lme_object_fitted[key])
            elif isinstance(self.lme_object[key], bambi.models.Model):
                generalized_extract_long[key] = Extract_bambi_glmm(self.lme_object[key], self.lme_object_fitted[key])
       
        #id_l : position of group id associated at each longitunal value
        id_l = {key: generalized_extract_long[key]['id_l'] for key in self.long_keys}

        #fe : Fixed effects
        fe = {key: generalized_extract_long[key]['fe'] for key in self.long_keys}

        #b : random effects
        b = {key: generalized_extract_long[key]['b'] for key in self.long_keys}

        #n_y : number of patients in longitudinal process (length of random effects vectors)
        n_y = {key: len(b[key]) for key in self.long_keys}

        #sigma : Residual standard deviation
        sigma = {key: generalized_extract_long[key]['sigma'] for key in self.long_keys}

        #family_link and family_name : family and link of longitudinal regression
        family_link = {key: generalized_extract_long[key]['family_link'] for key in self.long_keys}
        family_name = {key: generalized_extract_long[key]['family_name'] for key in self.long_keys}

        #Model matrixes for fixed and random effects from associated formulas
        design_fe = {key: patsy.dmatrices(self.lme_formula[key], self.lme_data[key]) for key in self.long_keys}
        design_re = {key: patsy.dmatrix(self.lme_re_formula[key], self.lme_data[key]) for key in self.long_keys}

        #x_lme : fixed effects design matrix in linear mixed model
        x_lme = {key: pd.DataFrame(design_fe[key][1], columns=design_fe[key][1].design_info.term_names) for key in self.long_keys}

        #z_lme : random effects design matrix in linear mixed model
        z_lme = {key: np.array(design_re[key]) for key in self.long_keys}

        #y_lme : response variable of linear mixed model
        y_lme = {key: np.array(design_fe[key][0]) for key in self.long_keys}
        
        #data_id : Reconstruction of design matrix used for lme with only first lign for each group and time to event replacing time associated to marker
        data_id = {}
        for key in self.long_keys:
            idx_unique = self.lme_data[key].index[np.unique(id_l[key], return_index=True)[1]]
            design_time = patsy.dmatrices(self.lme_formula[key], self.lme_data[key].loc[idx_unique,:])
            data_id[key] = pd.DataFrame(
                np.concatenate(
                    (design_time[0], design_time[1]),
                    axis=1
                ),
                columns = np.concatenate(
                    (design_time[0].design_info.term_names, design_time[1].design_info.term_names),
                    axis=None
                )
            )
            time_lag_0 = np.maximum(time - self.lag, np.zeros(len(time))) #Max beetween time - lag and 0
            data_id[key][self.time_var[key]] = time_lag_0

        #For curent marker :

        #Model matrixes for fixed and random effects from associated formulas        
        design_fe_time = {key: patsy.dmatrices(self.lme_formula[key], data_id[key]) if self.parametrization[key] in ['value', 'both'] else None for key in self.long_keys}
        design_re_time = {key: patsy.dmatrix(self.lme_re_formula[key], data_id[key]) if self.parametrization[key] in ['value', 'both'] else None for key in self.long_keys}

        #x_lme_time : fixed effects design matrix in linear mixed model with time to event replacing times of longitudinal marquors
        x_lme_time = {key: np.array(design_fe_time[key][1]) if self.parametrization[key] in ['value', 'both'] else None for key in self.long_keys}

        #z_lme_time : random effects design matrix in linear mixed model with time to event replacing times of longitudinal marquors
        z_lme_time = {key: np.array(design_re_time[key]) if self.parametrization[key] in ['value', 'both'] else None for key in self.long_keys}
        
        #long_cur : Estimation of response variable of linear mixed model at times of longitudinal markors
        long_cur = {key: np.matmul(np.array(x_lme[key]), fe[key]) + (np.array(z_lme[key]) * np.array(b[key])[id_l[key]]).sum(axis=1) if self.parametrization[key] in ['value', 'both'] else None for key in self.long_keys}

        #For derivated marker

        #Model matrixes for fixed and random effects from associated formulas
        design_fe_deriv = {key: patsy.dmatrix(self.derivForm[key]['fixed'], self.lme_data[key]) if self.parametrization[key] in ['slope', 'both'] else None for key in self.long_keys}
        design_re_deriv = {key: patsy.dmatrix(self.derivForm[key]['random'], self.lme_data[key]) if self.parametrization[key] in ['slope', 'both'] else None for key in self.long_keys}
        design_fe_time_deriv = {key: patsy.dmatrix(self.derivForm[key]['fixed'], data_id[key]) if self.parametrization[key] in ['slope', 'both'] else None for key in self.long_keys}
        design_re_time_deriv = {key: patsy.dmatrix(self.derivForm[key]['random'], data_id[key]) if self.parametrization[key] in ['slope', 'both'] else None for key in self.long_keys}

        #x_lme_deriv : fixed effects design matrix in linear mixed model
        x_lme_deriv = {key: pd.DataFrame(design_fe_deriv[key], columns=design_fe_deriv[key].design_info.term_names) if self.parametrization[key] in ['slope', 'both'] else None for key in self.long_keys}

        #z_lme_deriv : random effects design matrix in linear mixed model
        z_lme_deriv = {key: pd.DataFrame(design_re_deriv[key], columns=design_re_deriv[key].design_info.term_names) if self.parametrization[key] in ['slope', 'both'] else None for key in self.long_keys}

        #x_lme_time_deriv : fixed effects design matrix in linear mixed model with time to event replacing times of longitudinal marquors
        x_lme_time_deriv = {key: np.array(design_fe_time_deriv[key]) if self.parametrization[key] in ['slope', 'both'] else None for key in self.long_keys}

        #z_lme_time_deriv : random effects design matrix in linear mixed model with time to event replacing times of longitudinal marquors
        z_lme_time_deriv = {key: np.array(design_re_time_deriv[key]) if self.parametrization[key] in ['slope', 'both'] else None for key in self.long_keys}

        #long_deriv : Estimation of derivate of response variable of linear mixed model at times of longitudinal markors
        long_deriv = {key: np.matmul(np.array(x_lme_deriv[key]), fe[key][self.derivForm[key]['ind_fixed']]) + (np.array(z_lme_deriv[key]) * np.array(b[key].iloc[id_l[key], self.derivForm[key]['ind_random']])).sum(axis=1) if self.parametrization[key] in ['slope', 'both'] else None for key in self.long_keys}

        #vc : Variance-Covariance matrix for random effects
        #bs = {key: generalized_extract_long[key]['b'] for key in self.long_keys}
        #bs_concat = pd.concat(bs.values(), axis=1)

        #Empirical vc observed on posterior random effects of glmms
        #vc2 = empirical_covariance(bs_concat, assume_centered=True)

        #vc estimed by each random effects of glmms
        vcs = {key: np.array(generalized_extract_long[key]['vc']) for key in self.long_keys}
        vc = block_diag(*vcs.values())
        #vc_input = block_diag(*vcs.values())

        #Replacement of vc by non null values of vc_input
        #vc[np.where(vc_input!=0)] = vc_input[np.where(vc_input!=0)]
        inv_chol_vc = np.linalg.inv(np.linalg.cholesky(np.linalg.inv(vc))).T #Inverse of Cholesky decomposition of inverse of variance-covariance matrix for random effects. I transposed inv_chol_vc to have the same matrix than in R, but, i think the calculation stay correct without this transposition
        det_inv_chol_vc = np.linalg.det(np.linalg.inv(np.linalg.cholesky(np.linalg.inv(vc)))) #Determinant of inv_chol_vc


        ### Control parameters ---

        #ind_noadapt : Indicate if method parameter for estimation is adaptative or not
        ind_noadapt = self.method in ['weibull-PH-GH']            

        #con : Dictionary containing some control parameters for estimation
        con = {
            'only_em': False,
            'iter_em': 120 if self.method == 'spline-PH-GH' else 50,
            'iter_qN': 350,
            'optimizer': 'optim',
            'tol1': 1e-03,
            'tol2': 1e-04,
            'tol3': 1e-09 if self.competing_risks is not None else np.sqrt(np.finfo(float).eps),
            'tol4': 10e-100,
            'numeri_deriv': "cd",
            'eps_hes': 1e-06, 
            'parscale': None,
            'step_max': 0.1,
            'backtrack_steps': 2, 
            'knots': None,
            'obs_times_knots': True,
            'lng_in_kn': 6 if self.method == 'piecewise-PH-GH' else 5,
            'ord': 4, 
            'equal_strata_knots': True,
            'ghk': 15 if (max([z_lme[key].shape[1] for key in self.long_keys]) < 3 and max([z_lme[key].shape[0] for key in self.long_keys]) < 2000) else 9,
            'gkk': 7 if (self.method == 'piecewise-PH-GH' or len(time) > n_risks*nt) else 15,
            'nbmc': 2**10, #Number of Monte Carlo simulations for estimation of the integral over random effects in log likelihood
            'seed_MC': np.random.randint(1e18), #Seed for Monte carlo simulations
            'inv_chol_vc': inv_chol_vc,
            'det_inv_chol_vc': det_inv_chol_vc,
            'ranef': {key: b[key].to_numpy() for key in self.long_keys},
            'solver_minimize_em': 'L-BFGS-B', #Method used by scipy minimize to fit other parameters (with no analytic solution) in EM
            'solver_minimize': 'L-BFGS-B', #Method used by scipy minimize to fit parameters
            'options_minimize_em': {'ftol' : 1e-6} if self.competing_risks is not None else None, #Options passed to scipy minimize to fit other parameters (with no analytic solution) in EM
            'options_minimize': {'ftol' : 1e-7} if self.competing_risks is not None else None, #Options passed to scipy minimize to fit parameters
            'options_noise_minimize': {'max_ls_fails': 4},
            'jac_minimize': None, #Jacobian passed to scipy_minimize to fit parameters
            'jac_minimize_em': None, #Jacobian passed to scipy_minimize to fit other parameters (with no analytic solution) in EM
            'n_cores_grad': 1,
            'eps_f': None, #Noise level of log-likelihood function (when QMC)
            'eps_g': None #Noise level of the gradient of log-likelihood (when QMC)
        }
        
        #Replacing default values of con by values choosen by user in control and kwargs inputs of the class
        control = dict(**self.control, **self.kwargs)
        for key in control.keys():
            if key in con.keys():
                con[key] = control[key]
        
        #Avoiding EM algorithm for non-gaussian markors
        if len(self.long_keys) > 1:
            warnings.warn("Warning......EM algorithm isn't available for multiple longitudinal markors. JM will be estimated with quasi-newton method instead")
            con['iter_em'] = 0
        else:
            if family_name[self.long_keys[0]] != 'gaussian':
                warnings.warn("Warning......EM algorithm isn't available for non gaussian longitudinal markors. JM will be estimated with quasi-newton method instead")
                con['iter_em'] = 0


        


        ### Design matrices to estimates integrand for cumulative risk in log-likelihood  ---
        
        #Test if param have one of values concerned
        ind_method = self.method in ['weibull-PH-GH', 'weibull-PH-QMC']

        if ind_method :

            #wk and sk : Gauss-Kronrod quadrature weights and nodes
            wk, sk = Gauss_Kronrod(con['gkk'])

            #p : time to events values divided by 2
            p = time/2

            #st : All time to events (divided by 2) multiplied by all gauss-kronrod quadrature nodes + 1 (one lign corresponds to an observation and a column corresponds to one of quadrature values)
            st = np.matmul(p[np.newaxis].T, (sk+1)[np.newaxis])

            #data_id2 : Expansion of data_id with values used for quadrature estimation (of cumulative risk) replacing time to event
            id_gk = np.repeat(id_t, con['gkk'])
            data_id2 = {key: data_id[key].iloc[id_gk,:].reset_index() for key in self.long_keys} #reset index to avoid duplicated index
            stime_lag_0 = np.maximum(st.flatten() - self.lag, np.zeros(len(st.flatten()))) #Max beetween flatten st - lag and 0 (to avoid negative time)
            for key in self.long_keys:
                data_id2[key][self.time_var[key]] = stime_lag_0

            #For curent marker
            
            #Model matrixes for fixed and random effects from associated formulas        
            design_fe_s = {key: patsy.dmatrices(self.lme_formula[key], data_id2[key]) if self.parametrization[key] in ['value', 'both'] else None for key in self.long_keys}
            design_re_s = {key: patsy.dmatrix(self.lme_re_formula[key], data_id2[key]) if self.parametrization[key] in ['value', 'both'] else None for key in self.long_keys}

            #x_lme_s : fixed effects design matrix in linear mixed model with stime_lag_0 replacing times of longitudinal marquors
            x_lme_s = {key: np.array(design_fe_s[key][1]) if self.parametrization[key] in ['value', 'both'] else None for key in self.long_keys}
            
            #z_lme_s : random effects design matrix in linear mixed model with stime_lag_0 replacing times of longitudinal marquors
            z_lme_s = {key: np.array(design_re_s[key]) if self.parametrization[key] in ['value', 'both'] else None for key in self.long_keys}

            #For derivated marker

            #Model matrixes for fixed and random effects from associated formulas
            design_fe_s_deriv = {key: patsy.dmatrix(self.derivForm[key]['fixed'], data_id2[key]) if self.parametrization[key] in ['slope', 'both'] else None for key in self.long_keys}
            design_re_s_deriv = {key: patsy.dmatrix(self.derivForm[key]['random'], data_id2[key]) if self.parametrization[key] in ['slope', 'both'] else None for key in self.long_keys}

            #x_lme_s_deriv : fixed effects design matrix in linear mixed model with time to event replacing times of longitudinal marquors
            x_lme_s_deriv = {key: np.array(design_fe_s_deriv[key]) if self.parametrization[key] in ['slope', 'both'] else None for key in self.long_keys}

            #z_lme_s_deriv : random effects design matrix in linear mixed model with time to event replacing times of longitudinal marquors
            z_lme_s_deriv = {key: np.array(design_re_s_deriv[key]) if self.parametrization[key] in ['slope', 'both'] else None for key in self.long_keys}

            #wsint_f_vl and wsint_f_sl : wint_f_vl and wint_f_sl "shape-adapted" for st
            wsint_f_vl = wint_f_vl[id_gk]
            wsint_f_sl = wint_f_sl[id_gk]



        ###Initial values parameters of joint model ---

        #init_lme : initial values of parameters of longitudinal part to estimate the joint model
        init_lme = {
            'betas': fe,
            'sigma': sigma,
            'd_vc': vc
        }
        
        #init_surv : initial values of parameters of survival part to estimate the joint model
        init_surv = {}
        for risk in self.surv_object.keys():
            init_surv[risk] = Initial_surv(
                time_s = time,
                d = d[risk].flatten(),
                w = w[risk],
                id_l = id_l,
                time_l = {key: np.array(x_lme[key][self.time_var[key]]) for key in self.long_keys},
                method = self.method,
                parametrization = self.parametrization,
                long_cur = long_cur,
                long_deriv = long_deriv
            )

        #initial_values : Complete dictionnaray of parameters used to initialize estimation of joint model parameters
        initial_values = dict(**init_lme, **init_surv)

        #dict_na : initial_values but with only na's in place of values (not in place of keys)
        dict_na = deepcopy(initial_values)
        dict_na['d_vc'] = np.log(dict_na['d_vc']) if dict_na['d_vc'].ndim == 1 else Chol_transf(dict_na['d_vc'])
        dict_na = tb.Replace_values_by_na(dict_na)

        #Replacing values in initial_values by corresponding values inputed by user
        initial_values = tb.Init_replace(self.init, initial_values, self.surv_object)

        #Replacing values in initial_values by constant values inputed by user
        constant_values = tb.Init_replace(self.constant, dict_na, self.surv_object)
        
        ###Dictionary with all matrixes, parameters, etc... we want to return ---
        dict_params_matrix = {
            'con': con,
            'd': d,
            'deriv_form': self.derivForm,
            'family_link': family_link,
            'family_name': family_name,
            'initial_values': initial_values,
            'constant_values': constant_values,
            'id_l': id_l,
            'id_t': id_t,
            'ind_noadapt': ind_noadapt,
            'log_time': np.log(time)[np.newaxis].T,
            'n_risks': n_risks,
            'nt': nt,
            'n_y': n_y,
            'long_keys': self.long_keys,
            'surv_keys': self.surv_object.keys(),
            'competing_risks': self.competing_risks,
            'p': p if ind_method else None,
            'parametrization': self.parametrization,
            'scale_wb': self.scale_wb,
            'sk': sk if ind_method else None,
            'st': st.flatten() if ind_method else None,
            'vc': vc,
            'w': w,
            'wint_f_vl': wint_f_vl[np.newaxis].T,
            'wint_f_sl': wint_f_sl[np.newaxis].T,
            'wsint_f_vl': wsint_f_vl[np.newaxis].T if ind_method else None,
            'wsint_f_sl': wsint_f_sl[np.newaxis].T if ind_method else None,
            'wk': wk if ind_method else None,
            'x_lme': x_lme,
            'x_lme_s': x_lme_s,
            'x_lme_s_deriv': x_lme_s_deriv,
            'x_lme_time': x_lme_time,
            'x_lme_time_deriv': x_lme_time_deriv,
            'y_lme': y_lme,
            'z_lme': z_lme,
            'z_lme_s': z_lme_s,
            'z_lme_s_deriv': z_lme_s_deriv,
            'z_lme_time': z_lme_time,
            'z_lme_time_deriv': z_lme_time_deriv
        }
        return(dict_params_matrix)








    
    def fit(
        self
    ):

        ### Fit our model calling fit function associated to fit method called by user ---
        if self.method == 'weibull-PH-QMC':
            out = jmf.WeibullPH_fit(self._get_model_params(), 'QMC')
        if self.method == 'weibull-PH-GH':
            out = jmf.WeibullPH_fit(self._get_model_params(), 'GH')

        #Extract parameters
        thetas = np.array(out['coefficients']['thetas']).flatten()
        betas = out['coefficients']['betas']
        gammas = out['coefficients']['gammas']
        sigma = out['coefficients']['sigma']
        alpha = out['coefficients']['alpha']
        d_alpha = out['coefficients']['d_alpha']
        sigma_t = out['coefficients']['sigma_t']
        d_vc = out['coefficients']['d_vc']

        #Extract Random effects
        post_b = out['ranef']['post_b']

        #Extract other infos
        hessian = out['hessian']
        log_lik = out['log_lik']
        thetas_skel = out['coefficients']['thetas_skel']
        thetas_names = out['coefficients']['thetas_names']

        #thetas we want to report
        long_keys = list(self.lme_object.keys())
        surv_keys = list(self.surv_object.keys())
        var_long = thetas_names[np.array([thetas_skel['pos_betas'][key] for key in long_keys]).flatten()]
        var_gammas = thetas_names[np.concatenate([thetas_skel['pos_gammas'][risk] for risk in surv_keys], axis=None)]
        var_sigma_t = thetas_names[np.array([thetas_skel['pos_sigma_t'][risk] for risk in surv_keys]).flatten()]
        var_alpha = []
        var_d_alpha = []
        for risk in surv_keys:
            for key in long_keys:
                if self.parametrization[key] in ['value', 'both']:
                    var_alpha.append(thetas_names[thetas_skel['pos_alpha'][risk][key]])
                if self.parametrization[key] in ['slope', 'both']:
                    var_d_alpha.append(thetas_names[thetas_skel['pos_d_alpha'][risk][key]])
        var_alpha = np.array(var_alpha).flatten()
        var_d_alpha = np.array(var_d_alpha).flatten()
        if len(var_d_alpha) == 0:
            var_surv = np.concatenate((var_gammas, var_alpha, var_sigma_t), axis=None)
        elif len(var_alpha) == 0:
            var_surv = np.concatenate((var_gammas, var_d_alpha, var_sigma_t), axis=None)
        else:
            var_surv = np.concatenate((var_gammas, var_alpha, var_d_alpha, var_sigma_t), axis=None)
        var_coeffs = np.concatenate((var_long, var_surv))

        #var-cov of our thetas estimated
        #If hessian is singular, we choose to inverse it with pinv which can deal with it
        if np.linalg.det(hessian) != 0:
            vc_thetas = pd.DataFrame(np.linalg.inv(hessian), index=hessian.index, columns=hessian.columns)
        else:
            vc_thetas = pd.DataFrame(np.linalg.pinv(hessian), index=hessian.index, columns=hessian.columns)

        #Table with coeffs of our model, p-values etc
        coeffs_summary = pd.DataFrame(index=var_coeffs, columns=['value', 'sd', 'z-value', 'p-value'])

        #Filleng value of coeffs table
        for theta in coeffs_summary.index:
            coeffs_summary.loc[theta, 'value'] = (thetas[thetas_names == theta]).item()
            
        coeffs_summary['value'] = coeffs_summary['value'].astype('float64') #Reconverting value to numerical
        
        #Filling sd of coeffs table
        coeffs_summary['sd'] = np.array([np.sqrt(vc_thetas.loc[theta, theta]) for theta in var_coeffs])

        #Filling z_value of coeffs table
        coeffs_summary['z-value'] = coeffs_summary['value'] / coeffs_summary['sd']

        #Filling p_value of coeffs table
        p_val = 2 * (1 - norm.cdf(abs(coeffs_summary['z-value'])))
        coeffs_summary['p-value'] = [str(round(i,3)) if i > 0.001 else '< 0.001' for i in p_val]

        #Computation of AIC and BIC of our model
        k = hessian.shape[0]
        n = len(self.surv_data)
        aic = - 2 * log_lik + 2 * k
        bic = - 2 * log_lik + np.log(n) * k

        #final params
        self.prop = dict(
            log_lik = log_lik,
            aic = aic,
            bic = bic,
            residuals = sigma,
            var_surv = var_surv,
            var_gammas = var_gammas,
            var_long = var_long,
            coeffs = coeffs_summary,
            d_vc = d_vc,
            b = post_b,
            thetas = out['coefficients']['thetas'],
            thetas_skel = out['coefficients']['thetas_skel'],
            hessian = hessian,
            vc_thetas = vc_thetas,
            control = out['control'],
            success = out['success'],
            conv_message = out['conv_message'],
            log_lik_optim = out['log_lik_optim'],
            thetas_optim = out['thetas_optim']
        )










    def Summary(
        self
    ):

        #Infos about performances of model
        df_perf = pd.DataFrame({'Log lik':[self.prop['log_lik']], 'AIC':[self.prop['aic']], 'BIC':[self.prop['bic']]})
        p_perf = tabulate(df_perf, headers='keys', tablefmt='plain', showindex=False)

        #Longitudinal parameters
        df_long = self.prop['coeffs'].loc[self.prop['var_long']]
        p_long = tabulate(df_long, headers='keys', tablefmt='psql')

        #Survival parameters
        df_surv = self.prop['coeffs'].loc[self.prop['var_surv']]
        p_surv = tabulate(df_surv, headers='keys', tablefmt='psql')

        #Var cov on random effects
        #self.prop['d_vc']


        print(p_perf + '\n' + '\n' + '\n' + 'Longitudinal Parameters' + '\n' + p_long + '\n' + '\n' + '\n' + 'Survival Parameters' + '\n' + p_surv)










    def Survfit(
        self,
        new_data_baseline,
        new_data_long,
        id_var,
        surv_times = None,
        last_time = None,
        ci = np.array([0.025, 0.975]),
        M = 200,
        scale = 1.6,
        simulate = False,
        interest_evt = 'survival'
    ):
        
        #Verify if model is fitted
        if self.prop is None:
            raise ValueError("Joint model isn't fitted. Please fit a model with JointModel.fit() before")
        
        #method : For instant we only carry "weibull-PH-GH" method for parameter "method"
        accepted_method = ["weibull-PH-GH", "weibull-PH-QMC"]
        if self.method not in accepted_method:
            raise ValueError("Survfit only handle joint models fitted with method = {} for the moment".format(accepted_method))
        
        #Survival keys
        surv_keys = list(self.surv_object.keys())

        #Verify if interest event in input is in events
        if interest_evt not in surv_keys:
            raise ValueError("interest_evt must be in {}, you inputed '{}'".format(surv_keys, str(interest_evt)))

        #Implement surv_times if it's None and flatten it to have a vector if it's not None
        if surv_times is None:
            duration_col = self.surv_object[interest_evt].duration_col
            time = np.array(self.surv_data[duration_col])
            surv_times = np.linspace(time.min(), time.max() + 0.1, 35)
        else:
            #Verify if surv_times in input is valid
            if len(surv_times) == 0 or not np.issubdtype(surv_times.dtype, np.number):
                raise ValueError("surv_times argument must be None or Non empty numeric numpy array")
            #Flatten survtim to impose a vector
            surv_times = np.array(surv_times).flatten()

        #id_l : group id associated at each longitunal value
        id_l, groups_u = pd.factorize(new_data[id_var].values)

        #Model matrixes for fixed and random effects from associated formulas
        design_fe = patsy.dmatrices(self.lme_formula, new_data)
        design_re = patsy.dmatrix(self.lme_re_formula, new_data)

        #x_lme : fixed effects design matrix in linear mixed model
        x_lme = pd.DataFrame(design_fe[1], columns=design_fe[1].design_info.column_names)

        #z_lme : random effects design matrix in linear mixed model
        z_lme = pd.DataFrame(design_re, columns=design_re.design_info.column_names)

        #y_lme : response variable of linear mixed model
        y_lme = np.array(design_fe[0])

        #data_id : Last longitudinal markers for each group
        data_id = new_data.groupby(id_var).tail(1)
        id_keys = np.array(data_id[id_var])

        w = dict()
        for risk in surv_keys:
            design_w = patsy.dmatrix(self.surv_object[risk].formula, data_id)
            w[risk] = pd.DataFrame(design_w, columns = design_w.design_info.term_names)
            w[risk] = np.array(w[risk])[np.newaxis] if w[risk].ndim == 1 else np.array(w[risk])
        
        mat_1 = np.repeat(np.array([1]), len(data_id))
        wint_f_vl = mat_1.copy()
        wint_f_sl = mat_1.copy()

        #obs_times : observation times for each group
        gb = new_data.groupby(id_var)
        obs_times = {idx : np.array(gb.get_group(idx)[self.time_var]) for idx in gb.groups}
        
        #obs_times_surv : Time of last longitudinal observation for each group
        obs_times_surv = {idx : np.array(data_id[data_id[id_var] == idx][self.time_var]) for idx in id_keys}
        
        #last_time : Landmark time t of prediction (time from which prediction is made)
        if last_time is None:
            last_time = obs_times_surv
        elif isinstance(last_time, str):
            try:
                last_time_col = last_time
                last_time = {idx : np.array(data_id[data_id[id_var] == idx][last_time_col]) for idx in id_keys}
            except:
                raise ValueError("Not appropriated value for 'last_time' argument. Must be a numeric vector, a character string referring to a 'new_data' column or None value")
        elif isinstance(last_time, np.ndarray):
            if np.issubdtype(last_time.dtype, np.number):
                last_time_vec = last_time
                if len(last_time_vec) != len(id_keys):
                    raise ValueError("If 'last_time' argument is a vector, it must have the same length than you have groups in 'new_data'")
                last_time = dict()
                last_time = {id_keys[idx] : np.array(last_time_vec[idx]) for idx in range(len(id_keys))}
            else:
                raise ValueError("Not appropriated value for 'last_time' argument. Must be a numeric vector, a character string referring to a 'new_data' column or None value")            
        else:
            raise ValueError("Not appropriated value for 'last_time' argument. Must be a numeric vector, a character string referring to a 'new_data' column or None value")
        
        #pred_times : Times of predictions
        pred_times = {idx : surv_times[surv_times > last_time[idx]] for idx in last_time.keys()}

        #Number of groups in new_data
        n_tp = len(id_keys)

        #Number of random effects
        ncz = len(z_lme.columns)

        #Parameters of fitted model
        thetas = np.array(self.prop['thetas']).flatten()
        thetas_skel = self.prop['thetas_skel']
        betas = thetas[thetas_skel['pos_betas']]
        sigma = self.prop['residuals']
        d_vc = self.prop['d_vc']
        if self.prop['d_vc'].ndim == 1:
            d_vc = np.zeros((len(d_vc), len(d_vc)))
            np.fill_diagonal(d_vc, self.prop['d_vc'])
        else:
            d_vc = self.prop['d_vc']
        gammas = {risk : thetas[thetas_skel['pos_gammas'][risk]] for risk in surv_keys}
        alpha_value = {risk : thetas[thetas_skel['pos_alpha'][risk]] for risk in surv_keys} if self.parametrization in ['value', 'both'] else None
        alpha_slope = {risk : thetas[thetas_skel['pos_d_alpha'][risk]] for risk in surv_keys} if self.parametrization in ['slope', 'both'] else None
        sigma_t = {risk : np.exp(thetas[thetas_skel['pos_sigma_t'][risk]]) for risk in surv_keys}

        #Preparation of our design matrixes
        surv_mats = dict()
        surv_mats_last = dict()
        for i in range(n_tp):
            surv_mats[id_keys[i]] = [Model_mats(self, data_id, x, i, wint_f_vl, wint_f_sl) for x in pred_times[id_keys[i]]]
            surv_mats_last[id_keys[i]] = Model_mats(self, data_id, last_time[id_keys[i]], i, wint_f_vl, wint_f_sl)

        modes_b = np.empty((0, ncz))
        vars_b = dict()

        log_posterior_b_dict_args = dict(
            y_lme = y_lme,
            x_lme = x_lme,
            z_lme = z_lme,
            mats = surv_mats_last,
            w = w,
            method = self.method,
            parametrization = self.parametrization,
            derivForm = self.derivForm,
            betas = betas,
            sigma = sigma,
            d_vc = d_vc,
            gammas = gammas,
            alpha_value = alpha_value,
            alpha_slope = alpha_slope,
            sigma_t = sigma_t,
            surv_keys = surv_keys,
            id_l = id_l,
            ncz = ncz,
            f = Log_posterior_b,
            eps = 1e-03,
        )
        
        for i in range(n_tp):

            log_posterior_b_dict_args['i'] = i
            
            try:
                out = minimize(
                    fun = Log_posterior_b,
                    x0 = np.zeros(ncz),
                    method = 'BFGS',
                    options = {'gtol':1e-10, 'norm':-np.inf, 'eps':1e-03},
                    args = (log_posterior_b_dict_args)
                )
            except:
                out = minimize(
                    fun = Log_posterior_b,
                    x0 = np.zeros(ncz),
                    method = 'BFGS',
                    options = {'gtol':1e-10, 'norm':-np.inf, 'eps':1e-03},
                    args = (log_posterior_b_dict_args),
                    jac = Cd
                )
            modes_b = np.append(modes_b, [out.x], axis=0)
            vars_b[id_keys[i]] = out.hess_inv * scale
            

        if simulate:

            #Preparation before itration
            success_rate =  np.full((M, n_tp), False)
            b_old = modes_b.copy()
            b_new = modes_b.copy()

            log_posterior_b_dict_args = dict(
                y_lme = y_lme,
                x_lme = x_lme,
                z_lme = z_lme,
                mats = surv_mats_last,
                w = w,
                method = self.method,
                parametrization = self.parametrization,
                derivForm = self.derivForm,
                id_l = id_l,
                ncz = ncz,
                surv_keys = surv_keys
            )

            #Iterations
            out = dict()
            for m in tqdm(range(M)):

                #Simulation of new parameters value simulated by an multivariated normal distribution MVN(thetas, cov(thetas))
                thetas_new = multivariate_normal.rvs(mean = np.array(self.prop['thetas']).squeeze(), cov = self.prop['vc_thetas'], size = 1)

                #Thetas subdivised
                thetas_skel = self.prop['thetas_skel']
                betas_new = thetas_new[thetas_skel['pos_betas']]
                sigma_new = np.exp(thetas_new[thetas_skel['pos_sigma']])
                gammas_new = {risk : thetas_new[thetas_skel['pos_gammas'][risk]] for risk in surv_keys}
                alpha_new = {risk : thetas_new[thetas_skel['pos_alpha'][risk]] for risk in surv_keys} if thetas_skel['pos_alpha'] is not None else None
                d_alpha_new = {risk : thetas_new[thetas_skel['pos_d_alpha'][risk]] for risk in surv_keys} if thetas_skel['pos_d_alpha'] is not None else None
                d_vc_new = thetas_new[thetas_skel['pos_d_vc']]
                d_vc_new = np.exp(d_vc_new) if d_vc.ndim == 1 else Chol_transf(d_vc_new)['res']

                if self.method in ['weibull-PH-GH']:
                    sigma_t_new = {risk : np.exp(thetas_new[thetas_skel['pos_sigma_t'][risk]]) if self.scale_wb is None else self.scale_wb for risk in surv_keys}
                
                ss = dict()
                log_posterior_b_dict_args['betas'] = betas_new
                log_posterior_b_dict_args['sigma'] = sigma_new
                log_posterior_b_dict_args['d_vc'] = d_vc_new
                log_posterior_b_dict_args['gammas'] = gammas_new
                log_posterior_b_dict_args['alpha_value'] = alpha_new
                log_posterior_b_dict_args['alpha_slope'] = d_alpha_new
                log_posterior_b_dict_args['sigma_t'] = sigma_t_new
                
                #def Pred_iter(i):
                for i in range(n_tp):

                    log_posterior_b_dict_args['i'] = i

                    #Simulation of random effects
                    proposed_b = Rmvt(1, modes_b[i], vars_b[id_keys[i]], 4)
                    dmvt_old = Dmvt(b_old[i], modes_b[i], vars_b[id_keys[i]], 4, True)
                    dmvt_proposed = Dmvt(proposed_b, modes_b[i], vars_b[id_keys[i]], 4, True)
                    a = min(np.exp(- Log_posterior_b(proposed_b, log_posterior_b_dict_args) + dmvt_old + Log_posterior_b(b_old[i], log_posterior_b_dict_args) - dmvt_proposed), 1)
                    success_rate[m, i] = uniform.rvs() <= a
                    if uniform.rvs() <= a:
                        b_new[i] = proposed_b
                    
                    #Compute Pr(T > t_k | T > t_{k - 1}; theta.new, b.new)
                    s_last = S_b(
                        jm_obj = self,
                        t = last_time[id_keys[i]],
                        b = b_new[i],
                        i = i,
                        mats = surv_mats_last[id_keys[i]],
                        betas = betas_new,
                        gammas = gammas_new,
                        alpha_value = alpha_new,
                        alpha_slope = d_alpha_new,
                        sigma_t = sigma_t_new,
                        w = w,
                        surv_keys = surv_keys
                    )
                    
                    s_pred = np.array([])
                    for id_p in range(len(pred_times[id_keys[i]])):
                        s_pred_id_p = S_b(
                            jm_obj = self,
                            t = pred_times[id_keys[i]][id_p],
                            b = b_new[i],
                            i = i,
                            mats = surv_mats[id_keys[i]][id_p],
                            betas = betas_new,
                            gammas = gammas_new,
                            alpha_value = alpha_new,
                            alpha_slope = d_alpha_new,
                            sigma_t = sigma_t_new,
                            w = w,
                            surv_keys = [interest_evt]
                        )
                        s_pred = np.append(s_pred, s_pred_id_p)
                    
                    ss[id_keys[i]] = s_pred/s_last
                b_old = b_new
                out[m] = ss

                
                    
            res = dict()
            for i in range(n_tp):
                rr = np.array([list(out[idx][id_keys[i]]) for idx in out.keys()])
                res[id_keys[i]] = pd.DataFrame(
                    {
                        'time':pred_times[id_keys[i]],
                        'mean':rr.mean(axis=0),
                        'median':np.quantile(rr, 0.5, axis=0),
                        'lower':np.quantile(rr, ci[0], axis=0),
                        'upper':np.quantile(rr, ci[1], axis=0)
                    }
                )

        else:
            res = dict()
            for i in range(n_tp):
                s_last = S_b(
                    jm_obj = self,
                    t = last_time[id_keys[i]],
                    b = modes_b[i],
                    i = i,
                    mats = surv_mats_last[id_keys[i]],
                    betas = betas,
                    gammas = gammas,
                    alpha_value = alpha_value,
                    alpha_slope = alpha_slope,
                    sigma_t = sigma_t,
                    w = w,
                    surv_keys = surv_keys
                )
                s_pred = np.array([])
                for id_p in range(len(pred_times[id_keys[i]])):
                    s_pred_id_p = S_b(
                        jm_obj = self,
                        t = pred_times[id_keys[i]][id_p],
                        b = modes_b[i],
                        i = i,
                        mats = surv_mats[id_keys[i]][id_p],
                        betas = betas,
                        gammas = gammas,
                        alpha_value = alpha_value,
                        alpha_slope = alpha_slope,
                        sigma_t = sigma_t,
                        w = w,
                        surv_keys = [interest_evt]
                    )
                    s_pred = np.append(s_pred, s_pred_id_p)
                res[id_keys[i]] = pd.DataFrame({'time':pred_times[id_keys[i]], 'pred_surv':s_pred/s_last})

        return(res)

    def Survfit_CIF(
            self,
            new_data_baseline,
            new_data_long,
            id_var,
            surv_times = None,
            last_time = None,
            ci = np.array([0.025, 0.975]),
            M = 200,
            simulate = False,
            interest_evt = 'survival'
        ):
        """Function which computes Cumulative Incidence Function of experiencing event between landmark time "last_time" and the horizon time "surv_times"  given the probability of not having
        experienced event before

        Args:
            new_data_baseline (pd.DataFrame): Table containing baseline data for patient for which prediction is computed, columns must have the same name formula used for fitting model. See JM.formula
            new_data_long (dictionnary): Dictionnary containing tables of longitudinal follow-up for patient available at landmark time. Each key of the dictionnary is corresponding to a longitudinal key (each item of dict corresponding to one longitudinal variable)
            id_var (str): Name of primary key permitting merging baseline data and long data
            surv_times (np.array): Array containg horizon times for which predictions are computed
            last_time (np.array, optional): Array of one element containing last time subject is known to beeing event-free. Defaults to None, in this case, last time of observation in longitudinal data will be taken.
            ci (np.array, optional): Array of two elements specifying confidence interval around prediction np.array([lower_bound, upperbound]). Defaults to np.array([0.025, 0.975]).
            M (scalar, optional): Number of Monte-Carlo points for computing prediction and confidence interval. Car be ignored if simulate = False. Defaults to 200.
            simulate (bool, optional): If True, prediction will be computed through a Monte-Carlo process giving mean, median and confidence interval of simulated predictions. If False, prediction is computed by maximizing posterior distribution of random effects. Defaults to False.
            interest_evt (str, optional): Name of event for wich prediction is computed, must be specified in competing risk setting. Defaults to 'survival'.

        Returns:
            pd.DataFrame: Predictions with subject primary key, landmarks times, horizon times and predictions (['pi_evt'] if simulate = False and ['mean', 'median', 'lower', 'upper'] if simulate = True)
        """

        #Verify if model is fitted
        if self.prop is None:
            raise ValueError("Joint model isn't fitted. Please fit a model with JointModel.fit() before")

        #Verify if interest event in input is in events
        if interest_evt not in self.surv_keys:
            raise ValueError("interest_evt must be in {}, you inputed '{}'".format(self.surv_keys, str(interest_evt)))
        
        #Sorting baseline data by id to have same order of id between baseline and longitudinal data
        new_data_baseline.sort_values(id_var, inplace=True)

        #Raise an error if there are nans values is baseline new data
        if new_data_baseline.isna().values.any():
            raise ValueError("new_data_baseline contains missing values")

        
        #id_l : group id associated at each longitunal value
        id_l = {}

        #Manipuation on longitudinal new data
        for key in self.long_keys:
            
            #Reset index to avoid duplicated index
            new_data_long[key].reset_index(inplace=True, drop=True)

            #Sorting longitudinal data by id to have same order of id between baseline and longitudinal data
            new_data_long[key].sort_values(id_var, inplace=True)

            #id_l : group id associated at each longitunal value
            id_l[key], groups_u = pd.factorize(new_data_long[key][id_var].values)

            #Raise error if there is a missmatch between subjects in longitudinal and baseline new data
            if (groups_u != new_data_baseline[id_var].values).any():
                raise ValueError("Missmatch between new_data_baseline and new_data_long subjects")
            
            #Raise an error if there are nans values is baseline new data
            if new_data_long[key].isna().values.any():
                raise ValueError("new_data_long[{}] contains missing values".format(key))
        
        #List of all subjects
        subjects = new_data_baseline[id_var].values

        #last_time : Landmark time t of prediction (time from which prediction is made)
        if last_time is None:
            #Time of last longitudinal observation for each subject
            last_time = np.array([np.concatenate([new_data_long[key].loc[new_data_long[key][id_var] == subject, self.time_var[key]].values for key in self.long_keys]).max() for subject in subjects])
        elif isinstance(last_time, np.ndarray) and np.issubdtype(last_time.dtype, np.number):
            if len(last_time) != len(subjects):
                raise ValueError("If 'last_time' argument is a numpy array, it must have value for each subject in 'new_data_baseline'")
        else:
            raise ValueError("Not appropriated value for 'last_time' argument. Numeric vector or None is expected")

        #Implement surv_times if it's None
        if surv_times is None:
            #If none : 35 predictions times between min and max + 0.1 dropout times observed on learning sample
            duration_col = self.surv_object[interest_evt].duration_col
            times_learn = np.array(self.surv_data[duration_col])
            surv_times = np.linspace(times_learn.min(), times_learn.max() + 0.1, 35)
        else:
            #Verify if surv_times in input is valid
            if len(surv_times) == 0 or not np.issubdtype(surv_times.dtype, np.number):
                raise ValueError("surv_times argument must be None or Non empty numeric numpy array")
            elif surv_times.ndim != 1:
                raise ValueError("surv_times numpy array must be a 1D vector")
            #Flatten survtime to impose a vector
            surv_times = np.array(surv_times).flatten()
        
        #pred_times : Times of predictions (superiors at last observation for each patient)
        pred_times = {subjects[idx] : surv_times[surv_times > last_time[idx]] for idx in range(len(last_time))}
  
        #Extraction of baseline covariates
        w = {risk: np.array(patsy.dmatrix(self.surv_object[risk].formula, new_data_baseline)) for risk in self.surv_keys}
        
        #Extraction of longitudinal observations
        design_fe = {key: patsy.dmatrices(self.lme_formula[key], new_data_long[key]) for key in self.long_keys}
        design_re = {key: patsy.dmatrix(self.lme_re_formula[key], new_data_long[key]) for key in self.long_keys}
        
        #x_lme : fixed effects design matrix in linear mixed model
        x_lme = {key: np.array(design_fe[key][1]) for key in self.long_keys}

        #z_lme : random effects design matrix in linear mixed model
        z_lme = {key: np.array(design_re[key]) for key in self.long_keys}

        #ncz_dict: Number of random effects per longitudinal biomarker
        ncz_dict = Dapply(z_lme, lambda x: x.shape[1], self.long_keys)

        #y_lme : response variable of linear mixed model
        y_lme = {key: np.array(design_fe[key][0]).flatten() for key in self.long_keys}

        #Parameters of fitted model
        thetas = np.array(self.prop['thetas']).flatten()
        thetas_skel = self.prop['thetas_skel']

        #Extract thetas fitted parameters
        _, pos_betas, pos_sigma, pos_gammas, pos_alpha, pos_d_alpha, pos_sigma_t, pos_d_vc = Dict_skel(thetas_skel)
        family_long = {key: 'gaussian' if pos_sigma[key] is not None else 'bernoulli' for key in self.long_keys}#Family of each longitudinal biomarker
        betas = {key: thetas[pos_betas[key]] for key in self.long_keys}
        sigma = {key: np.exp(thetas[pos_sigma[key]]) if family_long[key] == 'gaussian' else None for key in self.long_keys}
        gammas = {risk: thetas[pos_gammas[risk]] for risk in self.surv_keys}
        alpha = {risk: {key: thetas[pos_alpha[risk][key]] if pos_alpha[risk][key] is not None else None for key in self.long_keys} for risk in self.surv_keys}
        d_alpha = {risk: {key: thetas[pos_d_alpha[risk][key]] if pos_d_alpha[risk][key] is not None else None for key in self.long_keys} for risk in self.surv_keys}
        sigma_t = {risk: np.exp(thetas[pos_sigma_t[risk]]) for risk in self.surv_keys}
        d_vc = Chol_transf(thetas[pos_d_vc])['res']

     
        log_posterior_b_dict_args = {
            'y_lme': y_lme,
            'x_lme': x_lme,
            'w': w,
            'method': self.method,
            'parametrization': self.parametrization,
            'derivForm': self.derivForm,
            'betas': betas,
            'sigma': sigma,
            'd_vc': d_vc,
            'gammas': gammas,
            'alpha_value': alpha,
            'alpha_slope': d_alpha,
            'sigma_t': sigma_t,
            'surv_keys': self.surv_keys,
            'long_keys': self.long_keys,
            'id_l': id_l,
            'last_time':last_time,
            'f': Log_posterior_b,
            'family_long': family_long,
            'ncz_dict': ncz_dict
        }

        #Dataframe summarising dynamic probabilities
        summary_predict = pd.DataFrame()

        for i,subject in enumerate(subjects):

            log_posterior_b_dict_args['i'] = i
            out = minimize(fun = Log_posterior_b, x0 = np.zeros(sum(ncz_dict.values())), method = 'BFGS', args = (log_posterior_b_dict_args))
            modes_b = out.x
            cov_b = out.hess_inv
            
            if simulate:

                log_posterior_b_dict_args['i'] = i

                #Preparation before itration
                success_rate =  np.full((M, len(subjects)), False)
                b_old = modes_b.copy()
                b_new = modes_b.copy()
                prob = np.zeros((M, len(pred_times[subject])))

                #Simulation of thetas parameters value simulated by an multivariate normal distribution MVN(thetas, cov(thetas))
                thetas_simul = multivariate_normal.rvs(mean = self.prop['thetas'].values.squeeze(), cov = self.prop['vc_thetas'], size = M)
                #thetas_simul = qmc.MultivariateNormalQMC(mean = self.prop['thetas'].values.squeeze(), cov = self.prop['vc_thetas']).random(M)

                for m in tqdm(range(M), disable=True):
                    betas_new = {key: thetas_simul[m, pos_betas[key]] for key in self.long_keys}
                    sigma_new = {key: np.exp(thetas_simul[m, pos_sigma[key]]) if family_long[key] == 'gaussian' else None for key in self.long_keys}
                    gammas_new = {risk: thetas_simul[m, pos_gammas[risk]] for risk in self.surv_keys}
                    alpha_new = {risk: {key: thetas_simul[m, pos_alpha[risk][key]] if pos_alpha[risk][key] is not None else None for key in self.long_keys} for risk in self.surv_keys}
                    d_alpha_new = {risk: {key: thetas_simul[m, pos_d_alpha[risk][key]] if pos_d_alpha[risk][key] is not None else None for key in self.long_keys} for risk in self.surv_keys}
                    sigma_t_new = {risk: np.exp(thetas_simul[m, pos_sigma_t[risk]]) for risk in self.surv_keys}
                    d_vc_new = Chol_transf(thetas_simul[m, pos_d_vc])['res']

                    log_posterior_b_dict_args['betas'] = betas_new
                    log_posterior_b_dict_args['sigma'] = sigma_new
                    log_posterior_b_dict_args['d_vc'] = d_vc_new
                    log_posterior_b_dict_args['gammas'] = gammas_new
                    log_posterior_b_dict_args['alpha_value'] = alpha_new
                    log_posterior_b_dict_args['alpha_slope'] = d_alpha_new
                    log_posterior_b_dict_args['sigma_t'] = sigma_t_new

                    #Metropolis-Hasting algotihm for simulation of b
                    proposed_b = Rmvt(1, modes_b, cov_b, 4)
                    dmvt_old = Dmvt(b_old, modes_b, cov_b, 4, True)
                    dmvt_proposed = Dmvt(proposed_b, modes_b, cov_b, 4, True)
                    a = min(np.exp(- Log_posterior_b(proposed_b, log_posterior_b_dict_args) + dmvt_old + Log_posterior_b(b_old, log_posterior_b_dict_args) - dmvt_proposed), 1)
                    random_u = uniform.rvs()
                    success_rate[m, 0] = random_u <= a
                    if random_u <= a:
                        b_new = proposed_b

                    #Adding random effects to betas parameters
                    begin_idx = 0
                    betas_b = {}
                    for key in self.long_keys:
                        ncz_key = ncz_dict[key]
                        end_idx = begin_idx+ncz_key
                        betas_b[key] = betas_new[key] + b_new[begin_idx:end_idx]
                        begin_idx = end_idx

                    #Pos of betas associated with time, works only for derivation of mi(t) = B0 + b0i + (B1 + b1i) * t
                    ind_fixed = 1

                    g_args = (betas_b, self.parametrization, alpha_new, d_alpha_new, ind_fixed, family_long)

                    #Probability to not even have experienced any outcome until landmark time t
                    s_t = Survival(
                        t=last_time[i],
                        w=w,
                        gammas=gammas_new,
                        sigma_t=sigma_t_new,
                        i=i,
                        surv_keys=self.surv_keys,
                        g_args=g_args
                    )

                    #Incidence of event of interest between landmark time t and prediction time u
                    cif_t_u = CIF(
                        t=last_time[i],
                        u=pred_times[subject],
                        w=w,
                        gammas=gammas_new,
                        sigma_t=sigma_t_new,
                        i=i,
                        surv_keys=self.surv_keys,
                        risk=interest_evt,
                        g_args=g_args
                    )

                    #Probability to observe event of interest between landmarks time t and prediction time u
                    prob[m] = min(cif_t_u/s_t, np.array([1.0])) #Rare case when because of gaussian quadrature approximation the result exceed 1
                    b_old = b_new

                res = pd.DataFrame(
                    {
                        'subject':subject,
                        't':last_time[i],
                        'u':pred_times[subject],
                        'mean':prob.mean(axis=0),
                        'median':np.quantile(prob, 0.5, axis=0),
                        'lower':np.quantile(prob, ci[0], axis=0),
                        'upper':np.quantile(prob, ci[1], axis=0)
                    }
                )

                summary_predict = pd.concat([summary_predict, res], ignore_index=True)

            else:

                #Adding random effects to betas parameters
                begin_idx = 0
                betas_b = {}
                for key in self.long_keys:
                    ncz_key = ncz_dict[key]
                    end_idx = begin_idx+ncz_key
                    betas_b[key] = betas[key] + modes_b[begin_idx:end_idx]
                    begin_idx = end_idx

                #Pos of betas associated with time, works only for derivation of mi(t) = B0 + b0i + (B1 + b1i) * t
                ind_fixed = 1

                g_args = (betas_b, self.parametrization, alpha, d_alpha, ind_fixed, family_long)

                #Probability to not even have experienced any outcome until landmark time t
                s_t = Survival(
                    t=last_time[i],
                    w=w,
                    gammas=gammas,
                    sigma_t=sigma_t,
                    i=i,
                    surv_keys=self.surv_keys,
                    g_args=g_args
                )

                #Incidence of event of interest between landmark time t and prediction time u
                cif_t_u = CIF(
                    t=last_time[i],
                    u=pred_times[subject],
                    w=w,
                    gammas=gammas,
                    sigma_t=sigma_t,
                    i=i,
                    surv_keys=self.surv_keys,
                    risk=interest_evt,
                    g_args=g_args
                )

                #Probability to observe event of interest between landmarks time t and prediction time u
                prob = min(cif_t_u/s_t, np.array([1.0])) #Rare case when because of gaussian quadrature approximation the result exceed 1

                res = pd.DataFrame(
                    {
                        'subject':subject,
                        't':last_time[i],
                        'u':pred_times[subject],
                        'pi_evt':prob
                    }
                )

                summary_predict = pd.concat([summary_predict, res], ignore_index=True)

        
        

        return summary_predict