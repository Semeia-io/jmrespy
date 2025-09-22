####################### -------------------- Import libraries -------------------- #######################

# Usual libraries
import pandas as pd
import numpy as np

# PH and GLMM model fitters
from lifelines import CoxPHFitter
from statsmodels.formula.api import mixedlm
from bambi import Model

# Vizualiation for bambi model results
import arviz as az

# JM fitter
from jmrespy.jointModel import JointModel

# Others
import joblib
from functools import reduce





####################### -------------------- Data preprocessing -------------------- #######################

# Import data
data_id = pd.read_csv('jmrespy/data/pbc_baseline.csv')
data_long = pd.read_csv('jmrespy/data/pbc_long.csv')

# Longitudinal marker 1 : log Serum Bilirubin
data_long['log_serBilir'] = np.log(data_long['serBilir'])

# Longitrudinal marker 2 : Binary histologic status (4 vs 1/2/3)
data_long['histologic_bin'] = np.where(data_long['histologic'] > 3, 1, 0)

data_long['histologic_bin'].isna().sum()
data_long.loc[data_long['id'] == 23, ['id', 'histologic', 'histologic_bin', 'year', 'years', 'status']]
data_id['status'].value_counts()
data_long.columns

# Suppression of missing values for glmms (no missing values in this dataset, but this step must be performed in other analysis)
data_glmm_log_serBilir = data_long.dropna(
    subset=[
        'id',
        'years',
        'log_serBilir'
    ]
)

data_glmm_histologic_bin = data_long.dropna(
    subset=[
        'id',
        'years',
        'histologic_bin'
    ]
)

# Suppression of patients without at least one observation for each longitudinal markers (no cases in this dataset, but this step must be performed in other analysis)
id_inter = reduce(
    np.intersect1d,
    (data_glmm_log_serBilir['id'].unique(), data_glmm_histologic_bin['id'].unique(), data_id['id'])
)
data_PH = data_id[data_id['id'].isin(id_inter)]
data_glmm_log_serBilir = data_glmm_log_serBilir[data_glmm_log_serBilir['id'].isin(id_inter)]
data_glmm_histologic_bin = data_glmm_histologic_bin[data_glmm_histologic_bin['id'].isin(id_inter)]

# Recoding of patient outcomes : competing event 1 (death)
data_PH['dead'] = np.where(data_PH['status'] == 'dead', 1, 0)

# Recoding of patient outcomes : competing event 2 (transplanted)
data_PH['transplanted'] = np.where(data_PH['status'] == 'transplanted', 1, 0)

# Recoding of categorical baseline predictors (one-hot encoding without reference modality)
data_PH['sex_f'] = np.where(data_PH['sex'] == 'female', 1, 0)
data_PH['edema_1'] = np.where(data_PH['edema'] == 'edema no diuretics', 1, 0)
data_PH['edema_2'] = np.where(data_PH['edema'] == 'edema despite diuretics', 1, 0)

# Suppression of missing values for propotional hazrd models (no missing values in this dataset, but this step must be performed in other analysis)
data_PH = data_PH[
    [
     'id',
     'age',
     'sex_f',
     'edema_1',
     'edema_2',
     'transplanted',
     'dead',
     'years',
     ]
].dropna()
data_glmm_log_serBilir = data_glmm_log_serBilir[data_glmm_log_serBilir['id'].isin(data_PH['id'])]
data_glmm_histologic_bin = data_glmm_histologic_bin[data_glmm_histologic_bin['id'].isin(data_PH['id'])]

# Storing longitudinal datasets in a dictionnary passed in JointModel (longitudinal key names are important and must be matched in other dictionnary defined latter in the script)
data_glmm = {
    'serBilir': data_glmm_log_serBilir,
    'histologic': data_glmm_histologic_bin
}



####################### -------------------- Modelization -------------------- #######################

### Longitudinal markers ---

# glmm for log Serum Bilirubin
re_f_1 = '~ year'
glmm_1 = mixedlm(
    'log_serBilir ~ 1 + year',
    data_glmm_log_serBilir,
    groups = 'id',
    re_formula = re_f_1
)
fitted_glmm_1 = glmm_1.fit(reml=False)
fitted_glmm_1.summary()

# glmm for Binary histologic status
re_f_2 = '~ year'
glmm_2 = Model(
    'histologic_bin ~ 1 + year + (1+ year|id)',
    data_glmm_histologic_bin,
    family = 'bernoulli'
)
fitted_glmm_2 = glmm_2.fit(draws=1000, tune=1000, chains=None, cores=None)
az.summary(fitted_glmm_2)

# Storing models in a dictionnary passed in JointModel
long_mod = {
    # Instanciated models (longitudinal key names must match with other dictionnaries)
    'glmm': {
        'serBilir': glmm_1,
        'histologic': glmm_2
    },
    # Fitted models (longitudinal key names must match with other dictionnaries)
    'fitted': {
        'serBilir': fitted_glmm_1,
        'histologic': fitted_glmm_2
    },
    # Formulas
    'formula': {
        # Fixed effects (longitudinal key names must match with other dictionnaries)
        'fe': {
            'serBilir': glmm_1.formula,
            'histologic': 'histologic_bin ~ 1 + year'
        },
        # Random effects (longitudinal key names must match with other dictionnaries)
        're': {
            'serBilir': re_f_1,
            'histologic': re_f_2
        }
    }
}

### Survival outcomes ---

# Cause-specific PH model for death
cph_dead = CoxPHFitter()
cph_dead.fit(
    data_PH,
    duration_col = 'years',
    event_col = 'dead',
    formula = 'age + sex_f + edema_1 + edema_2'
)
cph_dead.summary

# Cause-specific PH model for transplantation
cph_transplant = CoxPHFitter()
cph_transplant.fit(
    data_PH,
    duration_col = 'years',
    event_col = 'transplanted',
    formula = 'age'
)
cph_transplant.summary

### Joint model for survival and longitudinal data ---

# derivative form (longitudinal key names must match with other dictionnaries)
dform = {
    'serBilir': {'fixed': "~1", 'ind_fixed': [1], 'random': "~1", 'ind_random': [1]},
    'histologic': {'fixed': "~1", 'ind_fixed': [1], 'random': "~1", 'ind_random': [1]},
}

# Initiation parameters
init_dict = None

# Model instanciation
model_JM = JointModel(
    lme_object = long_mod['glmm'],
    lme_object_fitted = long_mod['fitted'],
    lme_data = data_glmm,
    lme_formula = long_mod['formula']['fe'],
    lme_re_formula = long_mod['formula']['re'],
    surv_object = {'dead': cph_dead, 'transplant': cph_transplant},
    surv_data = data_PH,
    time_var = {'serBilir': "year", 'histologic': "year"},
    parametrization = {'serBilir': 'value', 'histologic': 'value'},
    method = "weibull-PH-QMC",
    interfact = None,
    derivForm = dform,
    lag = 0,
    scale_wb = None,
    competing_risks = 'cs',
    init = init_dict,
    control = {
        "iter_em": 0,
        "eps_hes": 1e-3,
        "options_minimize": {
            "maxfun": 150000,
            "iprint": 99
        },
        "solver_minimize": "L-BFGS-B",
        "nbmc": 8192,
        "n_cores_grad": 6,
        "options_noise_minimize": {
            "terminate": 1,
            "display": 3,
            "max_iter": 100000,
            "max_geval": 800000,
            "max_ls_fails": 0
        }
    }
)

# Model fitting
model_JM.fit()

# Model summary
model_JM.Summary()

# Export fitted JM (For now, the model can't be exported with lme_object module, but its dispensable once the model is fitted)
del model_JM.lme_object
joblib.dump(model_JM, 'fitted_JM_pbc.sav')