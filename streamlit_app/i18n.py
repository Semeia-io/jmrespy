T = {
    "app_title": {"fr": "DynKiGraft", "en": "DynKiGraft"},
    "app_sub": {
        "fr": "Greffe rénale · modèle conjoint longitudinal–survie · jmrespy · Sêmeia Datalab",
        "en": "Kidney transplant · joint longitudinal–survival model · jmrespy · Sêmeia Datalab",
    },
    "ruo": {
        "fr": "<b>Research Use Only.</b> Outil de recherche, non destiné au diagnostic ni à la décision "
        "thérapeutique. Ne saisissez aucune donnée identifiante. Modèle Dynkigraft (rejet / échec de greffe rénale), "
        "non validé pour un usage clinique.",
        "en": "<b>Research Use Only.</b> Research tool, not intended for diagnosis or treatment decisions. "
        "Do not enter identifying data. Dynkigraft model (kidney graft rejection / failure), not validated for "
        "clinical use.",
    },
    "abstract_title": {"fr": "Résumé", "en": "Abstract"},
    "abstract_md": {
        "fr": "DynKiGraft fournit des prédictions dynamiques du risque d'échec du greffon rénal (retour en "
        "dialyse) sur un horizon glissant de 5 ans, en traitant le décès avec greffon fonctionnel comme risque "
        "concurrent. Ce modèle a été développé sur la cohorte française DIVAT à l'aide d'un modèle conjoint à effets aléatoires "
        "partagés. Il relie trois marqueurs longitudinaux — créatinine sérique, protéinurie et DSA anti-HLA de "
        "classe II — aux covariables disponibles à la greffe, et met à jour la prédiction à chaque nouvelle "
        "mesure. Disponible dès 1 an post greffe, la prédiction devient cependant plus précise au fil des années (AUC de 0,72 à 1 an jusqu'à 0,86 "
        "à 9 ans post-greffe).",
        "en": "DynKiGraft provides dynamic predictions of kidney graft failure (return to dialysis) over a "
        "sliding 5-year horizon, treating death with a functioning graft as a competing event. Developed on the "
        "French DIVAT cohort with a shared random-effects joint model, it links three longitudinal markers — "
        "serum creatinine, proteinuria and anti-HLA class II DSA — to covariates available at transplantation, "
        "and updates the prediction with each new measurement. On validation, discrimination improves along "
        "follow-up (AUC from 0.72 at 1 year to 0.86 at 9 years post-transplant).",
    },
    "patient_ref": {"fr": "Patient de référence (manuscrit)", "en": "Reference patient (manuscript)"},
    "current_curve": {"fr": "Saisie actuelle", "en": "Current input"},
    "ref_curve": {"fr": "État par défaut (papier)", "en": "Default state (paper)"},
    "sec_patient": {"fr": "Patient", "en": "Patient"},
    "sec_followup": {"fr": "Suivi longitudinal", "en": "Longitudinal follow-up"},
    "sec_pred": {"fr": "Paramètres de prédiction", "en": "Prediction settings"},
    "patient_help": {
        "fr": "Covariables de baseline, en unités cliniques (âge en années, créatinine en µmol/L).",
        "en": "Baseline covariates, in clinical units (age in years, creatinine in µmol/L).",
    },
    "binaries": {"fr": "Antécédents et évènements (oui / non)", "en": "History and events (yes / no)"},
    "followup_help": {
        "fr": "Une ligne par visite. Temps en années depuis M12. Créatinine en µmol/L, protéinurie en g/L, "
        "DSA anti-HLA II > 2000 à 0 ou 1. Une cellule vide = marqueur non mesuré à cette visite.",
        "en": "One row per visit. Time in years since M12. Creatinine in µmol/L, proteinuria in g/L, anti-HLA II "
        "DSA > 2000 as 0 or 1. Empty cell = marker not measured at that visit.",
    },
    "time": {"fr": "Temps (ans)", "en": "Time (yrs)"},
    "event": {"fr": "Évènement d'intérêt", "en": "Event of interest"},
    "landmark": {"fr": "Date de prédiction t (années)", "en": "Prediction time t (years)"},
    "landmark_help": {
        "fr": "Seules les visites jusqu'à t sont utilisées. Déplacez t pour voir la prédiction se mettre à jour.",
        "en": "Only visits up to t are used. Move t to see the prediction update.",
    },
    "horizon": {"fr": "Horizon Δ (années)", "en": "Horizon Δ (years)"},
    "compute": {"fr": "Calculer", "en": "Compute"},
    "tab_pred": {"fr": "Prédiction", "en": "Prediction"},
    "tab_data": {"fr": "Données utilisées", "en": "Data used"},
    "tab_about": {"fr": "À propos du modèle", "en": "About the model"},
    "not_computed": {"fr": "Renseignez le patient pour afficher la prédiction.",
                     "en": "Fill in the patient to display the prediction."},
    "recomputing": {"fr": "Recalcul…", "en": "Recomputing…"},
    "stale": {"fr": "Les entrées ont changé depuis le dernier calcul.", "en": "Inputs changed since last computation."},
    "risk_at": {"fr": "risque à t + {d} an{s}", "en": "risk at t + {d} yr{s}"},
    "compute_time": {"fr": "temps de calcul", "en": "computation time"},
    "plot_title": {"fr": "Probabilité cumulée de {evt} après t = {t:.1f} an{s}",
                   "en": "Cumulative probability of {evt} after t = {t:.1f} yr{s}"},
    "observed": {"fr": "{marker} observée", "en": "Observed {marker}"},
    "ignored": {"fr": "Visites après t (non utilisées)", "en": "Visits after t (not used)"},
    "cif": {"fr": "Probabilité cumulée", "en": "Cumulative probability"},
    "ci": {"fr": "IC 95 %", "en": "95% CI"},
    "time_axis": {"fr": "Années depuis M12", "en": "Years since M12"},
    "download": {"fr": "Télécharger les prédictions (CSV)", "en": "Download predictions (CSV)"},
    "err_markers": {
        "fr": "Chaque marqueur doit avoir au moins une mesure avant t : {m}.",
        "en": "Each marker needs at least one measurement before t: {m}.",
    },
    "err_model": {"fr": "Erreur pendant le calcul : {e}", "en": "Error during computation: {e}"},
    "err_nan": {
        "fr": "Prédiction hors domaine (résultat non défini). Vérifiez que les valeurs saisies sont "
        "physiologiquement plausibles (une créatinine ou un âge extrême peut faire diverger le modèle).",
        "en": "Out-of-range prediction (undefined result). Check that the entered values are physiologically "
        "plausible (an extreme creatinine or age can make the model diverge).",
    },
    "spinner": {"fr": "Calcul de la prédiction…", "en": "Computing prediction…"},
    "footer": {"fr": "Research Use Only · Prototype Sêmeia Datalab · jmrespy", "en": "Research Use Only · Sêmeia Datalab prototype · jmrespy"},
    "about_md": {
        "fr": """
**Modèle.** Dynkigraft — modèle conjoint à effets aléatoires partagés (jmrespy), risque de base de Weibull,
risques concurrents cause-spécifiques : **échec du greffon** (retour en dialyse) et **décès** (greffon
fonctionnel). Trois marqueurs longitudinaux liés au risque par leur valeur courante : log-créatinine et
log-protéinurie (modèles linéaires mixtes, axe `Tsuivi_PostM12`) et MFI anti-HLA de classe II (axe
`Timmun_PostM12`). Covariables de baseline : âge du receveur et du donneur, créatinine à 3 mois et du
donneur, type de donneur, anti-HLA II à J0, greffe avant 2008, antécédents cardio-vasculaire et de diabète,
reprise de fonction retardée, rejet dans la 1re année, technique d'épuration.

**Prédiction.** Probabilité de survenue de l'évènement entre *t* et *t + Δ*, sachant le patient sans
évènement à *t* et compte tenu de toutes les mesures disponibles jusqu'à *t*. Sans simulation, les effets
aléatoires sont fixés à leur mode a posteriori ; avec simulation, l'IC 95 % est obtenu par Monte-Carlo sur
les paramètres et les effets aléatoires.

**Unités de saisie.** Saisie en **unités cliniques**, converties automatiquement vers l'espace du modèle
d'après le code des auteurs (`fit_JM_10.py` / `pred_JM_10_test.py`) : standardisation sans centrage
(`x / écart-type`, diviseurs 14,39 / 16,26 / 53,22 / 60,04 pour âge receveur/donneur et créatinine 3 mois/
donneur) ; `log(créatinine)`, `log(protéinurie + 1)` ; DSA anti-HLA II > 2000 en binaire ; temps en années.

**À confirmer.** Unités créatinine (µmol/L supposé) et protéinurie (g/L supposé) ; nature exacte de
`antiClassII_jour` (utilisée brute) ; libellés des classes 2/3 de la technique d'épuration.

**Limites.** Modèle de démonstration, non validé pour un usage clinique. Research Use Only.
""",
        "en": """
**Model.** Dynkigraft — shared random-effects joint model (jmrespy), Weibull baseline hazard, cause-specific
competing risks: **graft failure** (return to dialysis) and **death** (functioning graft). Three longitudinal
markers linked to the hazard through their current value: log-creatinine and log-proteinuria (linear mixed
models, `Tsuivi_PostM12` axis) and anti-HLA class II MFI (`Timmun_PostM12` axis). Baseline covariates:
recipient and donor age, creatinine at 3 months and donor creatinine, donor type, anti-HLA II at day 0,
transplant before 2008, cardiovascular and diabetes history, delayed graft function, rejection within the
first year, dialysis technique.

**Prediction.** Probability of the event between *t* and *t + Δ*, given the patient is event-free at *t* and
all measurements available up to *t*. Without simulation, random effects are set to their posterior mode;
with simulation, the 95% CI is obtained by Monte Carlo over parameters and random effects.

**Input units.** Inputs are in **clinical units**, converted automatically to model space following the
authors' code (`fit_JM_10.py` / `pred_JM_10_test.py`): standardization without centering (`x / SD`, divisors
14.39 / 16.26 / 53.22 / 60.04 for recipient/donor age and 3-month/donor creatinine); `log(creatinine)`,
`log(proteinuria + 1)`; anti-HLA II DSA > 2000 as binary; time in years.

**To confirm.** Creatinine (assumed µmol/L) and proteinuria (assumed g/L) units; exact nature of
`antiClassII_jour` (used raw); labels of dialysis classes 2/3.

**Limitations.** Demonstration model, not validated for clinical use. Research Use Only.
""",
    },
}


def t(key: str, lang: str, **kw) -> str:
    s = T.get(key, {}).get(lang, key)
    return s.format(**kw) if kw else s
