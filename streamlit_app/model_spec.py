"""Description du modèle servi par l'application.

C'est le seul fichier à adapter pour brancher un autre modèle jmrespy :
- les covariables de baseline saisies dans le formulaire et leur encodage,
- les marqueurs longitudinaux (clé jmrespy, colonne saisie, transformation),
- les évènements prédictibles (clés de `surv_object`).

Modèle en production : Dynkigraft — modèle conjoint de rejet/échec de greffe rénale.
3 marqueurs longitudinaux (créatinine, protéinurie, MFI anti-HLA II) et 2 risques
concurrents cause-spécifiques : échec du greffon (retour en dialyse) et décès.

-------------------------------------------------------------------------------------------
UNITÉS CLINIQUES (constantes issues du code des auteurs)
-------------------------------------------------------------------------------------------
La saisie se fait en UNITÉS CLINIQUES. Les constantes proviennent des scripts d'entraînement
et de prédiction DynKiGraft (Chabeau, `fit_JM_10.py` / `pred_JM_10_test.py`) :

  - Standardisation `with_mean=False` (division par l'écart-type, SANS centrage) :
        creat_3m_s = creat_3m / 53.219925   ;  creatD_s = creatD / 60.035091
        ageR_s     = ageR     / 14.389882   ;  ageD_s   = ageD   / 16.257463
    => dans ContinuousCov : mean=0.0, std=<diviseur>, to_model(x) = x / std.
  - Marqueurs : log_creat = log(creat) ; log_protU_1 = log(protU + 1) ; mfi_2000_II = binaire
    (max MFI des DSA anti-HLA II > 2000 : 1/0).
  - Temps en années (post-visite M12).

Unités déduites (à confirmer) : créatinine en µmol/L (base française DynPG), protéinurie en g/L.
Nature exacte de `antiClassII_jour` à confirmer : utilisée BRUTE par le Cox (ni log ni standardisée).
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

MODEL_PATH = "models/Dynkigraft.sav"

# `id_var` passé à Survfit_CIF : simple clé de jointure baseline <-> longitudinal.
# (La colonne de groupe d'entraînement du modèle est `clef` ; on reprend ce nom.)
ID_VAR = "clef"

# Unité de l'axe temps « post-M12 ». HYPOTHÈSE : années (à confirmer).
TIME_UNIT = {"fr": "années", "en": "years"}

# Colonne temps unique du tableau de suivi de l'interface. Chaque marqueur possède
# sa propre variable temps dans le modèle (voir Marker.time_var) : encode_longitudinal
# renomme cette colonne vers la bonne variable par marqueur.
TIME_COL = "t_postM12"


def _identity(s: pd.Series) -> pd.Series:
    return s.astype(float)


# ============================================================================================
# Marqueurs longitudinaux
# ============================================================================================
@dataclass
class Marker:
    key: str                      # clé du marqueur dans le JointModel (long_keys)
    input_col: str                # colonne saisie dans le tableau de suivi
    model_col: str                # colonne attendue par la formule du modèle mixte (LHS)
    time_var: str                 # variable temps attendue par le modèle pour ce marqueur
    label: dict                   # libellé FR/EN
    short: Optional[dict] = None  # libellé court pour l'en-tête du tableau (défaut : label)
    unit: str = ""
    transform: Callable[[pd.Series], pd.Series] = _identity
    model_space: bool = True      # True : la saisie EST déjà model_col (pas de transformation)
    kind: str = "numeric"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: float = 0.1
    default: float = 0.0          # valeur par défaut d'une nouvelle ligne de suivi
    plot: bool = True             # tracé sur le graphique de suivi
    log_axis: bool = False        # axe log pour le tracé (False si la valeur est déjà un log)


MARKERS = [
    Marker(
        key="creat",
        input_col="creat",
        model_col="log_creat",
        time_var="Tsuivi_PostM12",
        transform=lambda s: np.log(s.astype(float)),   # log_creat = log(creat)
        model_space=False,
        label={"fr": "Créatinine", "en": "Creatinine"},
        short={"fr": "Créat.", "en": "Creat."},
        unit="µmol/L",
        min_value=10.0, max_value=1500.0, step=1.0, default=130.0,
        plot=True, log_axis=True,
    ),
    Marker(
        key="prot",
        input_col="protU",
        model_col="log_protU_1",
        time_var="Tsuivi_PostM12",
        transform=lambda s: np.log(s.astype(float) + 1.0),   # log_protU_1 = log(protU + 1)
        model_space=False,
        label={"fr": "Protéinurie", "en": "Proteinuria"},
        short={"fr": "Prot.", "en": "Prot."},
        unit="g/day",   # le manuscrit rapporte la protéinurie en g/jour (unité exacte à confirmer)
        min_value=0.0, max_value=30.0, step=0.1, default=0.2,
        plot=False,
    ),
    Marker(
        key="mfi_2000_II",
        input_col="mfi_2000_II",
        model_col="mfi_2000_II",
        time_var="Timmun_PostM12",
        model_space=True,        # binaire, utilisé tel quel par le modèle
        label={"fr": "DSA anti-HLA II > 2000", "en": "Anti-HLA II DSA > 2000"},
        short={"fr": "DSA II", "en": "DSA II"},
        unit="0/1",
        kind="binary",
        min_value=0.0, max_value=1.0, step=1.0, default=0.0,
        plot=False,
    ),
]


# ============================================================================================
# Évènements (risques concurrents cause-spécifiques — clés de surv_object)
# ============================================================================================
@dataclass
class Event:
    key: str
    label: dict


EVENTS = [
    # gf : event_col=Retour  -> échec du greffon (retour en dialyse)   [libellé inféré]
    Event("gf", {"fr": "Échec du greffon (retour en dialyse)", "en": "Graft failure (return to dialysis)"}),
    # pf : event_col=DecesFct -> décès avec greffon fonctionnel        [libellé inféré]
    Event("pf", {"fr": "Décès (greffon fonctionnel)", "en": "Death (functioning graft)"}),
]


# ============================================================================================
# Covariables de baseline (références des formules Cox de surv_object)
# ============================================================================================
@dataclass
class ContinuousCov:
    name: str                     # colonne modèle (référencée par la formule Cox)
    label: dict
    unit: str = ""
    mean: Optional[float] = None  # moyenne d'entraînement (unité brute) ; None -> saisie z-score
    std: Optional[float] = None   # écart-type d'entraînement ; None -> saisie z-score
    default: float = 0.0          # valeur par défaut DANS L'ESPACE DE SAISIE
    step: float = 0.1
    min_value: Optional[float] = None  # bornes DANS L'ESPACE DE SAISIE (z-score tant que non clinique)
    max_value: Optional[float] = None
    help: Optional[dict] = None

    @property
    def clinical(self) -> bool:
        return self.mean is not None and self.std is not None and self.std != 0

    def to_model(self, x: float) -> float:
        """Saisie -> valeur attendue par le modèle."""
        return (float(x) - self.mean) / self.std if self.clinical else float(x)


@dataclass
class BinaryCov:
    name: str
    label: dict
    default: int = 0
    help: Optional[dict] = None


@dataclass
class CategoricalCov:
    key: str                      # clé d'interface
    label: dict
    dummies: list                 # colonnes modèle des modalités non-référence (one-hot)
    levels: list                  # libellés FR/EN ; levels[0] = référence (toutes dummies à 0)
    default: int = 0              # index dans levels
    help: Optional[dict] = None


# Continues. Les `_s` sont standardisées : renseigner mean/std pour passer en clinique.
# `antiClassII_jour` n'a pas de suffixe `_s` -> saisie brute (pas de standardisation).
# Saisie en unités cliniques : mean=0, std=<diviseur> -> to_model(x) = x / std (with_mean=False).
# Valeurs par défaut = Patient B du manuscrit (Figure 6, page 38 : 35 ans, HD, donneur 41 ans, etc.).
CONTINUOUS = [
    ContinuousCov("ageR_s", {"fr": "Âge du receveur", "en": "Recipient age"},
                  unit="ans", mean=0.0, std=14.389882, default=35.0, step=1.0, min_value=18.0, max_value=90.0),
    ContinuousCov("ageD_s", {"fr": "Âge du donneur", "en": "Donor age"},
                  unit="ans", mean=0.0, std=16.257463, default=41.0, step=1.0, min_value=0.0, max_value=95.0),
    ContinuousCov("creat_3m_s", {"fr": "Créatinine à 3 mois", "en": "Creatinine at 3 months"},
                  unit="µmol/L", mean=0.0, std=53.219925, default=110.0, step=1.0, min_value=20.0, max_value=1200.0,
                  help={"fr": "Créatininémie sérique à 3 mois (unité supposée µmol/L, à confirmer).",
                        "en": "Serum creatinine at 3 months (unit assumed µmol/L, to confirm)."}),
    ContinuousCov("creatD_s", {"fr": "Créatinine du donneur", "en": "Donor creatinine"},
                  unit="µmol/L", mean=0.0, std=60.035091, default=125.0, step=1.0, min_value=20.0, max_value=1200.0),
    ContinuousCov("antiClassII_jour", {"fr": "Anti-HLA classe II (J0)", "en": "Anti-HLA class II (day 0)"},
                  unit="", default=0.0, step=0.1, min_value=0.0,
                  help={"fr": "Utilisée brute par le modèle (ni log ni standardisée). Nature/unité exacte à confirmer.",
                        "en": "Used raw by the model (neither logged nor standardized). Exact nature/unit to confirm."}),
]

# Binaires 0/1 (identiques en espace modèle et clinique).
BINARIES = [
    BinaryCov("ante_2008", {"fr": "Greffe avant 2008", "en": "Transplant before 2008"}),
    BinaryCov("anteCardioVasc", {"fr": "Antécédent cardio-vasculaire", "en": "Cardiovascular history"}),
    BinaryCov("anteDiab", {"fr": "Antécédent de diabète", "en": "Diabetes history"}),
    BinaryCov("delayedGraftFunc", {"fr": "Reprise de fonction retardée", "en": "Delayed graft function"}),
    BinaryCov("Rejet_1an", {"fr": "Rejet dans la 1re année", "en": "Rejection within first year"}),
]

# Catégorielles one-hot. Libellés de modalités INFÉRÉS (codebook à confirmer).
CATEGORICALS = [
    CategoricalCov(
        "typeD_ext", {"fr": "Type de donneur", "en": "Donor type"},
        dummies=["typeD_ext_1", "typeD_ext_2"],
        # D'après le code : 0 = donneur vivant ; 1 = donneur décédé, cause non cérébrovasculaire ;
        # 2 = donneur décédé, cause cérébrovasculaire. Patient B -> index 1.
        levels=[{"fr": "Donneur vivant", "en": "Living donor"},
                {"fr": "Décédé — cause non cérébrovasculaire", "en": "Deceased — non-cerebrovascular"},
                {"fr": "Décédé — cause cérébrovasculaire", "en": "Deceased — cerebrovascular"}],
        default=1,
    ),
    CategoricalCov(
        "techEpu3cl", {"fr": "Technique d'épuration (pré-greffe)", "en": "Pre-transplant dialysis"},
        dummies=["techEpu3cl_2", "techEpu3cl_3"],
        # classe 1 = greffe préemptive (réf., confirmé). Classes 2/3 = hémodialyse / dialyse péritonéale
        # (mapping HD=2, DP=3 INFÉRÉ, à confirmer). Patient B = hémodialyse -> index 1.
        levels=[{"fr": "Greffe préemptive (réf.)", "en": "Pre-emptive transplant (ref.)"},
                {"fr": "Hémodialyse", "en": "Haemodialysis"},
                {"fr": "Dialyse péritonéale", "en": "Peritoneal dialysis"}],
        default=1,
    ),
]


def encode_baseline(patient_id, cont: dict, binary: dict, categ: dict) -> pd.DataFrame:
    """Construit la ligne de baseline avec toutes les colonnes attendues par les formules Cox.

    cont   : {nom_covariable: valeur saisie}
    binary : {nom_covariable: 0/1}
    categ  : {clé_catégorielle: index de modalité}
    """
    row = {ID_VAR: [patient_id]}
    for c in CONTINUOUS:
        row[c.name] = [c.to_model(cont.get(c.name, c.default))]
    for b in BINARIES:
        row[b.name] = [int(binary.get(b.name, b.default))]
    for cat in CATEGORICALS:
        idx = int(categ.get(cat.key, cat.default))
        for j, dummy in enumerate(cat.dummies):
            row[dummy] = [1 if idx == j + 1 else 0]
    return pd.DataFrame(row)


def encode_longitudinal(patient_id, follow_up: pd.DataFrame) -> dict:
    """Construit le dictionnaire {clé marqueur: DataFrame} attendu par Survfit_CIF.

    Le tableau de suivi de l'interface a une colonne temps unique (TIME_COL) ; on la
    renomme vers la variable temps propre à chaque marqueur (time_var).
    """
    out = {}
    for m in MARKERS:
        df = follow_up[[TIME_COL, m.input_col]].dropna().copy()
        df[m.model_col] = df[m.input_col] if m.model_space else m.transform(df[m.input_col])
        df[ID_VAR] = patient_id
        df = df.rename(columns={TIME_COL: m.time_var})
        out[m.key] = df[[ID_VAR, m.time_var, m.model_col]].sort_values(m.time_var).reset_index(drop=True)
    return out


# ============================================================================================
# Patient par défaut — Patient B du manuscrit (Figure 6, page 38)
# ============================================================================================
# Suivi longitudinal lu sur la Figure 6 : créatinine croissante (137 -> 286 µmol/L), protéinurie
# faible puis ~0,39 g/j à 6 ans, DSA anti-HLA II apparaissant vers 4,9 ans. Le temps est exprimé
# en années depuis M12 (= année post-greffe − 1) : les visites « 1 à 6 ans post-greffe » du
# manuscrit correspondent à t_postM12 = 0 à 5. Valeurs approximées depuis la figure.
DEFAULT_FOLLOW = pd.DataFrame({
    TIME_COL:       [0.0,  1.0,  2.0,  3.0,  4.0,  5.0],
    "creat":        [137., 170., 195., 225., 255., 286.],   # µmol/L
    "protU":        [0.1,  0.1,  0.1,  0.15, 0.25, 0.39],    # g/jour
    "mfi_2000_II":  [0,    0,    0,    0,    1,    1],        # DSA anti-HLA II > 2000 (binaire)
})

# Patient A du manuscrit (Figure 5, page 37) : patient stable, pas d'échec de greffon.
# 41 ans, dialyse péritonéale, donneur 41... non : donneur 22 ans (DCD non cérébrovasc.),
# créatinine donneur 109, créatinine 3 mois 94. SCr stable ~105-112, protéinurie faible, pas de DSA II.
FOLLOW_A = pd.DataFrame({
    TIME_COL:       [0.0,  1.0,  2.0,  3.0,  4.0,  5.0],
    "creat":        [108., 110., 106., 112., 108., 110.],   # µmol/L (stable, lu sur la Figure 5)
    "protU":        [0.05, 0.05, 0.08, 0.05, 0.1,  0.08],   # g/jour (faible)
    "mfi_2000_II":  [0,    0,    0,    0,    0,    0],        # pas de DSA anti-HLA II
})


@dataclass
class Patient:
    key: str
    label: dict
    cont: dict        # {nom covariable continue: valeur clinique}
    binary: dict      # {nom: 0/1}
    categ: dict       # {clé catégorielle: index de modalité}
    follow: pd.DataFrame


# Deux patients illustratifs du manuscrit (pseudonymisés). Patient A (p.37) en 1er,
# Patient B (p.38) en 2e position mais sélectionné par défaut (cf. DEFAULT_PATIENT_KEY).
PATIENTS = [
    Patient(
        "A", {"fr": "Patient A — greffon stable (Fig. 5, p. 37)", "en": "Patient A — stable graft (Fig. 5, p. 37)"},
        cont={"ageR_s": 41.0, "ageD_s": 22.0, "creat_3m_s": 94.0, "creatD_s": 109.0, "antiClassII_jour": 0.0},
        binary={"ante_2008": 0, "anteCardioVasc": 0, "anteDiab": 0, "delayedGraftFunc": 0, "Rejet_1an": 0},
        categ={"typeD_ext": 1, "techEpu3cl": 2},   # DCD non cérébrovasc. ; dialyse péritonéale
        follow=FOLLOW_A,
    ),
    Patient(
        "B", {"fr": "Patient B — échec du greffon (Fig. 6, p. 38)", "en": "Patient B — graft failure (Fig. 6, p. 38)"},
        cont={"ageR_s": 35.0, "ageD_s": 41.0, "creat_3m_s": 110.0, "creatD_s": 125.0, "antiClassII_jour": 0.0},
        binary={"ante_2008": 0, "anteCardioVasc": 0, "anteDiab": 0, "delayedGraftFunc": 0, "Rejet_1an": 0},
        categ={"typeD_ext": 1, "techEpu3cl": 1},   # DCD non cérébrovasc. ; hémodialyse
        follow=DEFAULT_FOLLOW,
    ),
]
PATIENTS_BY_KEY = {p.key: p for p in PATIENTS}
DEFAULT_PATIENT_KEY = "B"
