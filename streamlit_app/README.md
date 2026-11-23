# jmrespy · Prototype RUO Streamlit

Interface de démonstration (Research Use Only) pour la prédiction dynamique avec un modèle conjoint
longitudinal–survie entraîné avec [jmrespy](https://github.com/Semeia-io/jmrespy), au design system
**Wiser** de Sêmeia.

Modèle en production : **Dynkigraft** (`models/Dynkigraft.sav`) — rejet / échec de greffe rénale.
Trois marqueurs longitudinaux (log-créatinine, log-protéinurie, MFI anti-HLA de classe II) et deux risques
concurrents cause-spécifiques : échec du greffon (retour en dialyse) et décès avec greffon fonctionnel.

Temps de calcul observés : environ 0,3 s par prédiction sans simulation, environ 0,3 s par simulation
Monte-Carlo pour l'IC 95 % (50 simulations ≈ 15 s).

> **Espace de saisie.** Le `.sav` ne contient ni les constantes de standardisation (covariables en `_s`)
> ni les transformations cliniques des marqueurs. Tant qu'elles ne sont pas renseignées dans `model_spec.py`
> (`ContinuousCov.mean/.std`, `Marker.transform`), la saisie se fait dans l'espace du modèle (z-scores,
> colonnes `log_creat` / `log_protU_1` / `mfi_2000_II`). Le passage en unités cliniques ne touche que
> `model_spec.py`.

## Structure

| Fichier | Rôle |
|---|---|
| `app.py` | Interface Streamlit (saisie, calcul, graphique, export) |
| `model_spec.py` | **Seul fichier à adapter pour un autre modèle** : covariables, marqueurs, évènements, encodage |
| `predict.py` | Chargement du modèle et appel à `JointModel.Survfit_CIF`, sans dépendance à Streamlit (réutilisable dans une API) |
| `wiser.py` | Tokens Wiser, CSS des composants, template Plotly |
| `i18n.py` | Libellés FR/EN |
| `.streamlit/config.toml` | Thème Streamlit (couleurs, police Inter, rayons) |
| `models/Dynkigraft.sav` | Modèle entraîné en production (joblib) |
| `models/fitted_JM_pbc.sav` | Ancien modèle de démonstration PBC (conservé pour référence) |

## Lancer en local

Python 3.10 obligatoire (versions figées par jmrespy).

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=..   # racine du dépôt jmrespy ; jmrespy n'est pas installable par pip (pas de __init__.py)
streamlit run app.py
```

## Docker

Le contexte de build est **la racine du dépôt jmrespy** (`jmrespy-main/`), afin
d'embarquer le paquet `jmrespy/` voisin de `streamlit_app/`. L'app est autonome :
ni base, ni auth, ni secret, ni volume. Deux fichiers compose, calqués sur le
pattern Belvédère :

- `docker-compose.local.yml` — développement local (build + port 8501) ;
- `docker-compose.yml` — production allégée (image `ghcr.io/semeia-io/…`,
  `platform: linux/amd64`, `restart: unless-stopped`), sans reverse-proxy.

**Local**, depuis ce dossier :

```bash
docker compose -f docker-compose.local.yml up --build
# → http://localhost:8501
```

**Production** (build/push puis déploiement de l'image publiée) :

```bash
JMRESPY_VERSION=1.0.0 docker compose build
JMRESPY_VERSION=1.0.0 docker compose push
JMRESPY_VERSION=1.0.0 docker compose up -d
```

Build direct sans compose, depuis la racine du dépôt :

```bash
docker build -f streamlit_app/Dockerfile -t jmrespy-ruo .
docker run -p 8501:8501 jmrespy-ruo
```

## Brancher un autre modèle

1. Entraîner le modèle, puis `del model.lme_object` et `joblib.dump(model, "models/mon_modele.sav")`.
2. Dans `model_spec.py` : `MODEL_PATH`, `MARKERS` (clé jmrespy, colonne saisie, colonne de la formule,
   transformation), `EVENTS` (clés de `surv_object`), `encode_baseline` (encodage identique à l'entraînement).
3. Adapter le formulaire « Patient » de `app.py` aux nouvelles covariables.

## Points connus sur jmrespy 1.5.0

- `Survfit_CIF` utilise `min(...)` natif sur des tableaux et plante dès qu'on demande plus d'un horizon.
  Contourné dans `predict.py` ; correctif à faire en amont : `np.minimum(...)` (l. 1340 et 1398 de `jointModel.py`).
- `JointModel.Survfit` (ancienne API) référence une variable `new_data` inexistante : utiliser `Survfit_CIF`.
- `setup.py` ne fige pas PyMC : avec numpy 1.24.2, il faut `pymc==5.0.2` (les versions récentes exigent numpy ≥ 1.25).
- `lifelines==0.27.4` sérialise une `formulaic.Formula` ; depuis formulaic 1.0 cette classe est abstraite et
  le modèle (CoxPHFitter picklé) ne se déserialise plus (`Can't instantiate abstract class Formula`). Figer
  `formulaic==0.6.6` (fait dans `requirements.txt`).
- Pas de `__init__.py` dans `jmrespy/` ni `jmrespy/utils/` : `pip install` produit un paquet vide, d'où le `PYTHONPATH`. Ajouter ces deux fichiers rendrait la librairie installable.

## Avertissement

Research Use Only. Outil non validé cliniquement, non destiné au diagnostic ni à la décision thérapeutique.
