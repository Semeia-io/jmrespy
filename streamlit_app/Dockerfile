# Image construite depuis la RACINE du dépôt jmrespy (le contexte doit inclure le
# paquet `jmrespy/`, voisin de `streamlit_app/`). Via docker compose, le contexte
# est fixé à `../` et ce fichier est désigné par `streamlit_app/Dockerfile` ; en
# direct :
#   docker build -f streamlit_app/Dockerfile -t jmrespy-ruo .
#
# Pas de pin `--platform` ici (pour autoriser les builds arm en local). La plateforme
# de production est figée côté docker-compose.yml (service `jmrespy`, linux/amd64).
FROM python:3.10-slim

WORKDIR /app

# Sans PYTHONUNBUFFERED, stdout est bufferisé par blocs hors TTY : les traces de
# calcul resteraient dans le tampon au lieu d'apparaître dans les logs du conteneur.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/jmrespy

# build-essential : certaines versions figées (pymc 5.0.2, statsmodels 0.13.2…)
# peuvent n'avoir aucune wheel pour python:3.10-slim et se compiler depuis les
# sources. Conservé pour garantir un `pip install` reproductible.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dépendances d'abord, isolées du reste du code : tant que requirements.txt ne
# change pas, Docker réutilise cette couche d'installation au lieu de tout
# réinstaller à chaque modification d'une page ou d'un module.
COPY streamlit_app/requirements.txt .
RUN pip install -r requirements.txt

# jmrespy n'a pas de __init__.py (pip install produirait un paquet vide) : on le
# place sur le PYTHONPATH. COPY d'un RÉPERTOIRE pour conserver la structure du
# paquet (jmrespy/ et jmrespy/utils/). Copié avant le code applicatif car il change
# rarement : la couche reste en cache tant que la librairie n'évolue pas.
COPY jmrespy /opt/jmrespy/jmrespy

# Code de l'app en dernier (change le plus souvent) pour préserver le cache des
# couches précédentes. MODEL_PATH est relatif ("models/Dynkigraft.sav") : c'est
# WORKDIR=/app qui doit contenir models/, ce que garantit ce COPY.
COPY streamlit_app/ .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request as r; r.urlopen('http://localhost:8501/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
