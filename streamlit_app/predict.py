"""Couche de prédiction, indépendante de l'interface.

Réutilisable telle quelle derrière une API (FastAPI) si le prototype devient un produit.
"""

import time
import warnings

import joblib
import numpy as np
import pandas as pd

import model_spec as spec


def load_model(path: str = spec.MODEL_PATH):
    # jmrespy doit être importable (pip install du dépôt) pour désérialiser le modèle
    import jmrespy.jointModel as jm

    # Contournement d'un bug de jmrespy 1.5.0 : Survfit_CIF borne la probabilité avec le `min` natif
    # de Python, qui lève « truth value of an array is ambiguous » dès qu'il y a plus d'un horizon.
    # Correctif à faire en amont : remplacer min(...) par np.minimum(...) (jointModel.py, l. 1340 et 1398).
    def _min(a, b):
        return np.minimum(a, b) if (np.ndim(a) or np.ndim(b)) else min(a, b)

    jm.min = _min
    return joblib.load(path)


def predict_cif(
    model,
    baseline: pd.DataFrame,
    longitudinal: dict,
    event: str,
    horizon: float,
    n_points: int = 30,
    landmark: float | None = None,
    simulate: bool = False,
    n_sim: int = 100,
) -> tuple[pd.DataFrame, float]:
    """Probabilité cumulée de l'évènement entre le landmark t et t + Δ.

    Retourne (DataFrame[t, u, delta, pi | mean, median, lower, upper], durée en s).
    """
    if landmark is None:
        landmark = max(longitudinal[m.key][m.time_var].max() for m in spec.MARKERS)
    surv_times = landmark + np.linspace(0, horizon, n_points + 1)[1:]

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.Survfit_CIF(
            new_data_baseline=baseline.copy(),
            new_data_long={k: v.copy() for k, v in longitudinal.items()},
            id_var=spec.ID_VAR,
            surv_times=surv_times,
            last_time=np.array([float(landmark)]),
            simulate=simulate,
            M=n_sim,
            interest_evt=event,
        )
    elapsed = time.perf_counter() - t0

    res = res.rename(columns={"pi_evt": "pi"})
    res["delta"] = res["u"] - res["t"]
    if simulate:
        res["pi"] = res["mean"]
    return res, elapsed


def risk_at(res: pd.DataFrame, delta: float) -> float:
    """Interpolation du risque à t + delta."""
    x = np.concatenate([[0.0], res["delta"].to_numpy(dtype=float)])
    y = np.concatenate([[0.0], res["pi"].to_numpy(dtype=float)])
    return float(np.interp(delta, x, y))
