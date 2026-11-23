"""Prototype RUO — prédiction dynamique avec un modèle conjoint jmrespy.

Modèle : Dynkigraft (rejet / échec de greffe rénale).
Lancement : streamlit run app.py   (avec PYTHONPATH pointant sur la racine du dépôt jmrespy)
"""

import hashlib
import json
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import model_spec as spec
import predict
import wiser
from i18n import t

st.set_page_config(page_title="jmrespy · RUO", page_icon="🟣", layout="wide")
wiser.inject_css()


# --- Ressources ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    return predict.load_model()

# --- En-tête ------------------------------------------------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "fr"

c_brand, c_lang = st.columns([5, 1], vertical_alignment="center")
with c_lang:
    lang_label = st.segmented_control(
        "Langue", ["FR", "EN"], default="FR", key="lang_ctl", label_visibility="collapsed"
    )
lang = (lang_label or "FR").lower()
with c_brand:
    wiser.header(t("app_title", lang), t("app_sub", lang))
wiser.ruo_banner(t("ruo", lang))
wiser.abstract(t("abstract_title", lang), t("abstract_md", lang))

left, right = st.columns([1, 1], gap="large")

# --- Saisie ----------------------------------------------------------------------
with left:
    pkeys = [p.key for p in spec.PATIENTS]
    sel_key = st.segmented_control(
        t("patient_ref", lang), pkeys, default=spec.DEFAULT_PATIENT_KEY,
        format_func=lambda k: spec.PATIENTS_BY_KEY[k].label[lang], key="patient_sel",
    ) or spec.DEFAULT_PATIENT_KEY
    patient = spec.PATIENTS_BY_KEY[sel_key]
    sfx = sel_key  # suffixe des clés de widgets : change le formulaire quand on change de patient

    tab_fu, tab_pat = st.tabs([t("sec_followup", lang), t("sec_patient", lang)])

    # 1er onglet : tableau de données longitudinales
    with tab_fu:
        st.caption(t("followup_help", lang))
        col_cfg = {spec.TIME_COL: st.column_config.NumberColumn(
            t("time", lang), min_value=0.0, step=0.1, format="%.2f")}
        for m in spec.MARKERS:
            short = (m.short or m.label)[lang]
            lbl = f"{short} ({m.unit})" if m.unit else short
            col_cfg[m.input_col] = st.column_config.NumberColumn(
                lbl, min_value=m.min_value, max_value=m.max_value,
                step=m.step, format="%d" if m.step >= 1 else "%.2f",
            )
        follow = st.data_editor(
            patient.follow.copy(), num_rows="dynamic", hide_index=True,
            column_config=col_cfg, width="stretch", key=f"follow_{sfx}",
        )
        follow = follow.dropna(subset=[spec.TIME_COL]).sort_values(spec.TIME_COL).reset_index(drop=True)

    # 2e onglet : covariables de baseline du patient
    with tab_pat:
        st.caption(t("patient_help", lang))
        cont_values, bin_values, cat_values = {}, {}, {}

        for c in spec.CONTINUOUS:
            lbl = f"{c.label[lang]} ({c.unit})" if c.unit else c.label[lang]
            cont_values[c.name] = st.number_input(
                lbl, value=float(patient.cont.get(c.name, c.default)), step=float(c.step), format="%.2f",
                min_value=c.min_value, max_value=c.max_value,
                help=c.help[lang] if c.help else None, key=f"cont_{c.name}_{sfx}",
            )

        for cat in spec.CATEGORICALS:
            cat_values[cat.key] = st.selectbox(
                cat.label[lang], list(range(len(cat.levels))), index=int(patient.categ.get(cat.key, cat.default)),
                format_func=lambda i, _cat=cat: _cat.levels[i][lang],
                help=cat.help[lang] if cat.help else None, key=f"cat_{cat.key}_{sfx}",
            )

        st.markdown(f"<div class='wr-muted'>{t('binaries', lang)}</div>", unsafe_allow_html=True)
        for b in spec.BINARIES:
            bin_values[b.name] = 1 if st.toggle(
                b.label[lang], value=bool(patient.binary.get(b.name, b.default)),
                help=b.help[lang] if b.help else None, key=f"bin_{b.name}_{sfx}",
            ) else 0

# Évènement fixé à l'échec du greffon : le risque concurrent de décès n'est pas affiché
event = spec.EVENTS[0].key

# Temps de prédiction t : la valeur est lue ici ; le curseur est rendu sous le graphique.
t_max = float(follow[spec.TIME_COL].max()) if len(follow) else 0.0
t_min = float(follow[spec.TIME_COL].min()) if len(follow) else 0.0
lm_lo = round(t_min, 2)
lm_hi = max(round(t_max, 2), lm_lo + 0.01)
if "lm" not in st.session_state:
    st.session_state["lm"] = round(t_max, 2)
else:  # garder la valeur du curseur dans la plage courante du suivi
    st.session_state["lm"] = float(min(max(float(st.session_state["lm"]), lm_lo), lm_hi))
landmark = float(st.session_state["lm"])


def landmark_slider():
    """Curseur du temps de prédiction t, rendu sous le graphique (clé 'lm')."""
    st.slider(t("landmark", lang), min_value=lm_lo, max_value=lm_hi,
              step=0.1, help=t("landmark_help", lang), key="lm")


horizon = 5.0      # horizon de prédiction fixé à 5 ans (curseur retiré)
simulate = False   # intervalle de confiance Monte-Carlo retiré
n_sim = 50

DEBOUNCE_S = 2.0   # secondes d'inactivité du tableau avant recalcul automatique


def _sig(obj) -> str:
    return hashlib.md5(json.dumps(obj, default=str).encode()).hexdigest()


@st.fragment(run_every=0.3)
def _debounce_poll() -> None:
    """Pendant l'attente, déclenche un recalcul complet dès 2 s sans édition du tableau.

    Le minuteur est piloté côté serveur (indépendant du clavier/souris) → fonctionne au tactile.
    """
    if st.session_state.get("debounce_active") and \
            time.monotonic() - st.session_state.get("table_edit_time", 0.0) >= DEBOUNCE_S:
        st.rerun()


# --- Préparation des données ---------------------------------------------------------
used = follow[follow[spec.TIME_COL] <= landmark + 1e-9]
baseline_df = spec.encode_baseline(1, cont_values, bin_values, cat_values)
long_dict = spec.encode_longitudinal(1, used)
missing = [m.label[lang] for m in spec.MARKERS if long_dict[m.key].empty]

# Signatures séparées : le formulaire (tout sauf le tableau) et le tableau de suivi.
form_sig = _sig([baseline_df.to_dict(), event, round(landmark, 4), horizon, sel_key])
table_sig = _sig(follow.to_dict())
inputs_sig = _sig([form_sig, table_sig])

# Toute modification du tableau (re)lance un minuteur : 2 s d'inactivité avant recalcul.
if st.session_state.get("seen_table_sig") != table_sig:
    st.session_state["seen_table_sig"] = table_sig
    st.session_state["table_edit_time"] = time.monotonic()

_last = st.session_state.get("result")
_up_to_date = _last is not None and _last.get("sig") == inputs_sig
_form_changed = _last is None or _last.get("form_sig") != form_sig
_idle = time.monotonic() - st.session_state.get("table_edit_time", time.monotonic())

act = waiting = False
if not _up_to_date:
    if _form_changed or _idle >= DEBOUNCE_S:
        act = True       # hors tableau, 1re exécution, ou tableau au repos 2 s → recalcul immédiat
    else:
        waiting = True   # tableau modifié récemment → on laisse l'utilisateur finir sa saisie
st.session_state["debounce_active"] = waiting

if act:
    if missing:
        st.session_state.pop("result", None)
        st.session_state.error = t("err_markers", lang, m=", ".join(missing))
    else:
        try:
            with right, st.spinner(t("spinner", lang)):
                res, elapsed = predict.predict_cif(
                    get_model(), baseline_df, long_dict, event=event, horizon=horizon,
                    landmark=landmark, simulate=simulate, n_sim=n_sim,
                )
                # Référence : le patient du papier dans son état par défaut, aux mêmes réglages
                ref_res = None
                ref_used = patient.follow[patient.follow[spec.TIME_COL] <= landmark + 1e-9]
                ref_long = spec.encode_longitudinal(1, ref_used)
                if all(not ref_long[m.key].empty for m in spec.MARKERS):
                    ref_base = spec.encode_baseline(1, patient.cont, patient.binary, patient.categ)
                    ref_tmp, _ = predict.predict_cif(
                        get_model(), ref_base, ref_long, event=event, horizon=horizon,
                        landmark=landmark, simulate=False,
                    )
                    if np.isfinite(ref_tmp["pi"].to_numpy()).all():
                        ref_res = ref_tmp
            if not np.isfinite(res["pi"].to_numpy()).all():
                st.session_state.pop("result", None)
                st.session_state.error = t("err_nan", lang)
            else:
                st.session_state.result = dict(res=res, elapsed=elapsed, sig=inputs_sig, form_sig=form_sig,
                                               event=event, landmark=landmark, simulate=simulate, horizon=horizon,
                                               ref_res=ref_res, patient_key=sel_key)
                st.session_state.error = None
        except Exception as e:  # prototype : on affiche l'erreur telle quelle
            st.session_state.pop("result", None)
            st.session_state.error = t("err_model", lang, e=e)

# Pendant l'attente (tableau en cours d'édition), un minuteur invisible relance le calcul.
if waiting:
    _debounce_poll()

# --- Résultats -----------------------------------------------------------------------
with right:
    if waiting:
        st.caption("⏳ " + t("recomputing", lang))
    err = st.session_state.get("error")
    if err:
        st.error(err)
    r = st.session_state.get("result")
    if not r:
        if not err:
            st.html(f'<div class="wr-empty">{t("not_computed", lang)}</div>')
        landmark_slider()
    else:
        res = r["res"]
        evt_label = next(e.label[lang] for e in spec.EVENTS if e.key == r["event"])
        items = []
        for d in [x for x in (1, 3, 5) if x <= r["horizon"]] or [r["horizon"]]:
            p = predict.risk_at(res, d)
            items.append((f"{p:.0%}", t("risk_at", lang, d=f"{d:g}", s="s" if d > 1 else ""), "pink"))
        items.append((f"{r['elapsed']:.1f} s", t("compute_time", lang), "grey"))
        wiser.pills(items)

        # Graphique multi-panneaux (comme la Figure 6) : un panneau par marqueur + le risque
        lm = r["landmark"]
        markers = spec.MARKERS
        n = len(markers)
        # Référence : patient du papier dans son état par défaut (toujours affichée, couleur sombre)
        ref_res = r.get("ref_res")
        ref_patient = spec.PATIENTS_BY_KEY.get(r.get("patient_key"))
        ref_follow = ref_patient.follow if ref_patient is not None else None
        fig = make_subplots(rows=n + 1, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05, row_heights=[0.2] * n + [0.4])

        for i, m in enumerate(markers, start=1):
            obs = follow.dropna(subset=[m.input_col])
            before = obs[obs[spec.TIME_COL] <= lm + 1e-9]
            after = obs[obs[spec.TIME_COL] > lm + 1e-9]
            fig.add_trace(go.Scatter(
                x=before[spec.TIME_COL], y=before[m.input_col], mode="markers+lines", showlegend=False,
                marker=dict(color=wiser.PURPLE, size=7), line=dict(color=wiser.PURPLE_MEDIUM, width=1),
                hovertemplate="%{x:.2f} : %{y}<extra></extra>",
            ), row=i, col=1)
            if len(after):
                fig.add_trace(go.Scatter(
                    x=after[spec.TIME_COL], y=after[m.input_col], mode="markers", showlegend=False,
                    marker=dict(color=wiser.GREY_MEDIUM, size=6, symbol="circle-open"),
                ), row=i, col=1)
            if ref_follow is not None:  # trajectoire de référence (papier), sombre et discrète
                robs = ref_follow.dropna(subset=[m.input_col])
                robs = robs[robs[spec.TIME_COL] <= lm + 1e-9]
                fig.add_trace(go.Scatter(
                    x=robs[spec.TIME_COL], y=robs[m.input_col], mode="lines+markers", showlegend=False,
                    line=dict(color=wiser.BLACK, width=1, dash="dot"),
                    marker=dict(color=wiser.BLACK, size=5, symbol="circle-open"), opacity=0.45,
                    hoverinfo="skip",
                ), row=i, col=1)
            fig.update_yaxes(title_text=f"{m.label[lang]}<br>({m.unit})" if m.unit else m.label[lang],
                             title_font=dict(size=11), color=wiser.PURPLE, row=i, col=1)
            if m.kind == "binary":
                fig.update_yaxes(tickvals=[0, 1], range=[-0.3, 1.3], row=i, col=1)
            elif m.log_axis:
                fig.update_yaxes(type="log", row=i, col=1)

        # Dernier panneau : probabilité cumulée de l'évènement
        rrow = n + 1
        x = np.concatenate([[lm], res["u"].to_numpy()])
        if r["simulate"]:
            hi = np.concatenate([[0], res["upper"].to_numpy()])
            lo = np.concatenate([[0], res["lower"].to_numpy()])
            fig.add_trace(go.Scatter(x=x, y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip"), row=rrow, col=1)
            fig.add_trace(go.Scatter(x=x, y=lo, line=dict(width=0), fill="tonexty",
                                     fillcolor="rgba(252,19,105,0.15)", showlegend=False, hoverinfo="skip"), row=rrow, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=np.concatenate([[0], res["pi"].to_numpy()]), mode="lines", showlegend=True,
            name=t("current_curve", lang),
            line=dict(color=wiser.PINK, width=3), hovertemplate="%{x:.2f} : %{y:.1%}<extra></extra>",
        ), row=rrow, col=1)
        if ref_res is not None:  # courbe de risque de référence (état par défaut du papier)
            rx = np.concatenate([[lm], ref_res["u"].to_numpy()])
            fig.add_trace(go.Scatter(
                x=rx, y=np.concatenate([[0], ref_res["pi"].to_numpy()]), mode="lines", showlegend=True,
                name=t("ref_curve", lang), line=dict(color=wiser.BLACK, width=2, dash="dot"),
                hovertemplate="%{x:.2f} : %{y:.1%}<extra></extra>",
            ), row=rrow, col=1)
        _refmax = float(ref_res["pi"].max()) if ref_res is not None else 0.0
        ymax = float(min(1.0, max(0.2, _refmax, res[["upper" if r["simulate"] else "pi"]].max().iloc[0]) * 1.15))
        fig.update_yaxes(range=[0, ymax], tickformat=".0%", title_text=t("cif", lang),
                         title_font=dict(size=11), color=wiser.PINK, row=rrow, col=1)

        # Ligne de landmark (t) et plage de temps communes à tous les panneaux
        for i in range(1, n + 2):
            fig.add_vline(x=lm, line=dict(color=wiser.BLACK, width=1, dash="dot"), row=i, col=1)
            fig.update_xaxes(range=[-0.2, lm + r["horizon"] + 0.2], row=i, col=1)
        fig.update_xaxes(title_text=t("time_axis", lang), title_standoff=8, row=rrow, col=1)
        fig.update_layout(
            template="wiser", height=610, showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.16, x=0, bgcolor="rgba(0,0,0,0)"),
            title=dict(text=t("plot_title", lang, evt=evt_label.lower(), t=lm, s="s" if lm > 1 else ""),
                       font=dict(size=15, color=wiser.BLACK), x=0, y=0.99),
            margin=dict(t=60, l=10, r=10, b=95),
        )
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        landmark_slider()  # temps de prédiction t, sous le graphique

        show = res[["t", "u", "delta", "pi"] + (["lower", "upper"] if r["simulate"] else [])].copy()
        st.download_button(t("download", lang), show.to_csv(index=False).encode(), "prediction_jmrespy.csv", "text/csv")

wiser.footer(t("footer", lang))
