"""Traduction du design system Wiser (Sêmeia) pour Streamlit.

Tokens issus de tailwind-wiser-base, styles de composants relevés dans le Storybook ember-wiser
(https://semeia-io.github.io/Platform/wiser/).
"""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --- Tokens -----------------------------------------------------------------
BLACK = "#1C1C1C"
WHITE = "#FFFFFF"
RED = "#C2003A"
ORANGE = "#FFA500"
GREEN = "#45B854"
TEAL = "#66BAAF"
PINK = "#FC1369"
PINK_MEDIUM = "#DF789B"
PINK_LIGHT = "#FEE3E9"
PURPLE = "#510C7B"
PURPLE_MEDIUM = "#AE80C3"
PURPLE_LIGHT = "#FAF3FF"
GREY = "#888888"
GREY_LIGHT = "#FAFAFA"
GREY_MEDIUM = "#CFCFCF"

SEMEIA_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 16 16" fill="none">'
    '<path d="M14.7515 7.03818L12.1866 2.46419C11.8499 1.86573 11.2317 1.5 10.5674 1.5H5.43775C4.76885 1.5 '
    "4.1507 1.86573 3.81396 2.46419L1.2491 7.03818C0.916965 7.63189 0.916965 8.36811 1.2491 8.96182L3.81396 "
    "13.5358C4.14609 14.1295 4.76424 14.5 5.43312 14.5H10.5628C11.2317 14.5 11.8499 14.1343 12.182 13.5358L14.7469 "
    "8.96182C15.0836 8.36811 15.0836 7.63189 14.7515 7.03818ZM7.53669 5.39002L8.7499 7.55589C8.90213 7.83139 "
    "8.90213 8.17337 8.7499 8.44884L7.53669 10.6147L7.47671 10.7192H7.59664H10.0277C10.5075 10.7192 10.895 11.1182 "
    "10.895 11.6122C10.895 12.1061 10.5075 12.5051 10.0277 12.5051H5.97748C5.6684 12.5051 5.38238 12.3341 5.22555 "
    "12.0587C5.07332 11.7832 5.07332 11.4412 5.22555 11.1657L6.97849 8.03562L6.99696 8.00237L6.97849 7.96914L5.22555 "
    "4.83905C5.07332 4.56358 5.07332 4.2216 5.22555 3.94611C5.37778 3.67063 5.6684 3.49963 5.97748 3.49963H10.0277C10.5075 "
    "3.49963 10.895 3.89861 10.895 4.39258C10.895 4.88655 10.5075 5.28552 10.0277 5.28552H7.59664H7.47671L7.53669 "
    '5.39002Z" fill="currentColor"/></svg>'
)

CSS = f"""
<style>
:root {{
  --wr-black:{BLACK}; --wr-pink:{PINK}; --wr-pink-light:{PINK_LIGHT}; --wr-pink-medium:{PINK_MEDIUM};
  --wr-purple:{PURPLE}; --wr-purple-light:{PURPLE_LIGHT}; --wr-purple-medium:{PURPLE_MEDIUM};
  --wr-grey:{GREY}; --wr-grey-light:{GREY_LIGHT}; --wr-grey-medium:{GREY_MEDIUM}; --wr-red:{RED};
}}
html, body, p, label, button, input, textarea, h1, h2, h3, li, td, th {{ font-family: Inter, sans-serif; }}
[data-testid="stIconMaterial"] {{ font-family: "Material Symbols Rounded" !important; }}
.block-container {{ max-width: 1500px; padding-top: 1.5rem; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* Titres Wiser : h1 2rem, h2 1.5rem, h3 1.25rem, medium */
h1, h2, h3 {{ font-weight: 500 !important; color: var(--wr-black); letter-spacing: 0; }}
h1 {{ font-size: 2rem !important; }}
h2 {{ font-size: 1.5rem !important; }}
h3 {{ font-size: 1.25rem !important; }}

/* Labels de formulaire : violets (WrForm) */
[data-testid="stWidgetLabel"] p {{ color: var(--wr-purple); font-size: 16px; font-weight: 400; }}

/* Champs : soulignés, sans cadre (Form/Controls Input) */
[data-testid="stNumberInputContainer"], [data-testid="stTextInputRootElement"],
[data-baseweb="input"], [data-baseweb="base-input"] {{
  border: none !important; border-radius: 0 !important; background: transparent !important;
}}
[data-testid="stNumberInputContainer"], [data-testid="stTextInputRootElement"] {{
  border-bottom: 1px solid var(--wr-grey) !important;
}}
[data-testid="stNumberInputContainer"]:focus-within, [data-testid="stTextInputRootElement"]:focus-within {{
  border-bottom-color: var(--wr-black) !important; box-shadow: 0 1px var(--wr-black);
}}
[data-testid="stNumberInputContainer"] input, [data-testid="stTextInputRootElement"] input {{
  background: transparent !important; padding-left: 0; font-weight: 500; font-size: 14px;
}}
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {{
  background: transparent !important; color: var(--wr-grey) !important;
}}
[data-baseweb="select"] > div {{
  border: none !important; border-bottom: 1px solid var(--wr-grey) !important; border-radius: 0 !important;
  background: transparent !important; font-weight: 500; font-size: 14px;
}}

/* Boutons Wiser : primaire rose, outlined gris */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
  border-radius: 8px; padding: 8px 16px; font-size: 16px; font-weight: 400;
  border: 1px solid var(--wr-grey-medium); background: #fff; color: var(--wr-black); box-shadow: none;
}}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{
  border-color: var(--wr-grey-medium); color: var(--wr-black); box-shadow: 2px 2px 4px 0 rgba(0,0,0,.15);
}}
button[kind="primary"], button[kind="primaryFormSubmit"] {{
  background: var(--wr-pink) !important; border-color: var(--wr-pink) !important; color: #fff !important;
}}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {{
  box-shadow: 2px 2px 4px 0 rgba(0,0,0,.15);
}}
button[kind="primary"]:active, button[kind="primaryFormSubmit"]:active {{
  background: linear-gradient(to right, var(--wr-pink), var(--wr-purple)) !important;
}}
button:focus-visible {{ outline: 2px solid var(--wr-purple) !important; outline-offset: 0; }}

/* Toggle buttons (Choice field / Radio toggle buttons) */
[data-testid="stButtonGroup"] button {{
  border: 1px solid var(--wr-grey-medium) !important; margin-left: -1px; color: var(--wr-grey) !important; background: #fff !important;
  font-size: 16px; padding: 8px 16px; box-shadow: none !important;
}}
[data-testid="stButtonGroup"] button[aria-checked="true"] {{
  background: var(--wr-purple-light) !important; color: var(--wr-purple) !important;
}}

/* Sections repliables (Details) : résumé en pastille violette */
[data-testid="stExpander"] details {{ border: none !important; }}
[data-testid="stExpander"] summary {{
  background: var(--wr-purple-light); color: var(--wr-purple); border-radius: 9999px;
  padding: 8px 16px; width: fit-content; margin-bottom: 1rem;
}}
[data-testid="stExpander"] summary p {{ font-size: 1.125rem; font-weight: 500; color: var(--wr-purple); }}
[data-testid="stExpander"] summary svg {{ color: var(--wr-purple); }}
[data-testid="stExpanderDetails"] {{ padding: 0 0 0.5rem 0; }}

/* Onglets (Tabs) : actif souligné violet */
[data-baseweb="tab-list"] {{ gap: 1rem; border-bottom: 1px solid var(--wr-grey); }}
[data-baseweb="tab"] {{ color: var(--wr-grey); padding: 16px 8px; }}
[data-baseweb="tab"]:hover {{ background: var(--wr-grey-light); color: var(--wr-black); }}
[data-baseweb="tab"][aria-selected="true"] {{ color: var(--wr-purple); box-shadow: inset 0 -4px var(--wr-purple); }}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{ display: none; }}

/* Composants maison */
.wr-header {{ display:flex; align-items:center; justify-content:space-between; gap:1rem;
  padding-bottom: 1rem; border-bottom: 1px solid var(--wr-grey-medium); margin-bottom: 1rem; }}
.wr-brand {{ display:flex; align-items:center; gap:.75rem; color: var(--wr-purple); }}
.wr-brand .wr-title {{ font-size: 1.5rem; font-weight: 500; color: var(--wr-black); line-height: 2rem; }}
.wr-brand .wr-sub {{ font-size: .875rem; color: var(--wr-grey); line-height: 1.25rem; }}
.wr-ruo {{ display:block; background: var(--wr-grey-light);
  border-left: 4px solid var(--wr-purple); border-radius: 8px; padding: 12px 16px; margin: .5rem 0 1.5rem;
  font-size: .875rem; line-height: 1.25rem; color: var(--wr-black); }}
.wr-ruo b {{ color: var(--wr-purple); font-weight: 700; }}
.wr-abstract {{ background: var(--wr-purple-light); border-radius: 8px; padding: 14px 18px;
  margin: 0 0 1.5rem; font-size: .875rem; line-height: 1.4rem; color: var(--wr-black); }}
.wr-abstract h4 {{ margin: 0 0 .4rem; font-size: .75rem; font-weight: 700; letter-spacing: .05em;
  text-transform: uppercase; color: var(--wr-purple); }}
.wr-abstract p {{ margin: 0; }}
.wr-pills {{ display:flex; flex-wrap:wrap; gap: .75rem; margin: .5rem 0 1rem; }}
.wr-pill {{ border-radius: 42px; padding: 16px 20px; display:inline-flex; flex-direction:column;
  align-items:flex-end; justify-content:center; min-width: 140px; }}
.wr-pill .v {{ font-size: 1.5rem; line-height: 2rem; font-weight: 500; }}
.wr-pill .l {{ font-size: .875rem; line-height: 1.25rem; }}
.wr-pill.pink {{ background: var(--wr-pink-light); color: var(--wr-pink); }}
.wr-pill.pink .l {{ color: var(--wr-pink-medium); }}
.wr-pill.purple {{ background: var(--wr-purple-light); color: var(--wr-purple); }}
.wr-pill.purple .l {{ color: var(--wr-purple-medium); }}
.wr-pill.grey {{ background: var(--wr-grey-light); color: var(--wr-black); }}
.wr-pill.grey .l {{ color: var(--wr-grey); }}
.wr-empty {{ border: 1px dashed var(--wr-grey-medium); border-radius: 8px; padding: 2.5rem 1.5rem;
  text-align:center; color: var(--wr-grey); }}
.wr-muted {{ color: var(--wr-grey); font-size: .875rem; }}
.wr-footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--wr-grey-medium);
  color: var(--wr-grey); font-size: .857rem; display:flex; gap:.5rem; align-items:center; }}
</style>
"""


def icon_img(size: int, color: str = PURPLE) -> str:
    """Icône Sêmeia en <img> (st.html filtre les <svg> inline)."""
    import base64

    svg = SEMEIA_ICON.format(s=size).replace("currentColor", color)
    b64 = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" width="{size}" height="{size}" alt="Sêmeia"/>'


def inject_css() -> None:
    st.html(CSS)


def header(title: str, subtitle: str) -> None:
    st.html(
        f"""<div class="wr-brand">{icon_img(40)}
        <div><div class="wr-title">{title}</div><div class="wr-sub">{subtitle}</div></div></div>"""
    )


def ruo_banner(html: str) -> None:
    st.html(f'<div class="wr-ruo">{html}</div>')


def abstract(title: str, body: str) -> None:
    st.html(f'<div class="wr-abstract"><h4>{title}</h4><p>{body}</p></div>')


def pills(items: list[tuple[str, str, str]]) -> None:
    """items = [(valeur, libellé, couleur pink|purple|grey)]"""
    body = "".join(
        f'<div class="wr-pill {c}"><span class="v">{v}</span><span class="l">{l}</span></div>' for v, l, c in items
    )
    st.html(f'<div class="wr-pills">{body}</div>')


def footer(text: str) -> None:
    st.html(f'<div class="wr-footer">{icon_img(16, GREY)}<span>{text}</span></div>')


# --- Plotly -------------------------------------------------------------------
pio.templates["wiser"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, sans-serif", size=13, color=BLACK),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        colorway=[PINK, PURPLE, TEAL, ORANGE, GREY],
        xaxis=dict(showgrid=False, linecolor=GREY_MEDIUM, ticks="outside", tickcolor=GREY_MEDIUM, zeroline=False),
        yaxis=dict(gridcolor=GREY_LIGHT, linecolor=GREY_MEDIUM, zeroline=False),
        hoverlabel=dict(bgcolor=WHITE, bordercolor=GREY_MEDIUM, font=dict(family="Inter, sans-serif", color=BLACK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
)
