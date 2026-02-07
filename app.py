import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="NOVA MCDA Map", layout="wide")

# --- OPTIONAL: reduce top padding/white space (CSS tweaks vary by Streamlit version) ---
st.markdown(
    """
    <style>
      section.stMain .block-container { padding-top: 0.8rem; padding-bottom: 0.5rem; }
      header.stAppHeader { background: rgba(0,0,0,0); }
    </style>
    """,
    unsafe_allow_html=True,
)  # CSS approach discussed by Streamlit users [web:358]

# ----------------------------
# Fixed layer paths (edit to match your repo)
# IMPORTANT: base layer must be your tessellating grid (many polygons)
# ----------------------------

#tessa-grid/northern_va_grid.geojson
LAYER_PATHS = {
    "base_grid": Path("data/curated/nova_grid.geojson"),
    "heat": Path("data/curated/nova_cells.geojson"),
    "water": Path("data/curated/nova_cells.geojson"),
    "eco": Path("land-use/land.geojson"),
    "infra": Path("data/curated/nova_cells.geojson"),
}

AOI_PATH = Path("aoi_layer/NoVA_Boundaries.geojson")  # outline only (optional)

ID_KEY = "cell_id"
VALUE_KEY = "value"  # change if your layers use a different metric property name

# ----------------------------
# Load + utilities
# ----------------------------
@st.cache_data
def load_fc(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.as_posix()}")
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        raise ValueError(f"Empty file: {path.as_posix()}")
    fc = json.loads(txt)
    if fc.get("type") != "FeatureCollection" or "features" not in fc:
        raise ValueError(f"Not a FeatureCollection: {path.as_posix()}")
    return fc

def index_by_id(fc: dict, id_key: str) -> dict[str, dict]:
    out = {}
    for ft in fc.get("features", []):
        cid = (ft.get("properties") or {}).get(id_key)
        if cid is not None:
            out[str(cid)] = ft
    return out

def robust_minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    good = s.dropna()
    if len(good) == 0:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    lo, hi = np.nanpercentile(good, 1), np.nanpercentile(good, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        return pd.Series(np.full(len(s), float(good.median())), index=s.index)
    x = (s.clip(lo, hi) - lo) / (hi - lo)
    return x.fillna(float(good.median()))

def softmax(w_raw: np.ndarray, gamma: float) -> np.ndarray:
    z = gamma * w_raw
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def tri_color(score: float, t_green: float, t_red: float):
    if score <= t_green:
        return [46, 204, 113, 175]   # green
    if score >= t_red:
        return [231, 76, 60, 175]    # red
    return [241, 196, 15, 175]       # yellow

def infer_center(fc: dict):
    xs, ys = [], []
    for ft in fc.get("features", []):
        geom = ft.get("geometry") or {}
        if geom.get("type") != "Polygon":
            continue
        ring = (geom.get("coordinates") or [[]])[0]
        for lon, lat in ring:
            xs.append(lon); ys.append(lat)
    if xs and ys:
        return float(np.mean(ys)), float(np.mean(xs))
    return 38.9, -77.4

# ----------------------------
# UI (simple)
# ----------------------------
st.title("NOVA Data Center Siting (Green/Yellow/Red)")

with st.sidebar:
    st.header("Weights")
    w_heat = st.slider("Heat stress", 0.0, 5.0, 2.0, 0.1)
    w_water = st.slider("Water stress", 0.0, 5.0, 2.5, 0.1)
    w_eco = st.slider("Eco / land use", 0.0, 5.0, 2.0, 0.1)
    w_infra = st.slider("Infrastructure", 0.0, 5.0, 1.5, 0.1)

    st.divider()
    st.header("Thresholds")
    t_green = st.slider("Green ≤", 0, 100, 40, 1)
    t_red = st.slider("Red ≥", 0, 100, 70, 1)

    with st.expander("Advanced (optional)", expanded=False):  # collapsible UI [web:368]
        st.caption("Nonlinear settings (hide these for most users).")
        gamma = st.slider("Weight nonlinearity γ", 0.0, 5.0, 1.5, 0.1)
        p = st.slider("Criterion power p", 0.5, 4.0, 2.0, 0.1)
        k = st.slider("Logistic steepness k", 0.5, 20.0, 8.0, 0.5)
        pivot = st.slider("Logistic pivot (0–1)", 0.0, 1.0, 0.5, 0.01)

# defaults if expander never opened (Streamlit will still define them only if block runs)
# So: set safe defaults here too:
gamma = locals().get("gamma", 1.5)
p = locals().get("p", 2.0)
k = locals().get("k", 8.0)
pivot = locals().get("pivot", 0.5)

# ----------------------------
# Load fixed files
# ----------------------------
base_fc = load_fc(LAYER_PATHS["base_grid"])
heat_fc = load_fc(LAYER_PATHS["heat"])
water_fc = load_fc(LAYER_PATHS["water"])
eco_fc = load_fc(LAYER_PATHS["eco"])
infra_fc = load_fc(LAYER_PATHS["infra"])
aoi_fc = load_fc(AOI_PATH) if AOI_PATH.exists() else None

# quick sanity: base grid must be “many polygons”
if len(base_fc.get("features", [])) < 20:
    st.warning("Your base_grid.geojson has very few features. To cover the map, base_grid must be a tessellating grid (lots of cells).")

idx_heat = index_by_id(heat_fc, ID_KEY)
idx_water = index_by_id(water_fc, ID_KEY)
idx_eco = index_by_id(eco_fc, ID_KEY)
idx_infra = index_by_id(infra_fc, ID_KEY)

rows = []
for ft in base_fc.get("features", []):
    props = ft.get("properties") or {}
    cid = props.get(ID_KEY)
    if cid is None:
        continue
    cid = str(cid)

    def get_val(idx):
        f = idx.get(cid)
        return (f.get("properties") or {}).get(VALUE_KEY, np.nan) if f else np.nan

    rows.append({
        "cell_id": cid,
        "heat": get_val(idx_heat),
        "water": get_val(idx_water),
        "eco": get_val(idx_eco),
        "infra": get_val(idx_infra),
    })

df = pd.DataFrame(rows)
if len(df) == 0:
    st.error(f"base_grid is missing properties.{ID_KEY} so the map can't be colored.")
    st.stop()

# Normalize each layer to 0–1 risk
n_heat = robust_minmax(df["heat"])
n_water = robust_minmax(df["water"])
n_eco = robust_minmax(df["eco"])
n_infra = robust_minmax(df["infra"])

# Make infrastructure a benefit by default: higher infra -> lower risk
n_infra = 1 - n_infra

w_raw = np.array([w_heat, w_water, w_eco, w_infra], dtype=float)
if w_raw.sum() <= 0:
    w_raw = np.ones_like(w_raw)
w_adj = softmax(w_raw, gamma=gamma)

x = np.vstack([n_heat, n_water, n_eco, n_infra]).T
x_pow = np.power(np.clip(x, 0, 1), p)
linear = x_pow @ w_adj
score01 = sigmoid(k * (linear - pivot))
score = (100.0 * score01).clip(0, 100)

df["score"] = score

# Push colors into base_fc feature properties so GeoJsonLayer can fill them. [web:93]
score_by_id = dict(zip(df["cell_id"], df["score"]))
for ft in base_fc["features"]:
    props = ft.setdefault("properties", {})
    cid = props.get(ID_KEY)
    if cid is None:
        continue
    sc = float(score_by_id.get(str(cid), np.nan))
    if not np.isfinite(sc):
        continue
    props["score"] = sc
    props["fill_color"] = tri_color(sc, t_green, t_red)

# ----------------------------
# Render full-width map
# ----------------------------
lat0, lon0 = infer_center(base_fc)
view = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=9)

main_layer = pdk.Layer(
    "GeoJsonLayer",
    data=base_fc,
    pickable=True,
    filled=True,
    stroked=False,  # removes grid outlines = more “solid coverage” look
    get_fill_color="properties.fill_color",
)

layers = [main_layer]
if aoi_fc:
    layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            data=aoi_fc,
            pickable=False,
            filled=False,
            stroked=True,
            get_line_color=[0, 0, 0, 220],
            line_width_min_pixels=2,
        )
    )

tooltip = {"html": "<b>Score</b>: {score} / 100<br/><b>Cell</b>: {cell_id}"}

st.pydeck_chart(
    pdk.Deck(layers=layers, initial_view_state=view, map_style="light", tooltip=tooltip),
    use_container_width=True,
    height=780,
)  # Streamlit PyDeck rendering [web:91]
