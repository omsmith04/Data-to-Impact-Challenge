import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="NOVA Constraint Dashboard", layout="wide")

DATA_PATH = Path("data/curated/nova_cells.geojson")

# ---------- data helpers ----------
@st.cache_data
def load_featurecollection(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Create it or add the sample GeoJSON.")
    with path.open("r", encoding="utf-8") as f:
        fc = json.load(f)
    if fc.get("type") != "FeatureCollection" or "features" not in fc:
        raise ValueError("GeoJSON must be a FeatureCollection with a 'features' array.")
    return fc

def robust_minmax(vals: pd.Series) -> pd.Series:
    vals = pd.to_numeric(vals, errors="coerce")
    lo, hi = np.nanpercentile(vals, 1), np.nanpercentile(vals, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        return pd.Series(np.zeros(len(vals)), index=vals.index)
    x = (vals.clip(lo, hi) - lo) / (hi - lo)
    return x.fillna(x.median())

def tri_color(score_0_100: float, t_green: float, t_red: float):
    # RGBA 0-255 (deck.gl format)
    if score_0_100 <= t_green:
        return [46, 204, 113, 160]  # green
    if score_0_100 >= t_red:
        return [231, 76, 60, 160]   # red
    return [241, 196, 15, 160]      # yellow

def infer_map_center(fc: dict):
    # crude center: average of all coordinates in all polygons
    xs, ys = [], []
    for ft in fc["features"]:
        geom = ft.get("geometry") or {}
        if geom.get("type") != "Polygon":
            continue
        # coords: [ [ [lon,lat], ... ] ]  (outer ring first)
        ring = geom.get("coordinates", [[]])[0]
        for lon, lat in ring:
            xs.append(lon); ys.append(lat)
    if xs and ys:
        return float(np.mean(ys)), float(np.mean(xs))
    return 38.9, -77.4  # fallback NOVA-ish

# ---------- UI ----------
st.title("NOVA Data Center Siting: Weighted Constraint Map")

with st.sidebar:
    st.header("Weights (auto-normalized)")
    w_uhi = st.slider("Heat stress (UHI / LST proxy)", 0.0, 5.0, 2.0, 0.1)
    w_water = st.slider("Water stress", 0.0, 5.0, 2.5, 0.1)
    w_drought = st.slider("Drought", 0.0, 5.0, 1.5, 0.1)
    w_imperv = st.slider("Imperviousness", 0.0, 5.0, 1.0, 0.1)

    st.divider()
    st.header("Color thresholds")
    t_green = st.slider("Green if score ≤", 0, 100, 40, 1)
    t_red = st.slider("Red if score ≥", 0, 100, 70, 1)
    show_only_red = st.checkbox("Show only red cells", False)

# keys in properties (must exist in your GeoJSON)
FIELD_MAP = {
    "uhi": "uhi",
    "water": "water_stress",
    "drought": "drought",
    "impervious": "impervious",
}

# ---------- load + score ----------
fc = load_featurecollection(DATA_PATH)

# Build a scoring DataFrame from feature properties
props_list = [ft.get("properties", {}) for ft in fc["features"]]
df = pd.DataFrame(props_list)

missing = [FIELD_MAP[k] for k in FIELD_MAP if FIELD_MAP[k] not in df.columns]
if missing:
    st.error(
        "Your GeoJSON is missing required properties: "
        + ", ".join(missing)
        + "\n\nExpected at least: cell_id, uhi, water_stress, drought, impervious"
    )
    st.stop()

# Normalize weights
w = np.array([w_uhi, w_water, w_drought, w_imperv], dtype=float)
if w.sum() <= 0:
    w = np.ones_like(w)
w = w / w.sum()

# Normalize criteria to 0–1 (robust)
n_uhi = robust_minmax(df[FIELD_MAP["uhi"]])
n_water = robust_minmax(df[FIELD_MAP["water"]])
n_drought = robust_minmax(df[FIELD_MAP["drought"]])
n_imperv = robust_minmax(df[FIELD_MAP["impervious"]])

score01 = w[0]*n_uhi + w[1]*n_water + w[2]*n_drought + w[3]*n_imperv
score = (100.0 * score01).clip(0, 100)

df["score"] = score.round(2)
df["label"] = np.where(df["score"] >= t_red, "RED (avoid)",
               np.where(df["score"] <= t_green, "GREEN (better)", "YELLOW (caution)"))

# Push score + color back into GeoJSON feature properties for deck.gl
for i, ft in enumerate(fc["features"]):
    ft.setdefault("properties", {})
    ft["properties"]["score"] = float(df.loc[i, "score"])
    ft["properties"]["label"] = str(df.loc[i, "label"])
    ft["properties"]["fill_color"] = tri_color(float(df.loc[i, "score"]), t_green, t_red)

# Optional filter
if show_only_red:
    fc_map = {"type": "FeatureCollection",
              "features": [ft for ft in fc["features"] if ft["properties"]["score"] >= t_red]}
else:
    fc_map = fc

# ---------- render map ----------
lat0, lon0 = infer_map_center(fc_map)
view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=9, pitch=0, bearing=0)

layer = pdk.Layer(
    "GeoJsonLayer",
    data=fc_map,
    pickable=True,
    stroked=True,
    filled=True,
    get_fill_color="properties.fill_color",
    get_line_color=[20, 20, 20, 80],
    line_width_min_pixels=1,
)

tooltip = {
    "html": """
    <b>Cell</b>: {cell_id}<br/>
    <b>Score</b>: {score} / 100<br/>
    <b>Class</b>: {label}<br/>
    Heat: {uhi}<br/>
    Water stress: {water_stress}<br/>
    Drought: {drought}<br/>
    Impervious: {impervious}<br/>
    """,
    "style": {"backgroundColor": "white", "color": "black"}
}

c1, c2 = st.columns([2, 1], gap="large")

with c1:
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style="light",
            tooltip=tooltip,
        ),
        use_container_width=True,
        height=700,
    )  # Streamlit renders a PyDeck Deck object [web:91]

with c2:
    st.subheader("Top cells (highest score)")
    cols = ["cell_id", "score", "label", "uhi", "water_stress", "drought", "impervious"]
    show = df.copy()
    if "cell_id" not in show.columns:
        show["cell_id"] = [ft.get("properties", {}).get("cell_id", f"cell_{i}") for i, ft in enumerate(fc["features"])]
    st.dataframe(show[cols].sort_values("score", ascending=False).head(20), use_container_width=True)

    st.download_button(
        "Download scored CSV",
        data=show.to_csv(index=False).encode("utf-8"),
        file_name="nova_scored_cells.csv",
        mime="text/csv",
    )
