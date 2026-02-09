from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="NOVA Site Suitability Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Paths (your repo)
# ----------------------------
BASE_GRID_PATH = Path("tessa-grid/northern_va_grid.geojson")

HEAT_PATH = Path("heat.geojson")
WATER_PATH = Path("nova_water_stress_clipped.geojson")

AOI_PATH = Path("aoi_layer/NoVA_Boundaries.geojson")

# Eco inputs
LAND_COVER_PATH = Path("land-use/land_cover.geojson")
FCT_IMP_PATH = Path("land-use/fct_imp.geojson")

# Infra-ish inputs (zoning)
ZONING_POLYGONS_PATH = Path("zoning-layer/Zoning_Polygons.geojson")
ZONING_OVERLAY_PATH = Path("zoning-layer/Zoning_Overlay_Districts.geojson")
LOUDOUN_ZONING_PATH = Path("zoning-layer/Loudoun_Zoning.geojson")
DC_OPP_ZONES_PATH = Path("zoning-layer/Data_Center_Opportunity_Zone_Overlay_Districts.geojson")

# ----------------------------
# UI header
# ----------------------------
st.title("NOVA Site Suitability Dashboard")
st.caption("Weighted scoring (0–100). Lower score = more suitable.")

st.markdown(
    """
<span style="display:inline-block;padding:0.15rem 0.55rem;border-radius:999px;font-weight:600;background:#2ecc71;color:white;">Green</span>
<span style="display:inline-block;padding:0.15rem 0.55rem;border-radius:999px;font-weight:600;background:#f1c40f;color:black;">Yellow</span>
<span style="display:inline-block;padding:0.15rem 0.55rem;border-radius:999px;font-weight:600;background:#e74c3c;color:white;">Red</span>
""",
    unsafe_allow_html=True,
)

with st.expander("How to use", expanded=True):
    st.write(
        "1) Pick a preset or adjust the 4 weights (Heat/Water/Infra/Eco).\n"
        "2) The map recolors on every change.\n"
        "3) Hover cells to see score.\n"
        "Notes: Heat is usually point-based; Water/Eco/Infra are polygon overlays in your data."
    )

# ----------------------------
# GeoJSON utilities
# ----------------------------
@st.cache_data
def load_fc(path: Path) -> dict:
    if not path.exists():
        return {"type": "FeatureCollection", "features": []}
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_cell_id_from_cellid(fc: dict):
    for ft in fc.get("features", []):
        props = ft.setdefault("properties", {})
        if "cell_id" not in props and "cellid" in props:
            props["cell_id"] = str(props["cellid"])
        if "cell_id" in props:
            props["cell_id"] = str(props["cell_id"])


def mercator_xy_to_lonlat(x, y):
    R = 6378137.0
    lon = (x / R) * 180.0 / math.pi
    lat = (2.0 * math.atan(math.exp(y / R)) - math.pi / 2.0) * 180.0 / math.pi
    return lon, lat


@st.cache_data
def convert_fc_3857_to_4326(fc: dict) -> dict:
    crs_name = ((fc.get("crs") or {}).get("properties") or {}).get("name", "")
    looks_3857 = "EPSG:3857" in str(crs_name)

    if not looks_3857:
        for ft in fc.get("features", [])[:5]:
            geom = ft.get("geometry") or {}
            coords = geom.get("coordinates")
            if not coords:
                continue
            c = coords
            while isinstance(c, list) and c and isinstance(c[0], list):
                c = c[0]
            if isinstance(c, list) and len(c) >= 2 and all(isinstance(v, (int, float)) for v in c[:2]):
                if abs(c[0]) > 200 or abs(c[1]) > 200:
                    looks_3857 = True
                break

    if not looks_3857:
        return fc

    def convert_coords(coords):
        if isinstance(coords, list) and len(coords) >= 2 and isinstance(coords[0], (int, float)):
            lon, lat = mercator_xy_to_lonlat(coords[0], coords[1])
            return [lon, lat]
        return [convert_coords(c) for c in coords]

    out = json.loads(json.dumps(fc))
    for ft in out.get("features", []):
        geom = ft.get("geometry") or {}
        if "coordinates" in geom:
            geom["coordinates"] = convert_coords(geom["coordinates"])
    out.pop("crs", None)
    return out


def flatten_first_pair(coords):
    if isinstance(coords, list) and len(coords) == 2 and all(isinstance(v, (int, float)) for v in coords):
        return coords
    if isinstance(coords, list):
        for x in coords:
            r = flatten_first_pair(x)
            if r is not None:
                return r
    return None


def looks_lonlat(fc: dict) -> bool:
    try:
        for feat in fc.get("features", [])[:5]:
            geom = feat.get("geometry", {}) or {}
            if "coordinates" not in geom:
                continue
            pair = flatten_first_pair(geom["coordinates"])
            if pair is None:
                continue
            return abs(pair[0]) <= 180 and abs(pair[1]) <= 90
    except Exception:
        return True
    return True


def polygon_centroid_lonlat(ft: dict):
    geom = ft.get("geometry") or {}
    coords = geom.get("coordinates") or []
    ring = None
    if geom.get("type") == "Polygon" and coords:
        ring = coords[0]
    elif geom.get("type") == "MultiPolygon" and coords and coords[0]:
        ring = coords[0][0]
    if not ring:
        return None
    xs = [p[0] for p in ring if isinstance(p, list) and len(p) >= 2]
    ys = [p[1] for p in ring if isinstance(p, list) and len(p) >= 2]
    if not xs or not ys:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def infer_center(fc: dict):
    xs, ys = [], []
    for ft in fc.get("features", []):
        c = polygon_centroid_lonlat(ft)
        if c:
            xs.append(c[0])
            ys.append(c[1])
    if xs and ys:
        return float(np.mean(ys)), float(np.mean(xs))
    return 38.9, -77.4


def geojson_geom_types(fc: dict) -> set[str]:
    out = set()
    for ft in fc.get("features", [])[:200]:
        g = (ft.get("geometry") or {}).get("type")
        if isinstance(g, str):
            out.add(g)
    return out


def point_value_key(fc: dict, candidates: list[str]) -> str | None:
    feats = fc.get("features", [])
    if not feats:
        return None
    for ft in feats[:50]:
        props = ft.get("properties") or {}
        for k in candidates:
            if k in props:
                return k
    return None


def pick_numeric_prop(fc: dict, preferred: list[str] | None = None) -> str | None:
    preferred = preferred or []
    for ft in fc.get("features", [])[:200]:
        props = ft.get("properties") or {}
        for k in preferred:
            if k in props and isinstance(props[k], (int, float)) and np.isfinite(props[k]):
                return k

    for ft in fc.get("features", [])[:200]:
        props = ft.get("properties") or {}
        for k, v in props.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                return k
    return None


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


def tri_color(score_0_100: float, t_green: float = 33.0, t_red: float = 66.0):
    # score_0_100 is already 0..100 (lower is better)
    if score_0_100 <= t_green:
        return [46, 204, 113, 175]
    if score_0_100 >= t_red:
        return [231, 76, 60, 175]
    return [241, 196, 15, 175]


def assign_points_to_cells_nearest(points_fc: dict, cell_centroids: pd.DataFrame, value_prop: str) -> pd.DataFrame:
    pts = []
    for ft in points_fc.get("features", []):
        geom = ft.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if not (isinstance(coords, list) and len(coords) >= 2):
            continue
        lon, lat = float(coords[0]), float(coords[1])
        val = (ft.get("properties") or {}).get(value_prop, np.nan)
        pts.append((lon, lat, val))

    if not pts:
        return pd.DataFrame({"cell_id": [], "value": []})

    P = np.array([[p[0], p[1]] for p in pts], dtype=float)
    V = np.array([p[2] for p in pts], dtype=float)

    C = cell_centroids[["lon", "lat"]].to_numpy(dtype=float)
    ids = cell_centroids["cell_id"].to_numpy()

    out_ids = []
    for i in range(P.shape[0]):
        d2 = np.sum((C - P[i]) ** 2, axis=1)
        j = int(np.argmin(d2))
        out_ids.append(ids[j])

    tmp = pd.DataFrame({"cell_id": out_ids, "value": V})
    tmp["value"] = pd.to_numeric(tmp["value"], errors="coerce")
    return tmp.groupby("cell_id", as_index=False)["value"].mean()


def polygon_to_cells_nearest(poly_fc: dict, cell_centroids: pd.DataFrame, value_prop: str | None = None) -> pd.DataFrame:
    feats = []
    for ft in poly_fc.get("features", []):
        geom = ft.get("geometry") or {}
        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue

        c = polygon_centroid_lonlat(ft)
        if c is None:
            continue

        lon, lat = c
        if value_prop is None:
            val = 1.0
        else:
            val = (ft.get("properties") or {}).get(value_prop, np.nan)

        feats.append((lon, lat, val))

    if not feats:
        return pd.DataFrame({"cell_id": [], "value": []})

    P = np.array([[p[0], p[1]] for p in feats], dtype=float)
    V = np.array([p[2] for p in feats], dtype=float)

    C = cell_centroids[["lon", "lat"]].to_numpy(dtype=float)
    ids = cell_centroids["cell_id"].to_numpy()

    out_ids = []
    for i in range(P.shape[0]):
        d2 = np.sum((C - P[i]) ** 2, axis=1)
        j = int(np.argmin(d2))
        out_ids.append(ids[j])

    tmp = pd.DataFrame({"cell_id": out_ids, "value": V})
    tmp["value"] = pd.to_numeric(tmp["value"], errors="coerce")
    return tmp.groupby("cell_id", as_index=False)["value"].mean()


def scored_fc_light(base_fc_4326: dict, score_by_id: dict[str, float], t_green: float = 33.0, t_red: float = 66.0) -> dict:
    feats = []
    for ft in base_fc_4326.get("features", []):
        props0 = ft.get("properties") or {}
        cid = props0.get("cell_id")
        if cid is None:
            continue
        cid = str(cid)

        sc = float(score_by_id.get(cid, np.nan))
        fill = [160, 160, 160, 70] if not np.isfinite(sc) else tri_color(sc, t_green, t_red)

        # deck.gl GeoJsonLayer expects [r,g,b,(a)] 0..255 arrays for colors. [web:561]
        feats.append(
            {
                "type": "Feature",
                "geometry": ft.get("geometry"),
                "properties": {
                    "cell_id": cid,
                    "score": None if not np.isfinite(sc) else float(sc),
                    "fill_color": fill,
                    # Tooltip-friendly keys
                    "cell": cid,
                    "score_txt": "" if not np.isfinite(sc) else f"{float(sc):.2f}",
                },
                # Top-level duplicates for tooltip substitution stability
                "cell": cid,
                "score": None if not np.isfinite(sc) else float(sc),
                "score_txt": "" if not np.isfinite(sc) else f"{float(sc):.2f}",
            }
        )
    return {"type": "FeatureCollection", "features": feats}


# ----------------------------
# Sidebar (presets + 4 sliders) — Custom default
# ----------------------------
PRESETS = {
    "Custom": None,
    "Balanced (25/25/25/25)": (25, 25, 25, 25),
    "Heat-heavy (45/20/20/15)": (45, 20, 20, 15),
    "Water-heavy (20/45/20/15)": (20, 45, 20, 15),
    "Infra-heavy (20/20/45/15)": (20, 20, 45, 15),
    "Eco-heavy (20/20/15/45)": (20, 20, 15, 45),
}


def _init_state():
    if "lock_weights" not in st.session_state:
        st.session_state.lock_weights = True

    if "w_heat" not in st.session_state:
        st.session_state.w_heat = 25
    if "w_water" not in st.session_state:
        st.session_state.w_water = 25
    if "w_infra" not in st.session_state:
        st.session_state.w_infra = 25
    if "w_eco" not in st.session_state:
        st.session_state.w_eco = 25

    if "preset" not in st.session_state:
        st.session_state.preset = "Custom"  # default back to Custom


def _normalize_to_100(changed_key: str):
    keys = ["w_heat", "w_water", "w_infra", "w_eco"]
    changed_val = int(st.session_state[changed_key])
    changed_val = max(0, min(100, changed_val))

    others = [k for k in keys if k != changed_key]
    other_vals = [int(st.session_state[k]) for k in others]
    other_sum = sum(other_vals)

    remaining = 100 - changed_val
    if other_sum <= 0:
        base = remaining // 3
        vals = [base, base, remaining - 2 * base]
    else:
        scaled = [remaining * (v / other_sum) for v in other_vals]
        vals = [int(round(x)) for x in scaled]
        vals[-1] += (remaining - sum(vals))

    st.session_state[changed_key] = changed_val
    for k, v in zip(others, vals):
        st.session_state[k] = max(0, min(100, int(v)))


def apply_preset(name: str):
    tpl = PRESETS.get(name)
    if tpl is None:
        return
    st.session_state.w_heat, st.session_state.w_water, st.session_state.w_infra, st.session_state.w_eco = tpl


with st.sidebar:
    st.header("Weights")
    _init_state()

    def _on_preset_change():
        apply_preset(st.session_state.preset)

    st.selectbox("Preset", list(PRESETS.keys()), key="preset", on_change=_on_preset_change)

    st.toggle("Lock total = 100", key="lock_weights")

    def _cb_factory(k):
        def _cb():
            # Any manual slider movement implies you're now in Custom
            st.session_state.preset = "Custom"
            if st.session_state.lock_weights:
                _normalize_to_100(k)

        return _cb

    st.slider("Heat", 0, 100, key="w_heat", step=5, on_change=_cb_factory("w_heat"))
    st.slider("Water", 0, 100, key="w_water", step=5, on_change=_cb_factory("w_water"))
    st.slider("Infrastructure", 0, 100, key="w_infra", step=5, on_change=_cb_factory("w_infra"))
    st.slider("Eco", 0, 100, key="w_eco", step=5, on_change=_cb_factory("w_eco"))

    total = int(st.session_state.w_heat) + int(st.session_state.w_water) + int(st.session_state.w_infra) + int(st.session_state.w_eco)
    st.write(f"Total: {total}%")
    if total != 100:
        st.error("Weights must sum to 100%.")
        st.stop()

# ----------------------------
# Load core data
# ----------------------------
base_fc_raw = load_fc(BASE_GRID_PATH)
ensure_cell_id_from_cellid(base_fc_raw)
base_fc_4326 = convert_fc_3857_to_4326(base_fc_raw)

heat_fc = load_fc(HEAT_PATH)
water_fc = load_fc(WATER_PATH)

land_cover_fc = load_fc(LAND_COVER_PATH)
fct_imp_fc = load_fc(FCT_IMP_PATH)

opp_fc = load_fc(DC_OPP_ZONES_PATH)
zoning_poly_fc = load_fc(ZONING_POLYGONS_PATH)
zoning_overlay_fc = load_fc(ZONING_OVERLAY_PATH)
loudoun_fc = load_fc(LOUDOUN_ZONING_PATH)

# Convert overlays if needed (rendering expects lon/lat)
for _fc in [heat_fc, water_fc, land_cover_fc, fct_imp_fc, opp_fc, zoning_poly_fc, zoning_overlay_fc, loudoun_fc]:
    if _fc and (not looks_lonlat(_fc)):
        _fc.update(convert_fc_3857_to_4326(_fc))

# ----------------------------
# Build centroids from grid
# ----------------------------
cell_rows = []
for ft in base_fc_4326.get("features", []):
    props = ft.get("properties") or {}
    cid = props.get("cell_id")
    c = polygon_centroid_lonlat(ft)
    if cid is None or c is None:
        continue
    cell_rows.append({"cell_id": str(cid), "lon": c[0], "lat": c[1]})
cells = pd.DataFrame(cell_rows)
if len(cells) == 0:
    st.error("Could not compute cell centroids from base grid.")
    st.stop()

# ----------------------------
# Heat -> cells
# ----------------------------
heat_types = geojson_geom_types(heat_fc)
heat_df = pd.DataFrame({"cell_id": [], "heat": []})

if "Point" in heat_types:
    heat_key = point_value_key(heat_fc, ["heat_normalized", "heat", "value"])
    if heat_key:
        heat_df = assign_points_to_cells_nearest(heat_fc, cells, heat_key).rename(columns={"value": "heat"})
else:
    heat_key = pick_numeric_prop(heat_fc, preferred=["heat_normalized", "heat", "value"])
    if heat_key:
        heat_df = polygon_to_cells_nearest(heat_fc, cells, value_prop=heat_key).rename(columns={"value": "heat"})

# ----------------------------
# Water -> cells
# ----------------------------
water_types = geojson_geom_types(water_fc)
water_df = pd.DataFrame({"cell_id": [], "water": []})

water_prop = pick_numeric_prop(
    water_fc,
    preferred=["water_stress_norm", "water_stress_raw", "water_stress_normalized", "water_normalized", "value"],
)

if ("Polygon" in water_types or "MultiPolygon" in water_types) and water_prop:
    water_df = polygon_to_cells_nearest(water_fc, cells, value_prop=water_prop).rename(columns={"value": "water"})
elif "Point" in water_types and water_prop:
    water_df = assign_points_to_cells_nearest(water_fc, cells, water_prop).rename(columns={"value": "water"})

# ----------------------------
# Eco -> cells
# ----------------------------
eco_cover_prop = pick_numeric_prop(land_cover_fc, preferred=["lcvalue", "value"])
eco_imp_prop = pick_numeric_prop(fct_imp_fc)

eco_cover = polygon_to_cells_nearest(land_cover_fc, cells, value_prop=eco_cover_prop).rename(columns={"value": "eco_cover"})
eco_imp = polygon_to_cells_nearest(fct_imp_fc, cells, value_prop=eco_imp_prop).rename(columns={"value": "eco_imp"})

# ----------------------------
# Infra (zoning) -> cells (presence-only)
# ----------------------------
infra_opp = polygon_to_cells_nearest(opp_fc, cells, value_prop=None).rename(columns={"value": "infra_opp"})
infra_z_overlay = polygon_to_cells_nearest(zoning_overlay_fc, cells, value_prop=None).rename(columns={"value": "infra_zov"})
infra_z_polys = polygon_to_cells_nearest(zoning_poly_fc, cells, value_prop=None).rename(columns={"value": "infra_zpol"})
infra_loudoun = polygon_to_cells_nearest(loudoun_fc, cells, value_prop=None).rename(columns={"value": "infra_loud"})

# ----------------------------
# Master table
# ----------------------------
df = cells[["cell_id"]].copy()
df = df.merge(heat_df, on="cell_id", how="left")
df = df.merge(water_df, on="cell_id", how="left")
df = df.merge(eco_cover, on="cell_id", how="left")
df = df.merge(eco_imp, on="cell_id", how="left")
df = df.merge(infra_opp, on="cell_id", how="left")
df = df.merge(infra_z_overlay, on="cell_id", how="left")
df = df.merge(infra_z_polys, on="cell_id", how="left")
df = df.merge(infra_loudoun, on="cell_id", how="left")

for col in ["infra_opp", "infra_zov", "infra_zpol", "infra_loud"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

# ----------------------------
# Build 4 criteria signals (0..1 where 1 = worse)
# ----------------------------
n_heat = robust_minmax(df["heat"]) if "heat" in df else pd.Series(0.5, index=df.index)
n_water = robust_minmax(df["water"]) if "water" in df else pd.Series(0.5, index=df.index)

eco_cover_norm = robust_minmax(df["eco_cover"]) if "eco_cover" in df else pd.Series(0.0, index=df.index)
eco_imp_norm = robust_minmax(df["eco_imp"]) if "eco_imp" in df else pd.Series(0.0, index=df.index)
n_eco = (0.6 * eco_cover_norm + 0.4 * eco_imp_norm).clip(0, 1)

# Infra presence is a BENEFIT → convert to "risk" by inverting
infra_raw = (
    1.0 * df.get("infra_opp", 0.0)
    + 0.3 * df.get("infra_zov", 0.0)
    + 0.2 * df.get("infra_zpol", 0.0)
    + 0.2 * df.get("infra_loud", 0.0)
)
infra_benefit = robust_minmax(pd.Series(infra_raw, index=df.index))
n_infra = (1.0 - infra_benefit).clip(0, 1)

# ----------------------------
# Weighted score 0..100
# ----------------------------
w_heat = st.session_state.w_heat / 100.0
w_water = st.session_state.w_water / 100.0
w_infra = st.session_state.w_infra / 100.0
w_eco = st.session_state.w_eco / 100.0

score01 = (w_heat * n_heat + w_water * n_water + w_infra * n_infra + w_eco * n_eco).clip(0, 1)
df["score"] = 100.0 * score01

score_by_id = dict(zip(df["cell_id"].astype(str), df["score"].astype(float)))

# ----------------------------
# Map layer
# ----------------------------
# Fixed thresholds makes colors stable/meaningful across runs (0-100 scale).
T_GREEN = 33.0
T_RED = 66.0

scored_fc = scored_fc_light(base_fc_4326, score_by_id, t_green=T_GREEN, t_red=T_RED)

view_lat, view_lon = infer_center(base_fc_4326)
view_state = pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=9, pitch=0)

grid_layer = pdk.Layer(
    "GeoJsonLayer",
    data=scored_fc,
    stroked=True,
    filled=True,
    get_fill_color="properties.fill_color",
    get_line_color=[50, 50, 50, 40],
    line_width_min_pixels=1,
    pickable=True,
    auto_highlight=True,
)

# deck.gl color arrays are [r,g,b,(a)] in 0..255. [web:561]
tooltip = {
    "html": "<b>Cell</b>: {cell}<br/><b>Score</b>: {score_txt}",
    "style": {"backgroundColor": "white", "color": "black"},
}

st.pydeck_chart(pdk.Deck(layers=[grid_layer], initial_view_state=view_state, tooltip=tooltip), use_container_width=True)

# ----------------------------
# Small table for sanity
# ----------------------------
with st.expander("Top/bottom cells", expanded=False):
    show_cols = ["cell_id", "score", "heat", "water", "eco_cover", "eco_imp", "infra_opp", "infra_zov", "infra_zpol", "infra_loud"]
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[show_cols].sort_values("score").head(20), use_container_width=True)
    st.dataframe(df[show_cols].sort_values("score", ascending=False).head(20), use_container_width=True)
