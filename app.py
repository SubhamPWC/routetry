
import os
import json
import streamlit as st
import pandas as pd
from optimization import (
    load_graph,
    compute_candidate_routes,
    recommend_route,
    summarize_routes,
)
from map_utils import make_map

st.set_page_config(page_title="Road Route Optimization (India)", layout="wide")
st.title("🚗 Road Route Optimization (India) — Distance · Time · Cost · Emissions")
st.caption("Not using Google Maps. Uses OpenStreetMap via OSMnx when online; includes an offline demo graph.")

# --- Static origin/destination choices (lat, lon) ---
STATIC_POINTS = {
    "Kolkata": (22.5726, 88.3639),
    "Bhubaneswar": (20.2961, 85.8245),
    "Ranchi": (23.3441, 85.3096),
    "Patna": (25.5941, 85.1376),
}

# Sidebar controls
st.sidebar.header("Route Inputs")
origin_name = st.sidebar.selectbox("From (Origin)", list(STATIC_POINTS.keys()), index=0)
dest_name = st.sidebar.selectbox("To (Destination)", list(STATIC_POINTS.keys()), index=1)
origin = STATIC_POINTS[origin_name]
dest = STATIC_POINTS[dest_name]
if origin == dest:
    st.sidebar.warning("Origin and destination are the same. Select different points.")

st.sidebar.header("Optimization Preferences")
objective = st.sidebar.selectbox(
    "Primary objective",
    ["balanced", "min_distance", "min_time", "min_cost", "min_emissions", "pareto"],
    index=0,
    help="Choose how to rank routes. 'pareto' shows non-dominated routes across all metrics."
)
k_routes = st.sidebar.slider("Number of route alternatives (k)", min_value=3, max_value=25, value=10)
use_offline_demo = st.sidebar.checkbox(
    "Use Offline Demo Graph (no internet)", value=False,
    help="If enabled, uses a small synthetic road graph. For real road names and more details, disable and ensure internet access to fetch OpenStreetMap data."
)

st.sidebar.header("Vehicle & Environment")
fuel_price_inr_per_l = st.sidebar.number_input("Fuel price (INR/L)", value=105.0, min_value=50.0, max_value=200.0)
fuel_consumption_l_per_100km = st.sidebar.number_input("Fuel consumption (L/100 km)", value=6.5, min_value=3.0, max_value=20.0)
emission_factor_g_co2_per_km = st.sidebar.number_input("Emission factor (g CO₂/km)", value=170.0, min_value=0.0, max_value=400.0)
prefer_highways = st.sidebar.checkbox("Prefer highways", value=True)
avoid_bridges_flyovers = st.sidebar.checkbox("Avoid bridges / flyovers", value=False)

st.sidebar.header("ML Speed Model (Optional)")
speed_model_path = st.sidebar.text_input(
    "Speed model file (models/speed_model.pkl)", value="models/speed_model.pkl",
    help="If present, a ML model adjusts segment speeds. Otherwise default speed per road type is used."
)

# Run optimization
with st.spinner("Loading road network and computing candidate routes..."):
    G = load_graph(origin, dest, use_offline_demo=use_offline_demo)
    routes = compute_candidate_routes(
        G,
        origin,
        dest,
        k=k_routes,
        fuel_price_inr_per_l=fuel_price_inr_per_l,
        fuel_consumption_l_per_100km=fuel_consumption_l_per_100km,
        emission_factor_g_co2_per_km=emission_factor_g_co2_per_km,
        prefer_highways=prefer_highways,
        avoid_bridges_flyovers=avoid_bridges_flyovers,
        speed_model_path=speed_model_path,
    )
    df_routes = summarize_routes(routes)
    recommended_idx, decision_tag = recommend_route(df_routes, objective)

# Show filters and outputs
col1, col2 = st.columns([0.55, 0.45])
with col1:
    st.subheader("🗺️ Map: Routes from {} to {}".format(origin_name, dest_name))
    fmap = make_map(routes, recommended_idx, origin, dest)
    st.components.v1.html(fmap.get_root().render(), height=650)

with col2:
    st.subheader("📋 Route Alternatives")
    st.write("**Decision tag:**", decision_tag)
    st.dataframe(
        df_routes[[
            "route_id",
            "optimized",
            "total_distance_km",
            "total_time_min",
            "total_cost_inr",
            "total_emissions_kg",
            "num_segments",
            "highway_share_pct",
            "has_flyover",
            "dominance_rank",
        ]].style.highlight_max(subset=["total_distance_km", "total_time_min", "total_cost_inr", "total_emissions_kg"], color="#ffe0e0")
    )

    csv_data = df_routes.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download routes CSV", csv_data, file_name="routes_{}_to_{}.csv".format(origin_name, dest_name), mime="text/csv")

# Show detailed segment table for the recommended route
st.subheader("🔎 Recommended Route Details")
rec_route = routes[recommended_idx]
seg_df = pd.DataFrame(rec_route["segments"])  # list of dict per edge
st.dataframe(seg_df[["name", "highway", "length_m", "maxspeed_kph", "is_flyover", "travel_time_min"]])

# Footer info
with st.expander("ℹ️ Notes"):
    st.markdown("""
    - **Data source**: When online, roads & names are fetched from OpenStreetMap via OSMnx. In offline demo, a small synthetic graph is used for illustration.
    - **Optimization**: Multi-criteria (distance, time, cost, emissions). Recommended route is chosen by selected objective, or balanced by normalized scores.
    - **ML**: Optional scikit-learn model can adjust segment speeds using road attributes if you provide `models/speed_model.pkl`.
    - **Flyover detection**: Based on OSM tags like `bridge=yes` or segment name containing 'Flyover'.
    """)
