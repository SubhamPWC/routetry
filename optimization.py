
import math
import os
from typing import Dict, Tuple, List, Optional
import networkx as nx
import numpy as np
import pandas as pd

# If available, use osmnx to fetch real road networks
try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except Exception:
    OSMNX_AVAILABLE = False

DEFAULT_SPEED_KPH_BY_HIGHWAY = {
    # conservative defaults; OSM may provide maxspeed per edge
    "motorway": 90,
    "trunk": 80,
    "primary": 60,
    "secondary": 50,
    "tertiary": 40,
    "unclassified": 35,
    "residential": 30,
    "service": 25,
    "motorway_link": 60,
    "trunk_link": 50,
    "primary_link": 45,
    "secondary_link": 35,
    "tertiary_link": 30,
}

def load_graph(origin: Tuple[float, float], dest: Tuple[float, float], use_offline_demo: bool=False):
    """Load a road network graph. If offline demo, return a small synthetic graph.
    Otherwise use OSMnx to fetch graph for a bbox around origin & dest.
    """
    if use_offline_demo or not OSMNX_AVAILABLE:
        return _build_demo_graph(origin, dest)

    # compute bbox covering origin & dest plus buffer
    lat_min = min(origin[0], dest[0]) - 0.5
    lat_max = max(origin[0], dest[0]) + 0.5
    lon_min = min(origin[1], dest[1]) - 0.5
    lon_max = max(origin[1], dest[1]) + 0.5
    try:
        G = ox.graph_from_bbox(lat_max, lat_min, lon_max, lon_min, network_type='drive')
        G = ox.add_edge_lengths(G)
        return G
    except Exception:
        # fallback: demo graph
        return _build_demo_graph(origin, dest)


def _build_demo_graph(origin, dest):
    """Construct a small synthetic road graph between origin & dest with example road names.
    This is for offline demonstration only.
    """
    G = nx.MultiDiGraph()
    # create a few nodes roughly between Kolkata and Bhubaneswar corridor for demo
    nodes = {
        1: {"y": origin[0], "x": origin[1]},  # Origin
        2: {"y": 22.637, "x": 88.225},  # Kona Expressway
        3: {"y": 22.517, "x": 87.982},  # NH16 entry
        4: {"y": 21.943, "x": 86.738},  # Kharagpur area
        5: {"y": 21.518, "x": 86.928},  # Balasore bypass
        6: {"y": 20.948, "x": 86.092},  # Cuttack bypass
        7: {"y": dest[0], "x": dest[1]},  # Destination
        8: {"y": 22.575, "x": 88.346},  # Vidyasagar Setu Flyover
    }
    for nid, attrs in nodes.items():
        G.add_node(nid, **attrs)

    def add_edge(u, v, length_km, name, highway, maxspeed_kph, is_flyover=False):
        G.add_edge(
            u, v, key=len(G.edges()),
            length=length_km*1000,
            name=name,
            highway=highway,
            maxspeed=maxspeed_kph,
            bridge='yes' if is_flyover else None,
        )

    # build demo paths
    add_edge(1, 8, 4.5, "Vidyasagar Setu Flyover", "primary", 60, is_flyover=True)
    add_edge(8, 2, 10.0, "Kona Expressway", "trunk", 80)
    add_edge(2, 3, 20.0, "NH16", "motorway", 90)
    add_edge(3, 4, 120.0, "NH16", "motorway", 90)
    add_edge(4, 5, 110.0, "NH16", "motorway", 90)
    add_edge(5, 6, 100.0, "NH16", "motorway", 90)
    add_edge(6, 7, 25.0, "NH16", "motorway", 90)

    # alternative via city roads
    add_edge(1, 2, 14.0, "Andul Road", "primary", 50)
    add_edge(2, 4, 180.0, "NH16", "motorway", 90)
    add_edge(4, 7, 120.0, "NH16", "motorway", 90)

    # small roads route
    add_edge(1, 3, 30.0, "State Highway 4", "secondary", 45)
    add_edge(3, 5, 210.0, "Old Kolkata-Balasore Rd", "secondary", 40)
    add_edge(5, 7, 120.0, "Cuttack-BBSR Rd", "primary", 55)

    # add reverse edges to allow bidirectional travel
    edges_to_copy = list(G.edges(keys=True, data=True))
    for u, v, k, d in edges_to_copy:
        G.add_edge(v, u, key=len(G.edges()), **d)

    return G


def _edge_speed_kph(data, speed_model=None):
    # choose maxspeed if available else default by highway
    speed = None
    if isinstance(data.get('maxspeed'), (int, float)):
        speed = float(data['maxspeed'])
    else:
        hw = data.get('highway')
        if isinstance(hw, list):
            hw = hw[0]
        speed = DEFAULT_SPEED_KPH_BY_HIGHWAY.get(hw, 35)
    # ML adjustment if model available
    if speed_model is not None:
        # features: highway type one-hot + is_bridge + length
        hw = data.get('highway')
        if isinstance(hw, list):
            hw = hw[0]
        is_bridge = 1 if (data.get('bridge') in ['yes', True]) else 0
        length_m = float(data.get('length', 0.0))
        X = pd.DataFrame([{ 'length_m': length_m, 'is_bridge': is_bridge, 'highway': hw }])
        # simple encoding: map highway to ordinal for model expecting numeric
        X['highway_code'] = X['highway'].map({
            'motorway': 6, 'trunk': 5, 'primary': 4, 'secondary': 3, 'tertiary': 2, 'residential': 1,
            'service': 0, 'unclassified': 1, 'motorway_link': 4, 'primary_link': 3, 'secondary_link': 2, 'tertiary_link': 2
        }).fillna(1)
        X_model = X[['length_m', 'is_bridge', 'highway_code']]
        try:
            speed_pred = float(speed_model.predict(X_model)[0])
            # blend predicted with base speed to avoid extremes
            speed = 0.6*speed + 0.4*max(15.0, min(100.0, speed_pred))
        except Exception:
            pass
    return speed


def compute_candidate_routes(
    G,
    origin: Tuple[float, float],
    dest: Tuple[float, float],
    k: int = 10,
    fuel_price_inr_per_l: float = 105.0,
    fuel_consumption_l_per_100km: float = 6.5,
    emission_factor_g_co2_per_km: float = 170.0,
    prefer_highways: bool = True,
    avoid_bridges_flyovers: bool = False,
    speed_model_path: Optional[str] = None,
) -> List[Dict]:
    """Compute up to k candidate routes and annotate metrics & segments.
    Returns list of route dicts with per-edge segments and totals.
    """
    # find nearest nodes to origin & dest
    try:
        origin_node = ox.nearest_nodes(G, origin[1], origin[0]) if OSMNX_AVAILABLE else list(G.nodes())[0]
        dest_node = ox.nearest_nodes(G, dest[1], dest[0]) if OSMNX_AVAILABLE else list(G.nodes())[-1]
    except Exception:
        origin_node = list(G.nodes())[0]
        dest_node = list(G.nodes())[-1]

    # load optional ML speed model
    speed_model = None
    if speed_model_path and os.path.exists(speed_model_path):
        try:
            import joblib
            speed_model = joblib.load(speed_model_path)
        except Exception:
            speed_model = None

    # prepare a weight for route generation
    def edge_weight(u, v, data):
        # base penalties to steer preferences
        highway = data.get('highway')
        if isinstance(highway, list):
            highway = highway[0]
        is_bridge = (data.get('bridge') in ['yes', True])
        length_m = float(data.get('length', 0.0))
        speed_kph = _edge_speed_kph(data, speed_model)
        travel_time_min = (length_m/1000.0) / max(5.0, speed_kph) * 60.0
        # cost and emission per edge
        cost_inr = (length_m/1000.0) * (fuel_consumption_l_per_100km/100.0) * fuel_price_inr_per_l
        emissions_kg = (length_m/1000.0) * (emission_factor_g_co2_per_km/1000.0)
        # preferences
        highway_bonus = -0.02*length_m if (prefer_highways and highway in ["motorway", "trunk"]) else 0.0
        bridge_penalty = 0.05*length_m if (avoid_bridges_flyovers and is_bridge) else 0.0
        return travel_time_min + 0.001*length_m + 0.0002*cost_inr + 0.0002*emissions_kg + highway_bonus + bridge_penalty

    # k-shortest simple paths by weight
    paths_gen = nx.shortest_simple_paths(G, origin_node, dest_node, weight=lambda u, v, d: edge_weight(u, v, d))
    routes = []
    for i, path in enumerate(paths_gen):
        if i >= k:
            break
        segments = []
        total_length_m = 0.0
        total_time_min = 0.0
        total_cost_inr = 0.0
        total_emissions_kg = 0.0
        num_flyovers = 0
        highway_len_m = 0.0

        for u, v in zip(path[:-1], path[1:]):
            # For MultiDiGraph, get the best parallel edge
            if G.has_edge(u, v):
                edges = G.get_edge_data(u, v)
                # choose the first edge
                data = list(edges.values())[0]
            else:
                data = {"length": 0.0, "name": None, "highway": None, "maxspeed": None}

            name = data.get('name') or 'Unnamed road'
            highway = data.get('highway')
            if isinstance(highway, list):
                highway = highway[0]
            length_m = float(data.get('length', 0.0))
            maxspeed_kph = _edge_speed_kph(data, speed_model)
            travel_time_min = (length_m/1000.0) / max(5.0, maxspeed_kph) * 60.0
            cost_inr = (length_m/1000.0) * (fuel_consumption_l_per_100km/100.0) * fuel_price_inr_per_l
            emissions_kg = (length_m/1000.0) * (emission_factor_g_co2_per_km/1000.0)
            is_flyover = (data.get('bridge') in ['yes', True]) or ('flyover' in str(name).lower())

            # node coordinates for mapping
            u_lat = float(G.nodes[u].get('y'))
            u_lon = float(G.nodes[u].get('x'))
            v_lat = float(G.nodes[v].get('y'))
            v_lon = float(G.nodes[v].get('x'))

            segments.append({
                "u": u, "v": v,
                "u_lat": u_lat, "u_lon": u_lon,
                "v_lat": v_lat, "v_lon": v_lon,
                "name": name,
                "highway": highway,
                "length_m": length_m,
                "maxspeed_kph": maxspeed_kph,
                "travel_time_min": travel_time_min,
                "cost_inr": cost_inr,
                "emissions_kg": emissions_kg,
                "is_flyover": is_flyover,
            })

            total_length_m += length_m
            total_time_min += travel_time_min
            total_cost_inr += cost_inr
            total_emissions_kg += emissions_kg
            if is_flyover:
                num_flyovers += 1
            if highway in ["motorway", "trunk"]:
                highway_len_m += length_m

        routes.append({
            "route_id": i+1,
            "path": path,
            "segments": segments,
            "total_distance_km": total_length_m/1000.0,
            "total_time_min": total_time_min,
            "total_cost_inr": total_cost_inr,
            "total_emissions_kg": total_emissions_kg,
            "num_segments": len(segments),
            "has_flyover": num_flyovers > 0,
            "highway_share_pct": (highway_len_m/total_length_m*100.0) if total_length_m>0 else 0.0,
        })

    return routes


def summarize_routes(routes: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(routes)
    # dominance rank for Pareto-optimality: 0 = nondominated
    metrics = df[["total_distance_km", "total_time_min", "total_cost_inr", "total_emissions_kg"]].values
    ranks = _pareto_ranks(metrics)
    df['dominance_rank'] = ranks
    df['optimized'] = ["candidate" for _ in range(len(df))]
    return df


def recommend_route(df_routes: pd.DataFrame, objective: str):
    idx = 0
    tag = "optimized:balanced"
    if objective == 'pareto':
        # choose the nondominated route with best balanced score
        df_nd = df_routes[df_routes['dominance_rank']==0].copy()
        if df_nd.empty:
            df_nd = df_routes.copy()
        idx_rel, tag_bal = _best_balanced_index(df_nd)
        # map back to original index
        idx = df_nd.index[idx_rel]
        tag = f"optimized:pareto-balanced (nondominated set size={len(df_nd)})"
    elif objective in ['min_distance', 'min_time', 'min_cost', 'min_emissions']:
        col_map = {
            'min_distance': 'total_distance_km',
            'min_time': 'total_time_min',
            'min_cost': 'total_cost_inr',
            'min_emissions': 'total_emissions_kg',
        }
        col = col_map[objective]
        idx = int(df_routes[col].idxmin())
        tag = f"optimized:{objective}"
    else:
        idx, tag = _best_balanced_index(df_routes)
        tag = f"optimized:balanced"

    # mark optimized row
    df_routes.loc[:, 'optimized'] = 'candidate'
    df_routes.loc[idx, 'optimized'] = 'recommended'
    return idx, tag


def _best_balanced_index(df: pd.DataFrame):
    # normalize metrics to [0,1] and choose minimal sum
    cols = ["total_distance_km", "total_time_min", "total_cost_inr", "total_emissions_kg"]
    X = df[cols].values.astype(float)
    # guard against zeros
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
    Xn = (X - mins) / denom
    scores = Xn.sum(axis=1)
    idx_rel = int(np.argmin(scores))
    tag = f"optimized:balanced(score={scores[idx_rel]:.3f})"
    return idx_rel, tag


def _pareto_ranks(values: np.ndarray) -> List[int]:
    n = values.shape[0]
    ranks = np.zeros(n, dtype=int)
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is <= in all metrics and < in at least one
            if np.all(values[j] <= values[i]) and np.any(values[j] < values[i]):
                dominated = True
                break
        ranks[i] = 0 if not dominated else 1
    return ranks.tolist()
