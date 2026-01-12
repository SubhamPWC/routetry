
import folium
from folium.features import DivIcon

def make_map(routes, recommended_idx, origin, dest):
    # center map mid-point between origin & dest
    center_lat = (origin[0] + dest[0]) / 2.0
    center_lon = (origin[1] + dest[1]) / 2.0
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=7, control_scale=True, tiles='OpenStreetMap')

    # markers
    folium.Marker(location=[origin[0], origin[1]], popup="Origin", tooltip="Origin", icon=folium.Icon(color='green')).add_to(fmap)
    folium.Marker(location=[dest[0], dest[1]], popup="Destination", tooltip="Destination", icon=folium.Icon(color='red')).add_to(fmap)

    # draw each route as a polyline over its segments
    for i, route in enumerate(routes):
        color = 'red' if i == recommended_idx else 'blue'
        weight = 6 if i == recommended_idx else 3
        coords = []
        for seg in route.get('segments', []):
            coords.append([seg['u_lat'], seg['u_lon']])
            coords.append([seg['v_lat'], seg['v_lon']])
        # deduplicate consecutive duplicates
        dedup = []
        for c in coords:
            if len(dedup) == 0 or (dedup[-1][0] != c[0] or dedup[-1][1] != c[1]):
                dedup.append(c)
        if len(dedup) >= 2:
            folium.PolyLine(locations=dedup, color=color, weight=weight, opacity=0.8,
                            tooltip=f"Route {route['route_id']}: {route['total_distance_km']:.1f} km, {route['total_time_min']:.1f} min").add_to(fmap)

        # Annotate route id near the center
        mid_lat = (origin[0] + dest[0]) / 2.0
        mid_lon = (origin[1] + dest[1]) / 2.0
        folium.map.Marker(
            [mid_lat, mid_lon],
            icon=DivIcon(icon_size=(150,36), icon_anchor=(0,0), html=f'<div style="font-size: 12px; color:{color}; font-weight:bold;">Route {route['route_id']}</div>')
        ).add_to(fmap)

    return fmap
