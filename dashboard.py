import calendar
import json
from pathlib import Path

import numpy as np
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objects as go
import pandas as pd
from sqlalchemy import text
from db_connection import DBConnection
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# Constants
YEAR_FILTER = 2025
DAY_NAMES = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
WEEKEND_DAYS = {'Saturday', 'Sunday'}
COLORS = {
    'primary': '#91C4E5',      # Soft blue
    'secondary': '#FFB7B2',    # Soft coral/pink
    'success': '#A8E6CF',      # Soft mint green
    'danger': '#FFA8A8',       # Soft red/pink
    'weekday': '#AAE3E2',      # Soft teal
    'weekend': '#C7CEEA',      # Soft purple/lavender
    'accent1': '#FFE5B4',      # Soft yellow
    'accent2': '#E0BBE4',      # Soft purple
    'accent3': '#D4A5A5'       # Soft mauve
}

# --- Helpers for Map Tiles / Zone labels -------------------------------------
ASSETS_GEOJSON = Path("assets/taxi_zones.geojson")
LOOKUP_CSV = Path("assets/taxi_zone_lookup.csv")


def _bbox_centroid(coords):
    """Centroid via bbox center (no heavy deps)."""
    if not coords:
        return None
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return ((min(lons) + max(lons)) / 2.0, (min(lats) + max(lats)) / 2.0)


def _extract_all_points(geometry):
    """Flatten Polygon/MultiPolygon coords -> list of [lon, lat]."""
    if not geometry or "type" not in geometry:
        return []
    gtype = geometry["type"]
    coords = geometry.get("coordinates", [])
    pts = []
    if gtype == "Polygon":
        for ring in coords:
            pts.extend(ring)
    elif gtype == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                pts.extend(ring)
    return pts


def _load_zone_centroids(geojson_path: Path = ASSETS_GEOJSON) -> dict[int, tuple[float, float]]:
    """{LocationID: (lon, lat)} from assets/taxi_zones.geojson; {} on failure."""
    try:
        gj = json.loads(geojson_path.read_text())
    except Exception as e:
        print(f"[Map] Could not read {geojson_path}: {e}")
        return {}
    cent = {}
    for f in gj.get("features", []):
        props = f.get("properties", {}) or {}
        loc = (props.get("LocationID") or props.get("locationid")
               or props.get("location_id") or props.get("LOCATIONID"))
        try:
            loc = int(loc)
        except Exception:
            continue
        c = _bbox_centroid(_extract_all_points(f.get("geometry", {})))
        if c:
            cent[loc] = c
    print(f"[Map] Loaded {len(cent)} zone centroids from {geojson_path}")
    return cent


def _load_zone_lookup(path: Path = LOOKUP_CSV) -> tuple[dict[int, str], dict[int, str]]:
    """
    Return (name_map, borough_map) from taxi_zone_lookup.csv.
    name_map[id] -> Zone, borough_map[id] -> Borough.
    """
    try:
        df = pd.read_csv(path)
        df = df.dropna(subset=["LocationID"])
        df["LocationID"] = df["LocationID"].astype(int)
        name_map = dict(zip(df["LocationID"], df["Zone"].astype(str)))
        borough_map = dict(zip(df["LocationID"], df["Borough"].astype(str)))
        print(f"[Map] Loaded {len(name_map)} zone labels from {path}")
        return name_map, borough_map
    except Exception as e:
        print(f"[Map] Could not read {path}: {e}")
        return {}, {}
# -----------------------------------------------------------------------------


def get_db_engine():
    if not hasattr(get_db_engine, 'engine'):
        db = DBConnection("postgres", "password123", "localhost", 5433, "ny_taxi_dwh")
        get_db_engine.engine = db.connect()
    return get_db_engine.engine


def execute_query(query: str, engine) -> pd.DataFrame:
    if engine is None:
        return pd.DataFrame()
    try:
        return pd.read_sql(text(query), engine)
    except Exception as e:
        print(f"Query error: {e}")
        return pd.DataFrame()


def load_data():
    engine = get_db_engine()

    queries = {
        'monthly': f"""
            SELECT dd.year,
                   dd.month,
                   ROUND(AVG(ft.trip_duration_min)::numeric, 2) as avg_duration,
                   COUNT(*) as trip_count
            FROM fact_trips ft
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            WHERE dd.year >= {YEAR_FILTER}
            GROUP BY dd.year, dd.month
            ORDER BY dd.year, dd.month
        """,
        'weekday': f"""
            SELECT EXTRACT(DOW FROM dd.full_datetime)::int as day_of_week,
                   ROUND(AVG(ft.trip_duration_min)::numeric, 2) as avg_duration,
                   COUNT(*) as trip_count
            FROM fact_trips ft
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            WHERE dd.year >= {YEAR_FILTER}
            GROUP BY day_of_week
            ORDER BY day_of_week
        """,
        'heatmap': f"""
            SELECT dd.month,
                   EXTRACT(DOW FROM dd.full_datetime)::int as day_of_week,
                   COUNT(*) as trip_count
            FROM fact_trips ft
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            WHERE dd.year >= {YEAR_FILTER}
            GROUP BY dd.month, day_of_week
        """,
        'fare_weekday': f"""
            SELECT EXTRACT(DOW FROM dd.full_datetime)::int as day_of_week,
                   ROUND(SUM(df.total_amount)::numeric, 2) as total_revenue,
                   ROUND(SUM(df.fare_amount)::numeric, 2) as total_fare,
                   ROUND(SUM(df.tip_amount)::numeric, 2) as total_tip
            FROM fact_trips ft
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            JOIN dim_fare df ON ft.fare_id = df.fare_id
            WHERE dd.year >= {YEAR_FILTER}
            GROUP BY day_of_week
            ORDER BY day_of_week
        """,
        'hourly': f"""
            SELECT dd.hour,
                   COUNT(*) as trip_count,
                   ROUND(AVG(ft.trip_duration_min)::numeric, 2) as avg_duration
            FROM fact_trips ft
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            WHERE dd.year >= {YEAR_FILTER}
            GROUP BY dd.hour
            ORDER BY dd.hour
        """,
        'top_pickup_locations': f"""
            SELECT ft."PULocationID" as pulocationid,
                   COUNT(*) as trip_count
            FROM fact_trips ft
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            WHERE dd.year >= {YEAR_FILTER}
            GROUP BY ft."PULocationID"
            ORDER BY trip_count DESC
            LIMIT 10
        """,
        'payment_type_distribution': f"""
            SELECT ft.payment_type,
                   CASE ft.payment_type
                       WHEN 1 THEN 'Credit Card'
                       WHEN 2 THEN 'Cash'
                       WHEN 3 THEN 'No Charge'
                       WHEN 4 THEN 'Dispute'
                       WHEN 5 THEN 'Unknown'
                       WHEN 6 THEN 'Voided Trip'
                       ELSE 'Other'
                   END as payment_name,
                   COUNT(*) as trip_count
            FROM fact_trips ft
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            WHERE dd.year >= {YEAR_FILTER}
            GROUP BY ft.payment_type
            ORDER BY trip_count DESC
        """,
        'weather_monthly': f"""
            SELECT DATE_TRUNC('month', dd.full_datetime) as year_month,
                   ROUND(AVG(dw.temp)::numeric, 2) as avg_temp,
                   ROUND(AVG(dw.precip)::numeric, 2) as avg_precip,
                   COUNT(ft.datetime_id) as trip_count
            FROM dim_weather dw
            LEFT JOIN fact_trips ft ON ft.datetime_id = dw.datetime_id
            JOIN dim_datetime dd ON dw.datetime_id = dd.datetime_id
            WHERE dd.year >= {YEAR_FILTER}
            GROUP BY year_month
            ORDER BY year_month
        """,
        'weather_conditions': f"""
            SELECT dw.conditions,
                   COUNT(ft.datetime_id) as trip_count
            FROM dim_weather dw
            LEFT JOIN fact_trips ft ON ft.datetime_id = dw.datetime_id
            JOIN dim_datetime dd ON dw.datetime_id = dd.datetime_id
            WHERE dd.year >= {YEAR_FILTER}
            GROUP BY dw.conditions
            ORDER BY trip_count DESC
        """,
        'weather_scatter': f"""
            SELECT dw.temp,
                   COUNT(ft.datetime_id) as trip_count
            FROM dim_weather dw
            LEFT JOIN fact_trips ft ON ft.datetime_id = dw.datetime_id
            JOIN dim_datetime dd ON dw.datetime_id = dd.datetime_id
            WHERE dd.year >= {YEAR_FILTER}
            GROUP BY dw.temp
            ORDER BY dw.temp
        """,
        'correlation': f"""
            SELECT
                ft.trip_duration_min,
                ft.trip_distance,
                df.fare_amount,
                df.tip_amount,
                df.total_amount,
                ft.passenger_count,
                dw.temp,
                dw.precip,
                dd.hour
            FROM fact_trips ft
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            JOIN dim_fare df ON ft.fare_id = df.fare_id
            LEFT JOIN dim_weather dw ON ft.datetime_id = dw.datetime_id
            WHERE dd.year >= {YEAR_FILTER}
                AND ft.trip_duration_min IS NOT NULL
                AND ft.trip_distance IS NOT NULL
                AND df.fare_amount IS NOT NULL
                AND df.tip_amount IS NOT NULL
            LIMIT 50000
        """,
        'clustering': f"""
            SELECT
                ft.trip_duration_min,
                ft.trip_distance,
                dd.hour,
                EXTRACT(DOW FROM dd.full_datetime)::int as day_of_week,
                df.fare_amount,
                df.total_amount,
                df.tip_amount
            FROM fact_trips ft
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            JOIN dim_fare df ON ft.fare_id = df.fare_id
            WHERE dd.year >= {YEAR_FILTER}
                AND ft.trip_duration_min > 0
                AND ft.trip_duration_min < 180
                AND ft.trip_distance > 0
                AND ft.trip_distance < 100
                AND df.fare_amount > 0
        """,
        # New: counts for maps (all zones)
        'pickup_counts_all': f"""
            SELECT ft."PULocationID" AS location_id, COUNT(*) AS trip_count
            FROM fact_trips ft
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            WHERE dd.year >= {YEAR_FILTER}
            GROUP BY ft."PULocationID"
        """,
        'dropoff_counts_all': f"""
            SELECT ft."DOLocationID" AS location_id, COUNT(*) AS trip_count
            FROM fact_trips ft
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            WHERE dd.year >= {YEAR_FILTER}
            GROUP BY ft."DOLocationID"
        """,
        'pickup_by_hour_location': f"""
            SELECT dd.hour,
                   ft."PULocationID" AS location_id,
                   COUNT(*) AS trip_count
            FROM fact_trips ft
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            WHERE dd.year >= {YEAR_FILTER}
            GROUP BY dd.hour, ft."PULocationID"
        """,
        'dropoff_by_hour_location': f"""
            SELECT dd.hour,
                   ft."DOLocationID" AS location_id,
                   COUNT(*) AS trip_count
            FROM fact_trips ft
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            WHERE dd.year >= {YEAR_FILTER}
            GROUP BY dd.hour, ft."DOLocationID"
        """,
    }

    return tuple(execute_query(q, engine) for q in queries.values())


def create_chart_layout(title: str, xaxis: str, yaxis: str, height: int = 450):
    return dict(
        title=title,
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        hovermode='x unified',
        template='plotly_white',
        height=height,
        margin=dict(l=50, r=50, t=50, b=50)
    )


def safe_agg(series, func, default=0.0):
    return func(series) if isinstance(series, pd.Series) and not series.empty and not series.isna().all() else default


def create_app():
    app = dash.Dash(__name__)

    # Load all data
    (df_monthly, df_weekday, df_heatmap, df_fare_weekday,
     df_hourly, df_top_pickup, df_payment,
     df_weather_monthly, df_weather_conditions, df_weather_scatter,
     df_correlation, df_clustering, df_pickup_all, df_dropoff_all, df_pickup_hourly, df_dropoff_hourly) = load_data()

    # Load zone label maps once (for charts & hovers)
    zone_name_map, zone_borough_map = _load_zone_lookup()

    # Prepare monthly data
    if not df_monthly.empty:
        df_monthly['year_month'] = pd.to_datetime(
            df_monthly[['year', 'month']].assign(day=1)
        )

    if not df_weekday.empty:
        df_weekday = df_weekday.sort_values('day_of_week')
        df_weekday['day_name'] = df_weekday['day_of_week'].astype(int).map(lambda x: DAY_NAMES[x])

    if not df_fare_weekday.empty:
        df_fare_weekday = df_fare_weekday.sort_values('day_of_week')
        df_fare_weekday['day_name'] = df_fare_weekday['day_of_week'].astype(int).map(lambda x: DAY_NAMES[x])

    # Replace IDs with names for the Top 10 pickup chart (fallback to manual dict or ID)
    manual_zone_names = {
        132: 'JFK Airport',
        138: 'LaGuardia Airport',
        161: 'Midtown Center',
        162: 'Midtown East',
        163: 'Midtown North',
        164: 'Midtown South',
        186: 'Penn Station/Madison Sq West',
        230: 'Times Sq/Theatre District',
        236: 'Upper East Side North',
        237: 'Upper East Side South',
        142: 'Lincoln Square East',
        239: 'Upper West Side North',
        140: 'Lenox Hill West',
        141: 'Lenox Hill East',
        79: 'East Harlem North',
        170: 'Murray Hill',
        234: 'Union Sq',
        48: 'Clinton East',
        68: 'East Chelsea',
        90: 'Financial District North'
    }

    def _label_for(lid: int) -> str:
        name = zone_name_map.get(lid)
        if name:
            boro = zone_borough_map.get(lid)
            return f"{name} ({boro})" if boro else name
        # fallback to your manual dict, then to "Zone {id}"
        return manual_zone_names.get(lid, f"Zone {lid}")

    # ---------- Clustering ----------
    df_clustered = pd.DataFrame()
    cluster_stats = pd.DataFrame()
    cluster_hourly = pd.DataFrame()

    if not df_clustering.empty:
        print(f"Clustering: Loaded {len(df_clustering):,} records")

        features = ['trip_duration_min', 'fare_amount', 'trip_distance']
        X = df_clustering[features].fillna(0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        df_clustering['trip_cluster'] = kmeans.fit_predict(X_scaled)

        cluster_means = df_clustering.groupby('trip_cluster')[features].mean()
        cluster_order = cluster_means.mean(axis=1).sort_values()
        cluster_names = {
            cluster_order.index[0]: 'Economy / Short Trips',
            cluster_order.index[1]: 'Premium / Long Trips'
        }
        df_clustering['trip_type'] = df_clustering['trip_cluster'].map(cluster_names)

        cluster_stats = df_clustering.groupby('trip_type').agg({
            'trip_distance': 'mean',
            'trip_duration_min': 'mean',
            'fare_amount': 'mean',
            'trip_cluster': 'count'
        }).round(2)
        cluster_stats.rename(columns={'trip_cluster': 'count'}, inplace=True)

        cluster_hourly = df_clustering.groupby(['hour', 'trip_type']).size().reset_index(name='count')

        df_clustered = df_clustering
        print(f"Clustering complete: {len(df_clustered):,} records - "
              f"Economy: {len(df_clustered[df_clustered['trip_type']=='Economy / Short Trips']):,} | "
              f"Premium: {len(df_clustered[df_clustered['trip_type']=='Premium / Long Trips']):,}")

    kpis = {
        'avg_duration': safe_agg(df_monthly.get('avg_duration', pd.Series(dtype=float)), lambda s: s.mean()),
        'total_trips': int(df_monthly['trip_count'].sum()) if not df_monthly.empty else 0
    }

    app.layout = html.Div([
        html.H1("NYC Yellow Taxi Dashboard", style={'textAlign': 'center', 'marginBottom': 30, 'color': '#FFD700'}),

        html.Div([
            *[html.Div([
                html.H3(title, style={'color': '#666'}),
                html.H2(value, style={'color': color}),
            ], style={'flex': '1', 'textAlign': 'center', 'padding': '20px',
                      'backgroundColor': '#f0f0f0', 'borderRadius': '10px', 'margin': '10px'})
            for title, value, color in [
                ("Avg Trip Duration", f"{kpis['avg_duration']:.1f} Min", COLORS['primary']),
                ("Total Trips", f"{kpis['total_trips']:,}", COLORS['danger']),
            ]]
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px', 'flexWrap': 'wrap'}),

        # Maps row
        html.Div([
            html.Div([dcc.Graph(id='pickup-map')], style={'flex': '1'}),
            html.Div([dcc.Graph(id='dropoff-map')], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),
        # Store for selected hours
        dcc.Store(id='selected-hours', data=[]),

        # Hourly histogram for selection
        html.Div([
            dcc.Graph(id='hourly-pickup-histogram')
        ], style={'marginBottom': '40px'}),

        # Time-based Analysis Row
        html.Div([
            html.Div([dcc.Graph(id='monthly-trips-chart')], style={'flex': '1'}),
            html.Div([dcc.Graph(id='weekday-duration-chart')], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        # Heatmap and Revenue Row
        html.Div([
            html.Div([dcc.Graph(id='heatmap-chart')], style={'flex': '1'}),
            html.Div([dcc.Graph(id='fare-weekday-chart')], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        # Top Locations and Payment Distribution
        html.Div([
            html.Div([dcc.Graph(id='top-pickup-locations-chart')], style={'flex': '1'}),
            html.Div([dcc.Graph(id='vendor-distribution-chart')], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        html.Div([
            html.Div([dcc.Graph(id='weather-condition-chart')], style={'flex': '1'}),
            html.Div([dcc.Graph(id='weather-scatter-chart')], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        html.Div([
            html.Div([dcc.Graph(id='correlation-matrix-chart')], style={'width': '100%'}),
        ], style={'marginBottom': '20px'}),

        html.H2("Trip Type Clustering Analysis", style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 20, 'color': '#333'}),

        html.Div([
            html.Div([dcc.Graph(id='cluster-distribution-chart')], style={'flex': '1'}),
            html.Div([dcc.Graph(id='cluster-characteristics-chart')], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        html.Div([
            html.Div([dcc.Graph(id='cluster-time-patterns-chart')], style={'width': '100%'}),
        ], style={'marginBottom': '20px'}),

        html.Div([
            html.Div([dcc.Graph(id='cluster-3d-scatter')], style={'width': '100%'}),
        ], style={'marginBottom': '20px'}),

    ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#fafafa'})

    # ------------------- Callbacks -------------------

    @callback(Output('weekday-duration-chart', 'figure'), Input('weekday-duration-chart', 'id'))
    def update_weekday_chart(_):
        fig = go.Figure()
        if not df_weekday.empty:
            colors = [COLORS['weekend'] if day in WEEKEND_DAYS else COLORS['weekday'] for day in df_weekday['day_name']]
            fig.add_trace(go.Bar(
                x=df_weekday['day_name'],
                y=df_weekday['avg_duration'],
                name='Avg Duration',
                marker_color=colors,
                hovertemplate='<b>%{x}</b><br>Avg Duration: %{y:.1f} Min<extra></extra>'
            ))
        fig.update_layout(**create_chart_layout('Average Trip Duration by Day of Week', 'Day of Week', 'Trip Duration (Minutes)'))
        return fig

    @callback(Output('heatmap-chart', 'figure'), Input('heatmap-chart', 'id'))
    def update_heatmap_chart(_):
        fig = go.Figure()
        if not df_heatmap.empty:
            pivot_data = df_heatmap.pivot(index='day_of_week', columns='month', values='trip_count')
            pivot_data = pivot_data.reindex(range(7)).fillna(0)
            pivot_data.index = DAY_NAMES
            pivot_data.columns = [calendar.month_name[i] for i in pivot_data.columns if i <= 12]
            fig.add_trace(go.Heatmap(
                z=pivot_data.values,
                x=list(pivot_data.columns),
                y=list(pivot_data.index),
                colorscale='Viridis',
                hovertemplate='<b>%{y}</b><br><b>%{x}</b><br>Trips: %{z:,.0f}<extra></extra>'
            ))
        fig.update_layout(**create_chart_layout('Trip Heatmap: Day of Week vs Month', 'Month', 'Day of Week', 500))
        return fig

    @callback(Output('fare-weekday-chart', 'figure'), Input('fare-weekday-chart', 'id'))
    def update_fare_weekday_chart(_):
        fig = go.Figure()
        if not df_fare_weekday.empty:
            fig.add_trace(go.Bar(
                x=df_fare_weekday['day_name'],
                y=df_fare_weekday['total_revenue'],
                name='Total Revenue',
                marker_color=COLORS['success'],
                hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.2f}<extra></extra>'
            ))
        fig.update_layout(**create_chart_layout('Total Revenue by Day of Week', 'Day of Week', 'Revenue ($)'))
        return fig

    @callback(Output('monthly-trips-chart', 'figure'), Input('monthly-trips-chart', 'id'))
    def update_monthly_trips_chart(_):
        fig = go.Figure()
        if not df_monthly.empty:
            fig.add_trace(go.Bar(
                x=df_monthly['year_month'],
                y=df_monthly['trip_count'],
                name='Trips',
                marker_color=COLORS['danger'],
                hovertemplate='<b>%{x|%B %Y}</b><br>Trips: %{y:,.0f}<extra></extra>'
            ))
        fig.update_layout(**create_chart_layout('Trips by Month', 'Month', 'Number of Trips'))
        return fig

    @callback(Output('top-pickup-locations-chart', 'figure'), Input('top-pickup-locations-chart', 'id'))
    def update_top_pickup_chart(_):
        fig = go.Figure()
        if not df_top_pickup.empty:
            df_sorted = df_top_pickup.copy()
            df_sorted['zone_name'] = df_sorted['pulocationid'].astype(int).map(_label_for)
            df_sorted = df_sorted.sort_values('trip_count', ascending=True)

            fig.add_trace(go.Bar(
                x=df_sorted['trip_count'],
                y=df_sorted['zone_name'],
                orientation='h',
                marker_color=COLORS['primary'],
                hovertemplate='<b>%{y}</b><br>Trips: %{x:,}<extra></extra>'
            ))
        layout = create_chart_layout('Top 10 Pickup Locations', 'Number of Trips', 'Zone')
        layout['height'] = 550
        fig.update_layout(**layout)
        return fig

    @callback(Output('vendor-distribution-chart', 'figure'), Input('vendor-distribution-chart', 'id'))
    def update_payment_chart(_):
        fig = go.Figure()
        if not df_payment.empty:
            # Use pastel color palette
            pastel_colors = [
                COLORS['primary'],
                COLORS['secondary'],
                COLORS['success'],
                COLORS['weekday'],
                COLORS['accent1'],
                COLORS['accent2'],
                COLORS['accent3']
            ]
            fig.add_trace(go.Pie(
                labels=df_payment['payment_name'],
                values=df_payment['trip_count'],
                hole=0.3,
                marker=dict(
                    colors=pastel_colors,
                    line=dict(color='white', width=2)
                ),
                textposition='auto',
                textinfo='label+percent',
                textfont=dict(size=13, color='#333'),
                pull=[0.05 if i == df_payment['trip_count'].idxmax() else 0 for i in range(len(df_payment))],
                hovertemplate='<b>%{label}</b><br>Trips: %{value:,}<br>Percentage: %{percent}<extra></extra>'
            ))
        fig.update_layout(
            title=dict(
                text='Payment Type Distribution',
                font=dict(size=18, color='#333'),
                x=0.5,
                xanchor='center'
            ),
            height=550,
            showlegend=True,
            legend=dict(
                orientation='v',
                yanchor='middle',
                y=0.5,
                xanchor='left',
                x=1.02,
                font=dict(size=12)
            ),
            template='plotly_white',
            margin=dict(l=20, r=150, t=80, b=20)
        )
        return fig

    # ------------------- Maps -------------------

    def _build_map_figure(df_counts: pd.DataFrame, title: str) -> go.Figure:
        fig = go.Figure()
        centroids = _load_zone_centroids()
        if df_counts.empty or not centroids:
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox_center={"lon": -73.9851, "lat": 40.7589},
                mapbox_zoom=9, height=550, title=title,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            fig.add_annotation(text="No data or taxi_zones.geojson missing.",
                               showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
            return fig

        merged = df_counts.copy()
        merged["location_id"] = merged["location_id"].astype(int)
        merged["lon"] = merged["location_id"].map(lambda x: centroids.get(x, (None, None))[0])
        merged["lat"] = merged["location_id"].map(lambda x: centroids.get(x, (None, None))[1])
        before = len(merged)
        merged = merged.dropna(subset=["lon", "lat"])
        dropped = before - len(merged)
        print(f"[Map] {title}: rows={before}, kept={len(merged)}, dropped(no centroid)={dropped}")

        def _label(loc_id: int) -> str:
            return _label_for(loc_id)

        max_count = merged["trip_count"].max()
        size = 6 + 20 * np.sqrt(merged["trip_count"] / max_count)

        fig.add_trace(go.Scattermapbox(
            lon=merged["lon"],
            lat=merged["lat"],
            mode="markers",
            marker=dict(
                size=size,
                color=merged["trip_count"],
                colorscale=[[0, "rgb(12,51,131)"], [0.25, "rgb(10,136,186)"], [0.5, "rgb(242,211,56)"], [0.75, "rgb(242,143,56)"], [1, "rgb(217,30,30)"]],
                opacity=0.8,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Trips", side="right"),
                    thickness=12,
                    len=0.5,
                    x=0.98,
                    y=0.5,
                    xanchor="right",
                    yanchor="middle",
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#333",
                    borderwidth=1,
                    tickfont=dict(color="#333", size=10),
                    titlefont=dict(size=11)
                )
            ),
            text=[f"{_label(lid)}<br>Trips: {cnt:,}" for lid, cnt in zip(merged["location_id"], merged["trip_count"])],
            hoverinfo="text"
        ))
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_center={"lon": -73.9851, "lat": 40.7589},
            mapbox_zoom=9, height=550, title=title,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="#f5f5f5",
            font=dict(color="#333")
        )
        return fig

    @callback(
        Output('dropoff-map', 'figure'),
        Input('selected-hours', 'data')
    )
    def update_dropoff_map_filtered(selected_hours):
        """Update dropoff map based on selected hours."""
        # If no hours selected, show all data
        if not selected_hours:
            return _build_map_figure(df_dropoff_all, "Dropoff Locations (All Hours)")

        # Filter data by selected hours
        df_filtered = df_dropoff_hourly[df_dropoff_hourly['hour'].isin(selected_hours)]
        df_aggregated = df_filtered.groupby('location_id', as_index=False)['trip_count'].sum()

        hours_str = ', '.join([f"{h}:00" for h in sorted(selected_hours)])
        title = f"Dropoff Locations (Hours: {hours_str})"

        return _build_map_figure(df_aggregated, title)

    # ------------------- Weather / Correlation -------------------

    @callback(Output('weather-condition-chart', 'figure'), Input('weather-condition-chart', 'id'))
    def update_weather_condition_chart(_):
        fig = go.Figure()
        if not df_weather_conditions.empty:
            fig.add_trace(go.Bar(
                x=df_weather_conditions['conditions'],
                y=df_weather_conditions['trip_count'],
                marker_color=COLORS['primary'],
                hovertemplate='Condition: %{x}<br>Trips: %{y:,}<extra></extra>'
            ))
        fig.update_layout(**create_chart_layout('Trips by Weather Conditions', 'Weather Condition', 'Number of Trips'))
        return fig

    @callback(Output('weather-scatter-chart', 'figure'), Input('weather-scatter-chart', 'id'))
    def update_weather_scatter_chart(_):
        fig = go.Figure()
        if not df_weather_scatter.empty:
            fig.add_trace(go.Scatter(
                x=df_weather_scatter['temp'],
                y=df_weather_scatter['trip_count'],
                mode='markers',
                marker=dict(size=10, color=df_weather_scatter['trip_count'], colorscale='Viridis', showscale=True),
                hovertemplate='Temp: %{x}°C<br>Trips: %{y:,}<extra></extra>'
            ))
        fig.update_layout(**create_chart_layout('Trips vs Temperature', 'Temperature (°C)', 'Number of Trips', 500))
        return fig

    @callback(Output('correlation-matrix-chart', 'figure'), Input('correlation-matrix-chart', 'id'))
    def update_correlation_matrix(_):
        fig = go.Figure()
        if not df_correlation.empty:
            display_names = {
                'trip_duration_min': 'Trip Duration',
                'trip_distance': 'Trip Distance',
                'fare_amount': 'Fare Amount',
                'tip_amount': 'Tip Amount',
                'total_amount': 'Total Amount',
                'passenger_count': 'Passenger Count',
                'temp': 'Temperature',
                'precip': 'Precipitation',
                'hour': 'Hour of Day'
            }

            corr_matrix = df_correlation.corr()

            corr_matrix.columns = [display_names.get(col, col) for col in corr_matrix.columns]
            corr_matrix.index = [display_names.get(idx, idx) for idx in corr_matrix.index]

            fig.add_trace(go.Heatmap(
                z=corr_matrix.values,
                x=list(corr_matrix.columns),
                y=list(corr_matrix.index),
                colorscale='RdBu',
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation"),
                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
            ))

        fig.update_layout(
            title='Correlation Matrix: Trip & Weather Metrics',
            xaxis_title='',
            yaxis_title='',
            height=600,
            template='plotly_white',
            margin=dict(l=150, r=50, t=80, b=150),
            xaxis={'side': 'bottom'},
            yaxis={'autorange': 'reversed'}
        )
        return fig

    # ------------------- Clustering charts -------------------

    @callback(Output('cluster-distribution-chart', 'figure'), Input('cluster-distribution-chart', 'id'))
    def update_cluster_distribution(_):
        fig = go.Figure()
        total_trips = 0
        if not df_clustered.empty:
            cluster_counts = df_clustered['trip_type'].value_counts()
            total_trips = len(df_clustered)
            fig.add_trace(go.Pie(
                labels=cluster_counts.index,
                values=cluster_counts.values,
                hole=0.4,
                marker=dict(colors=[COLORS['weekday'], COLORS['secondary']]),
                hovertemplate='%{label}<br>Count: %{value:,}<br>%{percent}<extra></extra>'
            ))
        fig.update_layout(
            title=f'Trip Type Distribution (2 Clusters) - Total: {total_trips:,} trips',
            height=450,
            template='plotly_white',
            margin=dict(l=20, r=20, t=60, b=20)
        )
        return fig

    @callback(Output('cluster-characteristics-chart', 'figure'), Input('cluster-characteristics-chart', 'id'))
    def update_cluster_characteristics(_):
        fig = go.Figure()
        if not cluster_stats.empty:
            cluster_info = [f"{cluster}<br>({int(cluster_stats.loc[cluster, 'count']):,} trips)"
                            for cluster in cluster_stats.index]

            fig.add_trace(go.Bar(
                name='Avg Distance (mi)',
                x=cluster_info,
                y=cluster_stats['trip_distance'],
                marker_color=COLORS['weekday'],
                hovertemplate='Distance: %{y:.2f} mi<extra></extra>'
            ))
            fig.add_trace(go.Bar(
                name='Avg Duration (min)',
                x=cluster_info,
                y=cluster_stats['trip_duration_min'],
                marker_color=COLORS['accent1'],
                hovertemplate='Duration: %{y:.2f} min<extra></extra>'
            ))
            fig.add_trace(go.Bar(
                name='Avg Fare ($)',
                x=cluster_info,
                y=cluster_stats['fare_amount'],
                marker_color=COLORS['secondary'],
                hovertemplate='Fare: $%{y:.2f}<extra></extra>'
            ))
        fig.update_layout(
            title='Cluster Characteristics (Based on All Data)',
            xaxis_title='',
            yaxis_title='Average Value',
            barmode='group',
            height=450,
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=20, t=80, b=100)
        )
        return fig

    @callback(Output('cluster-time-patterns-chart', 'figure'), Input('cluster-time-patterns-chart', 'id'))
    def update_cluster_time_patterns(_):
        fig = go.Figure()
        if not cluster_hourly.empty:
            colors = {
                'Economy / Short Trips': COLORS['weekday'],
                'Premium / Long Trips': COLORS['secondary']
            }

            for trip_type in cluster_hourly['trip_type'].unique():
                data = cluster_hourly[cluster_hourly['trip_type'] == trip_type]
                fig.add_trace(go.Scatter(
                    x=data['hour'],
                    y=data['count'],
                    mode='lines+markers',
                    name=trip_type,
                    line=dict(color=colors.get(trip_type, COLORS['primary']), width=3),
                    marker=dict(size=8),
                    hovertemplate='%{fullData.name}<br>Hour: %{x}:00<br>Trips: %{y:,}<extra></extra>'
                ))
        total_data_points = len(df_clustered) if not df_clustered.empty else 0
        fig.update_layout(
            title=f'Trip Patterns by Hour (2 Clusters) - All {total_data_points:,} Data Points',
            xaxis_title='Hour of Day',
            yaxis_title='Number of Trips',
            height=450,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=60, r=20, t=80, b=60)
        )
        fig.update_xaxes(tickmode='linear', dtick=2)
        return fig

    @callback(Output('cluster-3d-scatter', 'figure'), Input('cluster-3d-scatter', 'id'))
    def update_3d_scatter(_):
        fig = go.Figure()
        df_sample = pd.DataFrame()

        if not df_clustered.empty:
            max_total = 50000
            df_sample_list = []

            for trip_type in df_clustered['trip_type'].unique():
                cluster_data = df_clustered[df_clustered['trip_type'] == trip_type]
                cluster_size = len(cluster_data)
                proportion = cluster_size / len(df_clustered)
                samples_for_cluster = max(int(max_total * proportion), 100)

                sampled = cluster_data.sample(min(samples_for_cluster, cluster_size), random_state=42)
                df_sample_list.append(sampled)

            df_sample = pd.concat(df_sample_list, ignore_index=True)

            colors_map = {
                'Economy / Short Trips': COLORS['weekday'],
                'Premium / Long Trips': COLORS['secondary']
            }

            for trip_type in df_sample['trip_type'].unique():
                data = df_sample[df_sample['trip_type'] == trip_type]

                fig.add_trace(go.Scatter3d(
                    x=data['trip_duration_min'],
                    y=data['fare_amount'],
                    z=data['trip_distance'],
                    mode='markers',
                    name=trip_type,
                    marker=dict(
                        size=2,
                        color=colors_map.get(trip_type, COLORS['primary']),
                        opacity=0.5,
                        line=dict(width=0)
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  'Duration: %{x:.1f} min<br>' +
                                  'Fare: $%{y:.2f}<br>' +
                                  'Distance: %{z:.2f} mi<extra></extra>'
                ))

        fig.update_layout(
            title=f'3D Cluster Visualization: Duration × Price × Distance (Sample: {len(df_sample):,} / {len(df_clustered):,} points)',
            scene=dict(
                xaxis=dict(title='Trip Duration (min)', backgroundcolor='white', gridcolor='lightgray'),
                yaxis=dict(title='Fare Amount ($)', backgroundcolor='white', gridcolor='lightgray'),
                zaxis=dict(title='Trip Distance (mi)', backgroundcolor='white', gridcolor='lightgray'),
                bgcolor='white'
            ),
            height=700,
            template='plotly_white',
            showlegend=True,
            legend=dict(
                x=0.7,
                y=0.9,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        return fig

    # ------------------- Interactive Hourly Pickup Analysis -------------------

    @callback(
        Output('hourly-pickup-histogram', 'figure'),
        Input('selected-hours', 'data')
    )
    def update_hourly_histogram(selected_hours):
        """Create histogram showing trips by hour with visual feedback for selected hours."""
        fig = go.Figure()
        if df_hourly.empty:
            return fig

        # Create bar colors based on selection
        colors = [COLORS['secondary'] if hour in selected_hours else COLORS['weekday'] for hour in df_hourly['hour']]

        fig.add_trace(go.Bar(
            x=df_hourly['hour'],
            y=df_hourly['trip_count'],
            marker_color=colors,
            width=0.5,
            hovertemplate='<b>Hour %{x}:00</b><br>Trips: %{y:,.0f}<br>Click to filter map<extra></extra>'
        ))

        title_text = 'Select Hour to Filter Pickup Map'
        if selected_hours:
            hours_str = ', '.join([f"{h}:00" for h in sorted(selected_hours)])
            title_text = f'Pickup Trips by Hour - Filtering: {hours_str}'

        fig.update_layout(
            title=title_text,
            xaxis_title='Hour of Day',
            yaxis_title='Number of Trips',
            hovermode='closest',
            template='plotly_white',
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(tickmode='linear', dtick=2),
            bargap=0.3
        )
        return fig

    @callback(
        Output('selected-hours', 'data'),
        Input('hourly-pickup-histogram', 'clickData'),
        State('selected-hours', 'data'),
        prevent_initial_call=True
    )
    def update_selected_hours(clickData, current_hours):
        """Toggle hour selection when clicking histogram bars."""
        if not clickData:
            return current_hours or []

        # Get clicked hour
        clicked_hour = clickData['points'][0]['x']

        # Toggle selection
        if clicked_hour in current_hours:
            current_hours.remove(clicked_hour)
        else:
            current_hours.append(clicked_hour)

        return current_hours

    @callback(
        Output('pickup-map', 'figure'),
        Input('selected-hours', 'data')
    )
    def update_pickup_map_filtered(selected_hours):
        """Update pickup map based on selected hours."""
        # If no hours selected, show all data
        if not selected_hours:
            return _build_map_figure(df_pickup_all, "Pickup Locations (All Hours)")

        # Filter data by selected hours
        df_filtered = df_pickup_hourly[df_pickup_hourly['hour'].isin(selected_hours)]
        df_aggregated = df_filtered.groupby('location_id', as_index=False)['trip_count'].sum()

        hours_str = ', '.join([f"{h}:00" for h in sorted(selected_hours)])
        title = f"Pickup Locations (Hours: {hours_str})"

        return _build_map_figure(df_aggregated, title)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8051, debug=False)