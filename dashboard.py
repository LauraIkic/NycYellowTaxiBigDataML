import calendar
import json
from pathlib import Path

import numpy as np
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
from sqlalchemy import text
from db_connection import DBConnection
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Constants
YEAR_FILTER = 2025
DAY_NAMES = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
WEEKEND_DAYS = {'Saturday', 'Sunday'}
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'weekday': '#4ECDC4',
    'weekend': '#FF6B6B'
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
            SELECT 
                ft.trip_distance,
                ft.trip_duration_min
            FROM fact_trips ft
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            WHERE dd.year >= {YEAR_FILTER}
                AND ft.trip_duration_min IS NOT NULL
                AND ft.trip_duration_min > 0
                AND ft.trip_duration_min < 120
                AND ft.trip_distance IS NOT NULL
                AND ft.trip_distance > 0
                AND ft.trip_distance < 50
            LIMIT 100000
        """,
        'temp_trips_scatter': f"""
            SELECT 
                dw.temp,
                COUNT(*) as trip_count
            FROM fact_trips ft
            JOIN dim_weather dw ON ft.datetime_id = dw.datetime_id
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            WHERE dd.year >= {YEAR_FILTER}
                AND dw.temp IS NOT NULL
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
            INNER JOIN dim_weather dw ON ft.datetime_id = dw.datetime_id
            WHERE dd.year >= {YEAR_FILTER}
                AND ft.trip_duration_min IS NOT NULL
                AND ft.trip_distance IS NOT NULL
                AND df.fare_amount IS NOT NULL
                AND df.tip_amount IS NOT NULL
                AND dw.temp IS NOT NULL
                AND dw.precip IS NOT NULL
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
                LIMIT 10000
        """,
        'cluster_data': f"""
              SELECT
                  df.tip_amount,
                  df.fare_amount,
                  ft.trip_distance
              FROM fact_trips ft
              JOIN dim_fare df ON ft.fare_id = df.fare_id
              JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
              WHERE dd.year >= {YEAR_FILTER}
                AND df.tip_amount IS NOT NULL
                AND df.fare_amount IS NOT NULL
                AND ft.trip_distance IS NOT NULL
              LIMIT 10000
          """,
        'regression_data': f"""
              SELECT
                  ft.trip_duration_min,
                  ft.trip_distance,
                  ft.passenger_count,
                  df.fare_amount,
                  df.tip_amount,
                  df.total_amount,
                  dw.temp,
                  dw.precip,
                  dw.humidity,
                  dw.windspeed,
                  dw.visibility,
                  dw.conditions,
                  dd.hour,
                  EXTRACT(DOW FROM dd.full_datetime)::int as day_of_week
              FROM fact_trips ft
              JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
              JOIN dim_fare df ON ft.fare_id = df.fare_id
              LEFT JOIN dim_weather dw ON ft.datetime_id = dw.datetime_id
              WHERE dd.year >= {YEAR_FILTER}
                AND ft.trip_duration_min IS NOT NULL
                AND ft.trip_distance IS NOT NULL
                AND ft.trip_duration_min > 0
                AND ft.trip_duration_min < 180
                AND ft.trip_distance > 0
                AND dw.temp IS NOT NULL
              LIMIT 100000
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
        'zone_regression': f"""
            SELECT 
                ft.trip_distance,
                df.fare_amount
            FROM fact_trips ft
            JOIN dim_fare df ON ft.fare_id = df.fare_id
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            WHERE dd.year >= {YEAR_FILTER}
                AND ft.trip_distance IS NOT NULL
                AND ft.trip_distance > 0
                AND ft.trip_distance < 50
                AND df.fare_amount IS NOT NULL
                AND df.fare_amount > 0
                AND df.fare_amount < 200
            LIMIT 100000
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
    return func(series) if not series.empty and not series.isna().all() else default


def create_app():
    app = dash.Dash(__name__)

    # Load all data
    (df_monthly, df_weekday, df_heatmap, df_fare_weekday,
     df_hourly, df_top_pickup, df_payment,
     df_weather_monthly, df_weather_conditions, df_weather_scatter, df_temp_trips_scatter, df_correlation, df_clustering, df_cluster,
     df_regression, df_pickup_all, df_dropoff_all, df_zone_regression) = load_data()

    # Debug output
    print(f"[LOAD_DATA] df_weather_scatter shape: {df_weather_scatter.shape}")
    if not df_weather_scatter.empty:
        print(f"[LOAD_DATA] df_weather_scatter columns: {df_weather_scatter.columns.tolist()}")
        print(f"[LOAD_DATA] df_weather_scatter first rows:\n{df_weather_scatter.head()}")
    else:
        print("[LOAD_DATA] WARNING: df_weather_scatter is EMPTY!")

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

        kmeans = KMeans(n_clusters=2, n_init=10)
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
        print(
            f"Clustering complete: {len(df_clustered):,} records - Economy: {len(df_clustered[df_clustered['trip_type'] == 'Economy / Short Trips']):,} | Premium: {len(df_clustered[df_clustered['trip_type'] == 'Premium / Long Trips']):,}")

    # ---------- Regression Analysis ----------
    regression_results = {}
    df_reg_pred = pd.DataFrame()
    feature_importance = pd.DataFrame()

    if not df_regression.empty:
        print(f"Regression: Loaded {len(df_regression):,} records")

        # Prepare features
        df_reg = df_regression.copy()

        # Encode categorical variable (conditions)
        le = LabelEncoder()
        df_reg['conditions_encoded'] = le.fit_transform(df_reg['conditions'].fillna('Unknown'))

        # Select features and target
        feature_cols = ['temp', 'precip', 'humidity', 'windspeed', 'visibility',
                        'conditions_encoded', 'hour', 'day_of_week', 'passenger_count', 'trip_distance']
        target_col = 'trip_duration_min'

        # Remove rows with missing values
        df_reg_clean = df_reg[feature_cols + [target_col]].dropna()

        X = df_reg_clean[feature_cols]
        y = df_reg_clean[target_col]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print(f"Regression: Training set size: {len(X_train):,}, Test set size: {len(X_test):,}")

        # Train multiple models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5)
        }

        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            regression_results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'model': model
            }

            print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")

        # Use best model for predictions
        best_model_name = max(regression_results.keys(), key=lambda k: regression_results[k]['R2'])
        best_model = regression_results[best_model_name]['model']

        # Create prediction DataFrame
        y_pred_best = best_model.predict(X_test)
        df_reg_pred = pd.DataFrame({
            'actual': y_test.values,
            'predicted': y_pred_best
        })

        # Feature importance (for tree-based models)
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"Feature importance calculated for {best_model_name}")

        print(f"Regression complete: Best model is {best_model_name}")

    kpis = {
        'avg_duration': safe_agg(df_monthly.get('avg_duration', pd.Series()), lambda s: s.mean()),
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

        html.Div([
            html.Div([dcc.Graph(id='monthly-duration-chart')], style={'flex': '1'}),
            html.Div([dcc.Graph(id='weekday-duration-chart')], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        html.Div([
            html.Div([dcc.Graph(id='heatmap-chart')], style={'flex': '1'}),
            html.Div([dcc.Graph(id='fare-weekday-chart')], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        html.Div([
            html.Div([dcc.Graph(id='hourly-pickups-chart')], style={'flex': '1'}),
            html.Div([dcc.Graph(id='monthly-trips-chart')], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        html.Div([
            html.Div([dcc.Graph(id='top-pickup-locations-chart')], style={'flex': '1'}),
            html.Div([dcc.Graph(id='vendor-distribution-chart')], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        # New: Maps row
        html.Div([
            html.Div([dcc.Graph(id='pickup-map')], style={'flex': '1'}),
            html.Div([dcc.Graph(id='dropoff-map')], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        html.Div([
            html.Div([dcc.Graph(id='weather-condition-chart')], style={'flex': '1'}),
            html.Div([dcc.Graph(id='temp-trips-scatter-chart')], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        html.Div([
            html.Div([dcc.Graph(id='weather-scatter-chart')], style={'flex': '1'}),
            html.Div([dcc.Graph(id='correlation-matrix-chart')], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),


        html.H2("Trip Type Clustering Analysis",
                style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 20, 'color': '#333'}),

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

        # Tip clustering chart
        html.Div([
            html.Div([dcc.Graph(id='tip-cluster-chart')],
                     style={'width': '100%', 'display': 'inline-block', 'marginTop': '20px'}),
        ]),

        # Regression Analysis Section
        html.H2("Supervised Learning: Trip Duration Prediction (Weather & Trip Features)",
                style={'textAlign': 'center', 'marginTop': 60, 'marginBottom': 20, 'color': '#333'}),

        html.Div([
            html.Div([dcc.Graph(id='regression-comparison-chart')], style={'flex': '1'}),
            html.Div([dcc.Graph(id='regression-predictions-chart')], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        html.Div([
            html.Div([dcc.Graph(id='feature-importance-chart')], style={'flex': '1'}),
            html.Div([dcc.Graph(id='residuals-chart')], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        html.Div([
            html.Div([dcc.Graph(id='weather-polynomial-actual-vs-predicted-chart')], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        html.H2("Linear & Polynomial Regression: Trip Distance vs Fare Amount",
                style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 20, 'color': '#333'}),

        html.Div([
            html.Div([dcc.Graph(id='zone-linear-regression-chart')], style={'flex': '1'}),
            html.Div([dcc.Graph(id='zone-polynomial-regression-chart')], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        html.Div([
            html.Div([dcc.Graph(id='zone-linear-actual-vs-predicted-chart')], style={'flex': '1'}),
            html.Div([dcc.Graph(id='zone-polynomial-actual-vs-predicted-chart')], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

        html.H2("Linear & Polynomial Regression: Trip Distance vs Trip Duration",
                style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 20, 'color': '#333'}),

        html.Div([
            html.Div([dcc.Graph(id='ml-polynomial-regression-chart')], style={'flex': '1'}),
            html.Div([dcc.Graph(id='ml-actual-vs-predicted-chart')], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}),

    ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#fafafa'})

    @callback(Output('monthly-duration-chart', 'figure'), Input('monthly-duration-chart', 'id'))
    def update_monthly_chart(_):
        fig = go.Figure()
        if not df_monthly.empty:
            fig.add_trace(go.Scatter(
                x=df_monthly['year_month'],
                y=df_monthly['avg_duration'],
                mode='lines+markers',
                name='Avg Duration',
                line=dict(color=COLORS['primary'], width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor=f"rgba(31, 119, 180, 0.2)",
                hovertemplate='<b>%{x|%B %Y}</b><br>Avg Duration: %{y:.1f} Min<extra></extra>'
            ))
        fig.update_layout(**create_chart_layout('Average Trip Duration by Month', 'Month', 'Trip Duration (Minutes)'))
        return fig

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
        fig.update_layout(
            **create_chart_layout('Average Trip Duration by Day of Week', 'Day of Week', 'Trip Duration (Minutes)'))
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

    @callback(Output('hourly-pickups-chart', 'figure'), Input('hourly-pickups-chart', 'id'))
    def update_hourly_chart(_):
        fig = go.Figure()
        if not df_hourly.empty:
            fig.add_trace(go.Scatter(
                x=df_hourly['hour'],
                y=df_hourly['trip_count'],
                mode='lines+markers',
                name='Trips',
                line=dict(color=COLORS['secondary'], width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor=f"rgba(255, 127, 14, 0.2)",
                hovertemplate='<b>%{x}:00</b><br>Trips: %{y:,.0f}<extra></extra>'
            ))
        fig.update_layout(**create_chart_layout('Trips by Hour of Day', 'Hour', 'Number of Trips'))
        fig.update_xaxes(tickmode='linear', dtick=2)
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
            zone_names = {
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

            df_sorted = df_top_pickup.copy()
            df_sorted['zone_name'] = df_sorted['pulocationid'].map(zone_names).fillna(
                'Zone ' + df_sorted['pulocationid'].astype(str)
            )
            df_sorted = df_sorted.sort_values('trip_count', ascending=True)

            fig.add_trace(go.Bar(
                x=df_sorted['trip_count'],
                y=df_sorted['zone_name'],
                orientation='h',
                marker_color=COLORS['primary'],
                hovertemplate='<b>%{y}</b><br>Trips: %{x:,}<br>Location ID: ' + df_sorted['pulocationid'].astype(
                    str) + '<extra></extra>'
            ))
        fig.update_layout(**create_chart_layout('Top 10 Pickup Locations', 'Number of Trips', 'Zone'))
        return fig

    @callback(Output('vendor-distribution-chart', 'figure'), Input('vendor-distribution-chart', 'id'))
    def update_payment_chart(_):
        fig = go.Figure()
        if not df_payment.empty:
            fig.add_trace(go.Pie(
                labels=df_payment['payment_name'],
                values=df_payment['trip_count'],
                hole=0.4,
                hovertemplate='<b>%{label}</b><br>Trips: %{value:,}<br>%{percent}<extra></extra>'
            ))
        fig.update_layout(title_text='Payment Type Distribution', height=400)
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
            marker=dict(size=size, color=merged["trip_count"], colorscale="Viridis", showscale=True),
            text=[f"{_label(lid)}<br>Trips: {cnt:,}" for lid, cnt in zip(merged["location_id"], merged["trip_count"])],
            hoverinfo="text"
        ))
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_center={"lon": -73.9851, "lat": 40.7589},
            mapbox_zoom=9, height=550, title=title,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig

    @callback(Output('pickup-map', 'figure'), Input('pickup-map', 'id'))
    def update_pickup_map(_):
        return _build_map_figure(df_pickup_all, "Pickup Locations (All)")

    @callback(Output('dropoff-map', 'figure'), Input('dropoff-map', 'id'))
    def update_dropoff_map(_):
        return _build_map_figure(df_dropoff_all, "Dropoff Locations (All)")

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

    @callback(Output('temp-trips-scatter-chart', 'figure'), Input('temp-trips-scatter-chart', 'id'))
    def update_temp_trips_scatter_chart(_):
        fig = go.Figure()
        if not df_temp_trips_scatter.empty:
            fig.add_trace(go.Scatter(
                x=df_temp_trips_scatter['temp'],
                y=df_temp_trips_scatter['trip_count'],
                mode='markers',
                marker=dict(size=10, color=df_temp_trips_scatter['trip_count'], colorscale='Viridis', showscale=True),
                hovertemplate='Temp: %{x:.1f}°F<br>Trips: %{y:,}<extra></extra>'
            ))
        fig.update_layout(**create_chart_layout('Temperature vs Trip Count', 'Temperature (°F)', 'Number of Trips', 500))
        return fig

    @callback(Output('weather-scatter-chart', 'figure'), Input('weather-scatter-chart', 'id'))
    def update_weather_scatter_chart(_):
        fig = go.Figure()
        if not df_weather_scatter.empty:
            sample_size = min(10000, len(df_weather_scatter))
            df_sample = df_weather_scatter.sample(sample_size)
            fig.add_trace(go.Scatter(
                x=df_sample['trip_distance'],
                y=df_sample['trip_duration_min'],
                mode='markers',
                marker=dict(size=5, color=df_sample['trip_duration_min'], colorscale='Viridis', showscale=True, opacity=0.5),
                hovertemplate='Distance: %{x:.1f} mi<br>Duration: %{y:.1f} min<extra></extra>'
            ))
        fig.update_layout(**create_chart_layout('Trip Distance vs Duration (per Trip)', 'Trip Distance (mi)', 'Trip Duration (min)', 500))
        return fig

    @callback(Output('ml-polynomial-regression-chart', 'figure'), Input('ml-polynomial-regression-chart', 'id'))
    def update_ml_polynomial_regression(_):
        fig = go.Figure()

        if not df_weather_scatter.empty and len(df_weather_scatter) > 10:
            X = df_weather_scatter['trip_distance'].values.reshape(-1, 1)
            y = df_weather_scatter['trip_duration_min'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)

            poly_reg_model = LinearRegression()
            poly_reg_model.fit(X_train_poly, y_train)

            X_poly = poly.transform(X)
            r_sq = poly_reg_model.score(X_poly, y)
            print(f"Trip Distance vs Duration - Polynomial R2: {r_sq:.4f}")

            fig.add_trace(go.Scatter(
                x=X_test.flatten(),
                y=y_test,
                mode='markers',
                name='Test Data (30%)',
                marker=dict(size=4, color='#ff7f0e', opacity=0.3),
                hovertemplate='Distance: %{x:.1f} mi<br>Duration: %{y:.1f} min<extra></extra>'
            ))

            fig.add_trace(go.Scatter(
                x=X_train.flatten(),
                y=y_train,
                mode='markers',
                name='Training Data (70%)',
                marker=dict(size=4, color='#1f77b4', opacity=0.3),
                hovertemplate='Distance: %{x:.1f} mi<br>Duration: %{y:.1f} min<extra></extra>'
            ))

            X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            X_line_poly = poly.transform(X_line)
            y_line = poly_reg_model.predict(X_line_poly)
            fig.add_trace(go.Scatter(
                x=X_line.flatten(),
                y=y_line,
                mode='lines',
                name='Polynomial Fit (Degree 2)',
                line=dict(color='red', width=3),
                hovertemplate='Distance: %{x:.1f} mi<br>Predicted: %{y:.1f} min<extra></extra>'
            ))

            fig.update_layout(
                title=f'Trip Distance vs Duration - Polynomial Regression (Degree 2)<br>' +
                      f'<sub>R2 Score: {r_sq:.4f} | Data Points: {len(X):,}</sub>',
                xaxis_title='Trip Distance (mi)',
                yaxis_title='Trip Duration (min)',
                height=600,
                template='plotly_white',
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                hovermode='closest'
            )
        else:
            fig.update_layout(
                title='Polynomial Regression - Insufficient Data',
                xaxis_title='Trip Distance (mi)',
                yaxis_title='Trip Duration (min)',
                height=600,
                template='plotly_white',
                annotations=[{
                    'text': 'Not enough data points for regression analysis',
                    'xref': 'paper', 'yref': 'paper',
                    'x': 0.5, 'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 16, 'color': 'gray'}
                }]
            )

        return fig

    @callback(Output('ml-actual-vs-predicted-chart', 'figure'), Input('ml-actual-vs-predicted-chart', 'id'))
    def update_ml_actual_vs_predicted(_):
        fig = go.Figure()

        if not df_weather_scatter.empty and len(df_weather_scatter) > 50:
            X = df_weather_scatter['trip_distance'].values.reshape(-1, 1)
            y = df_weather_scatter['trip_duration_min'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)

            poly_reg_model = LinearRegression()
            poly_reg_model.fit(X_train_poly, y_train)

            X_poly = poly.transform(X)
            y_pred_full = poly_reg_model.predict(X_poly)
            r_sq = poly_reg_model.score(X_poly, y)

            y_train_pred = poly_reg_model.predict(X_train_poly)
            y_test_pred = poly_reg_model.predict(X_test_poly)

            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_test_pred,
                mode='markers',
                name='Test Data (30%)',
                marker=dict(size=4, color='red', opacity=0.3),
                hovertemplate='Actual: %{x:.1f} min<br>Predicted: %{y:.1f} min<extra></extra>'
            ))

            fig.add_trace(go.Scatter(
                x=y_train,
                y=y_train_pred,
                mode='markers',
                name='Training Data (70%)',
                marker=dict(size=4, color='green', opacity=0.3),
                hovertemplate='Actual: %{x:.1f} min<br>Predicted: %{y:.1f} min<extra></extra>'
            ))

            min_val = min(y.min(), y_pred_full.min())
            max_val = max(y.max(), y_pred_full.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='black', width=2, dash='dash'),
                hovertemplate='Perfect Prediction Line<extra></extra>'
            ))

            fig.update_layout(
                title=f'Trip Distance → Duration: Actual vs Predicted<br>' +
                      f'<sub>R2 Score: {r_sq:.4f}</sub>',
                xaxis_title='Actual Duration (min)',
                yaxis_title='Predicted Duration (min)',
                height=600,
                template='plotly_white',
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                hovermode='closest'
            )
        else:
            fig.update_layout(
                title='Actual vs Predicted - Insufficient Data',
                xaxis_title='Actual Duration (min)',
                yaxis_title='Predicted Duration (min)',
                height=600,
                template='plotly_white',
                annotations=[{
                    'text': 'Not enough data points for regression analysis',
                    'xref': 'paper', 'yref': 'paper',
                    'x': 0.5, 'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 16, 'color': 'gray'}
                }]
            )

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
                marker=dict(colors=['#4ECDC4', '#FF6B6B']),
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
                marker_color='#4ECDC4',
                hovertemplate='Distance: %{y:.2f} mi<extra></extra>'
            ))
            fig.add_trace(go.Bar(
                name='Avg Duration (min)',
                x=cluster_info,
                y=cluster_stats['trip_duration_min'],
                marker_color='#FFD93D',
                hovertemplate='Duration: %{y:.2f} min<extra></extra>'
            ))
            fig.add_trace(go.Bar(
                name='Avg Fare ($)',
                x=cluster_info,
                y=cluster_stats['fare_amount'],
                marker_color='#FF6B6B',
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
                'Economy / Short Trips': '#4ECDC4',
                'Premium / Long Trips': '#FF6B6B'
            }

            for trip_type in cluster_hourly['trip_type'].unique():
                data = cluster_hourly[cluster_hourly['trip_type'] == trip_type]
                fig.add_trace(go.Scatter(
                    x=data['hour'],
                    y=data['count'],
                    mode='lines+markers',
                    name=trip_type,
                    line=dict(color=colors.get(trip_type, '#999'), width=3),
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

                sampled = cluster_data.sample(min(samples_for_cluster, cluster_size))
                df_sample_list.append(sampled)

            df_sample = pd.concat(df_sample_list, ignore_index=True)

            colors_map = {
                'Economy / Short Trips': '#4ECDC4',
                'Premium / Long Trips': '#FF6B6B'
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
                        color=colors_map.get(trip_type, '#999'),
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

    @callback(Output('tip-cluster-chart', 'figure'), Input('tip-cluster-chart', 'id'))
    def update_tip_cluster_chart(_):
        fig = go.Figure()

        if not df_cluster.empty:
            from sklearn.cluster import KMeans

            # Prepare features
            features = ['fare_amount', 'tip_amount']
            X = df_cluster[features].values

            # Run KMeans
            n_clusters = 3
            kmeans = KMeans(n_clusters=n_clusters, n_init=10)
            clusters = kmeans.fit_predict(X)
            df_cluster['cluster'] = clusters

            # Count points per cluster
            cluster_counts = df_cluster['cluster'].value_counts().sort_index()
            total_points = len(df_cluster)

            # Define colors for clusters
            colors = ['#4ECDC4', '#FF6B6B', '#FFD93D']

            # Plot points per cluster
            for c in range(n_clusters):
                cluster_data = df_cluster[df_cluster['cluster'] == c]
                fig.add_trace(go.Scatter(
                    x=cluster_data['fare_amount'],
                    y=cluster_data['tip_amount'],
                    mode='markers',
                    marker=dict(color=colors[c], size=6),
                    name=f'Cluster {c} ({cluster_counts[c]} points)'
                ))

            # Plot centroids
            centroids = kmeans.cluster_centers_
            fig.add_trace(go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode='markers+text',
                marker=dict(size=12, symbol='x', color='black'),
                text=[f'C{i}' for i in range(n_clusters)],
                textposition='top center',
                name='Centroids'
            ))

            title_text = f'Tip Clustering — Fare vs Tip (Total: {total_points:,} points)'
        else:
            title_text = 'Tip Clustering — Fare vs Tip (No data)'

        fig.update_layout(
            title=title_text,
            xaxis_title='Fare Amount ($)',
            yaxis_title='Tip Amount ($)',
            template='plotly_white',
            height=600
        )

        return fig

    @callback(Output('regression-comparison-chart', 'figure'), Input('regression-comparison-chart', 'id'))
    def update_regression_comparison(_):
        fig = go.Figure()

        if regression_results:
            model_names = list(regression_results.keys())
            rmse_values = [regression_results[m]['RMSE'] for m in model_names]
            mae_values = [regression_results[m]['MAE'] for m in model_names]
            r2_values = [regression_results[m]['R2'] for m in model_names]

            fig.add_trace(go.Bar(
                name='RMSE (min)',
                x=model_names,
                y=rmse_values,
                marker_color='#FF6B6B',
                yaxis='y',
                hovertemplate='RMSE: %{y:.2f} min<extra></extra>'
            ))

            fig.add_trace(go.Bar(
                name='MAE (min)',
                x=model_names,
                y=mae_values,
                marker_color='#4ECDC4',
                yaxis='y',
                hovertemplate='MAE: %{y:.2f} min<extra></extra>'
            ))

            fig.add_trace(go.Scatter(
                name='R2 Score',
                x=model_names,
                y=r2_values,
                mode='lines+markers',
                marker=dict(size=12, color='#FFD93D'),
                line=dict(width=3),
                yaxis='y2',
                hovertemplate='R2: %{y:.4f}<extra></extra>'
            ))

        fig.update_layout(
            title='Regression Model Performance Comparison',
            xaxis_title='Model',
            yaxis=dict(title='Error (minutes)', side='left'),
            yaxis2=dict(title='R2 Score', side='right', overlaying='y', range=[0, 1]),
            barmode='group',
            height=450,
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='x unified'
        )
        return fig

    @callback(Output('regression-predictions-chart', 'figure'), Input('regression-predictions-chart', 'id'))
    def update_regression_predictions(_):
        fig = go.Figure()

        best_r2 = 0
        if regression_results:
            best_model_name = max(regression_results.keys(), key=lambda k: regression_results[k]['R2'])
            best_r2 = regression_results[best_model_name]['R2']

        if not df_reg_pred.empty:
            df_plot = df_reg_pred.sample(min(5000, len(df_reg_pred)))

            fig.add_trace(go.Scatter(
                x=df_plot['actual'],
                y=df_plot['predicted'],
                mode='markers',
                marker=dict(size=5, color='#4ECDC4', opacity=0.5),
                name='Predictions',
                hovertemplate='Actual: %{x:.1f} min<br>Predicted: %{y:.1f} min<extra></extra>'
            ))

            max_val = max(df_plot['actual'].max(), df_plot['predicted'].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Perfect Prediction',
                hoverinfo='skip'
            ))

        fig.update_layout(
            title=f'Actual vs Predicted Trip Duration (Best Model)<br><sub>R2 Score: {best_r2:.4f}</sub>',
            xaxis_title='Actual Duration (minutes)',
            yaxis_title='Predicted Duration (minutes)',
            height=450,
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        return fig

    @callback(Output('feature-importance-chart', 'figure'), Input('feature-importance-chart', 'id'))
    def update_feature_importance(_):
        fig = go.Figure()

        if not feature_importance.empty:
            # Map feature names to readable labels
            feature_labels = {
                'temp': 'Temperature',
                'precip': 'Precipitation',
                'humidity': 'Humidity',
                'windspeed': 'Wind Speed',
                'visibility': 'Visibility',
                'conditions_encoded': 'Weather Conditions',
                'hour': 'Hour of Day',
                'day_of_week': 'Day of Week',
                'passenger_count': 'Passenger Count',
                'trip_distance': 'Trip Distance'
            }

            df_imp = feature_importance.copy()
            df_imp['feature_label'] = df_imp['feature'].map(feature_labels)

            fig.add_trace(go.Bar(
                x=df_imp['importance'],
                y=df_imp['feature_label'],
                orientation='h',
                marker_color='#FF6B6B',
                hovertemplate='%{y}<br>Importance: %{x:.4f}<extra></extra>'
            ))

        fig.update_layout(
            title='Feature Importance (Best Model)',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            height=450,
            template='plotly_white',
            margin=dict(l=150, r=50, t=60, b=60)
        )
        return fig

    @callback(Output('residuals-chart', 'figure'), Input('residuals-chart', 'id'))
    def update_residuals(_):
        fig = go.Figure()

        if not df_reg_pred.empty:
            # Calculate residuals
            df_plot = df_reg_pred.copy()
            df_plot['residual'] = df_plot['actual'] - df_plot['predicted']

            # Sample for visualization
            df_plot = df_plot.sample(min(5000, len(df_plot)))

            fig.add_trace(go.Scatter(
                x=df_plot['predicted'],
                y=df_plot['residual'],
                mode='markers',
                marker=dict(size=5, color='#4ECDC4', opacity=0.5),
                hovertemplate='Predicted: %{x:.1f} min<br>Residual: %{y:.1f} min<extra></extra>'
            ))

            # Zero line
            fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)

        fig.update_layout(
            title='Residual Plot (Prediction Errors)',
            xaxis_title='Predicted Duration (minutes)',
            yaxis_title='Residual (Actual - Predicted)',
            height=450,
            template='plotly_white'
        )
        return fig

    @callback(Output('weather-polynomial-actual-vs-predicted-chart', 'figure'), Input('weather-polynomial-actual-vs-predicted-chart', 'id'))
    def update_weather_polynomial_actual_vs_predicted(_):
        fig = go.Figure()

        if not df_reg_pred.empty and len(df_reg_pred) > 50:
            df_sample = df_reg_pred.sample(min(50000, len(df_reg_pred)))
            X = df_sample['actual'].values.reshape(-1, 1)
            y = df_sample['predicted'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)

            poly_reg_model = LinearRegression()
            poly_reg_model.fit(X_train_poly, y_train)

            X_poly = poly.transform(X)
            y_pred_full = poly_reg_model.predict(X_poly)
            r_sq = poly_reg_model.score(X_poly, y)

            y_train_pred = poly_reg_model.predict(X_train_poly)
            y_test_pred = poly_reg_model.predict(X_test_poly)

            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_test_pred,
                mode='markers',
                name='Test Data (30%)',
                marker=dict(size=4, color='red', opacity=0.3),
                hovertemplate='Actual: %{x:.1f} min<br>Predicted: %{y:.1f} min<extra></extra>'
            ))

            fig.add_trace(go.Scatter(
                x=y_train,
                y=y_train_pred,
                mode='markers',
                name='Training Data (70%)',
                marker=dict(size=4, color='green', opacity=0.3),
                hovertemplate='Actual: %{x:.1f} min<br>Predicted: %{y:.1f} min<extra></extra>'
            ))

            min_val = min(y.min(), y_pred_full.min())
            max_val = max(y.max(), y_pred_full.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='black', width=2, dash='dash'),
                hovertemplate='Perfect Prediction Line<extra></extra>'
            ))

            fig.update_layout(
                title=f'Weather: Polynomial Regression - Actual vs Predicted<br>' +
                      f'<sub>R2 Score: {r_sq:.4f}</sub>',
                xaxis_title='Actual Duration (min)',
                yaxis_title='Predicted Duration (min)',
                height=600,
                template='plotly_white',
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                hovermode='closest'
            )
        else:
            fig.update_layout(
                title='Weather Polynomial: Actual vs Predicted - Insufficient Data',
                xaxis_title='Actual Duration (min)',
                yaxis_title='Predicted Duration (min)',
                height=600,
                template='plotly_white'
            )

        return fig

    @callback(Output('zone-linear-regression-chart', 'figure'), Input('zone-linear-regression-chart', 'id'))
    def update_zone_linear_regression(_):
        fig = go.Figure()

        if not df_zone_regression.empty and len(df_zone_regression) > 10:
            X = df_zone_regression['trip_distance'].values.reshape(-1, 1)
            y = df_zone_regression['fare_amount'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)

            r_sq = linear_model.score(X, y)
            print(f"Distance vs Fare - Linear R2: {r_sq:.4f}")

            fig.add_trace(go.Scatter(
                x=X_test.flatten(),
                y=y_test,
                mode='markers',
                name='Test Data (30%)',
                marker=dict(size=4, color='#ff7f0e', opacity=0.3),
                hovertemplate='Distance: %{x:.1f} mi<br>Fare: $%{y:.2f}<extra></extra>'
            ))

            fig.add_trace(go.Scatter(
                x=X_train.flatten(),
                y=y_train,
                mode='markers',
                name='Training Data (70%)',
                marker=dict(size=4, color='#1f77b4', opacity=0.3),
                hovertemplate='Distance: %{x:.1f} mi<br>Fare: $%{y:.2f}<extra></extra>'
            ))

            X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_line = linear_model.predict(X_line)
            fig.add_trace(go.Scatter(
                x=X_line.flatten(),
                y=y_line,
                mode='lines',
                name='Linear Fit',
                line=dict(color='red', width=3),
                hovertemplate='Distance: %{x:.1f} mi<br>Predicted: $%{y:.2f}<extra></extra>'
            ))

            fig.update_layout(
                title=f'Trip Distance vs Fare Amount - Linear Regression<br>' +
                      f'<sub>R2 Score: {r_sq:.4f} | Data Points: {len(X):,}</sub>',
                xaxis_title='Trip Distance (mi)',
                yaxis_title='Fare Amount ($)',
                height=600,
                template='plotly_white',
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                hovermode='closest'
            )
        else:
            fig.update_layout(
                title='Distance vs Fare Linear Regression - Insufficient Data',
                xaxis_title='Trip Distance (mi)',
                yaxis_title='Fare Amount ($)',
                height=600,
                template='plotly_white',
                annotations=[{
                    'text': 'Not enough data points for regression analysis',
                    'xref': 'paper', 'yref': 'paper',
                    'x': 0.5, 'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 16, 'color': 'gray'}
                }]
            )

        return fig

    @callback(Output('zone-linear-actual-vs-predicted-chart', 'figure'), Input('zone-linear-actual-vs-predicted-chart', 'id'))
    def update_zone_linear_actual_vs_predicted(_):
        fig = go.Figure()

        if not df_zone_regression.empty and len(df_zone_regression) > 10:
            X = df_zone_regression['trip_distance'].values.reshape(-1, 1)
            y = df_zone_regression['fare_amount'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)

            r_sq = linear_model.score(X, y)

            y_train_pred = linear_model.predict(X_train)
            y_test_pred = linear_model.predict(X_test)

            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_test_pred,
                mode='markers',
                name='Test Data (30%)',
                marker=dict(size=4, color='red', opacity=0.3),
                hovertemplate='Actual: $%{x:.2f}<br>Predicted: $%{y:.2f}<extra></extra>'
            ))

            fig.add_trace(go.Scatter(
                x=y_train,
                y=y_train_pred,
                mode='markers',
                name='Training Data (70%)',
                marker=dict(size=4, color='green', opacity=0.3),
                hovertemplate='Actual: $%{x:.2f}<br>Predicted: $%{y:.2f}<extra></extra>'
            ))

            y_pred_full = linear_model.predict(X)
            min_val = min(y.min(), y_pred_full.min())
            max_val = max(y.max(), y_pred_full.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='black', width=2, dash='dash'),
                hovertemplate='Perfect Prediction Line<extra></extra>'
            ))

            fig.update_layout(
                title=f'Distance → Fare: Actual vs Predicted (Linear)<br>' +
                      f'<sub>R2 Score: {r_sq:.4f}</sub>',
                xaxis_title='Actual Fare ($)',
                yaxis_title='Predicted Fare ($)',
                height=600,
                template='plotly_white',
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                hovermode='closest'
            )
        else:
            fig.update_layout(
                title='Distance → Fare: Actual vs Predicted - Insufficient Data',
                xaxis_title='Actual Fare ($)',
                yaxis_title='Predicted Fare ($)',
                height=600,
                template='plotly_white',
                annotations=[{
                    'text': 'Not enough data points for regression analysis',
                    'xref': 'paper', 'yref': 'paper',
                    'x': 0.5, 'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 16, 'color': 'gray'}
                }]
            )

        return fig

    @callback(Output('zone-polynomial-regression-chart', 'figure'), Input('zone-polynomial-regression-chart', 'id'))
    def update_zone_polynomial_regression(_):
        fig = go.Figure()

        if not df_zone_regression.empty and len(df_zone_regression) > 10:
            X = df_zone_regression['trip_distance'].values.reshape(-1, 1)
            y = df_zone_regression['fare_amount'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)

            poly_reg_model = LinearRegression()
            poly_reg_model.fit(X_train_poly, y_train)

            X_poly = poly.transform(X)

            r_sq = poly_reg_model.score(X_poly, y)
            print(f"Distance vs Fare - Polynomial R2: {r_sq:.4f}")

            fig.add_trace(go.Scatter(
                x=X_test.flatten(),
                y=y_test,
                mode='markers',
                name='Test Data (30%)',
                marker=dict(size=4, color='#ff7f0e', opacity=0.3),
                hovertemplate='Distance: %{x:.1f} mi<br>Fare: $%{y:.2f}<extra></extra>'
            ))

            fig.add_trace(go.Scatter(
                x=X_train.flatten(),
                y=y_train,
                mode='markers',
                name='Training Data (70%)',
                marker=dict(size=4, color='#1f77b4', opacity=0.3),
                hovertemplate='Distance: %{x:.1f} mi<br>Fare: $%{y:.2f}<extra></extra>'
            ))

            X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            X_line_poly = poly.transform(X_line)
            y_line = poly_reg_model.predict(X_line_poly)
            fig.add_trace(go.Scatter(
                x=X_line.flatten(),
                y=y_line,
                mode='lines',
                name='Polynomial Fit (Degree 2)',
                line=dict(color='red', width=3),
                hovertemplate='Distance: %{x:.1f} mi<br>Predicted: $%{y:.2f}<extra></extra>'
            ))

            fig.update_layout(
                title=f'Trip Distance vs Fare Amount - Polynomial Regression (Degree 2)<br>' +
                      f'<sub>R2 Score: {r_sq:.4f} | Data Points: {len(X):,}</sub>',
                xaxis_title='Trip Distance (mi)',
                yaxis_title='Fare Amount ($)',
                height=600,
                template='plotly_white',
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                hovermode='closest'
            )
        else:
            fig.update_layout(
                title='Distance vs Fare Polynomial Regression - Insufficient Data',
                xaxis_title='Trip Distance (mi)',
                yaxis_title='Fare Amount ($)',
                height=600,
                template='plotly_white',
                annotations=[{
                    'text': 'Not enough data points for regression analysis',
                    'xref': 'paper', 'yref': 'paper',
                    'x': 0.5, 'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 16, 'color': 'gray'}
                }]
            )

        return fig

    @callback(Output('zone-polynomial-actual-vs-predicted-chart', 'figure'), Input('zone-polynomial-actual-vs-predicted-chart', 'id'))
    def update_zone_polynomial_actual_vs_predicted(_):
        fig = go.Figure()

        if not df_zone_regression.empty and len(df_zone_regression) > 10:
            X = df_zone_regression['trip_distance'].values.reshape(-1, 1)
            y = df_zone_regression['fare_amount'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)

            poly_reg_model = LinearRegression()
            poly_reg_model.fit(X_train_poly, y_train)

            X_poly = poly.transform(X)
            y_pred_full = poly_reg_model.predict(X_poly)

            r_sq = poly_reg_model.score(X_poly, y)

            y_train_pred = poly_reg_model.predict(X_train_poly)
            y_test_pred = poly_reg_model.predict(X_test_poly)

            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_test_pred,
                mode='markers',
                name='Test Data (30%)',
                marker=dict(size=4, color='red', opacity=0.3),
                hovertemplate='Actual: $%{x:.2f}<br>Predicted: $%{y:.2f}<extra></extra>'
            ))

            fig.add_trace(go.Scatter(
                x=y_train,
                y=y_train_pred,
                mode='markers',
                name='Training Data (70%)',
                marker=dict(size=4, color='green', opacity=0.3),
                hovertemplate='Actual: $%{x:.2f}<br>Predicted: $%{y:.2f}<extra></extra>'
            ))

            min_val = min(y.min(), y_pred_full.min())
            max_val = max(y.max(), y_pred_full.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='black', width=2, dash='dash'),
                hovertemplate='Perfect Prediction Line<extra></extra>'
            ))

            fig.update_layout(
                title=f'Distance → Fare: Actual vs Predicted (Polynomial)<br>' +
                      f'<sub>R² Score: {r_sq:.4f}</sub>',
                xaxis_title='Actual Fare ($)',
                yaxis_title='Predicted Fare ($)',
                height=600,
                template='plotly_white',
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                hovermode='closest'
            )
        else:
            fig.update_layout(
                title='Distance → Fare: Actual vs Predicted - Insufficient Data',
                xaxis_title='Actual Fare ($)',
                yaxis_title='Predicted Fare ($)',
                height=600,
                template='plotly_white',
                annotations=[{
                    'text': 'Not enough data points for regression analysis',
                    'xref': 'paper', 'yref': 'paper',
                    'x': 0.5, 'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 16, 'color': 'gray'}
                }]
            )

        return fig

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8051, debug=False)