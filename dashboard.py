import calendar
import numpy as np
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
from sqlalchemy import text
from db_connection import DBConnection

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

def get_db_engine():
    """Returns a singleton database engine."""
    if not hasattr(get_db_engine, 'engine'):
        db = DBConnection("postgres", "password123", "localhost", 5433, "ny_taxi_dwh")
        get_db_engine.engine = db.connect()
    return get_db_engine.engine

def execute_query(query: str, engine) -> pd.DataFrame:
    """Executes a SQL query and returns the result as a DataFrame."""
    if engine is None:
        return pd.DataFrame()

    try:
        return pd.read_sql(text(query), engine)
    except Exception as e:
        print(f"Query error: {e}")
        return pd.DataFrame()

def load_data():
    """Loads all necessary data from the database."""
    # Use single engine for all queries
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
        """
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
     df_weather_monthly, df_weather_conditions, df_weather_scatter) = load_data()

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

    kpis = {
        'avg_duration': safe_agg(df_monthly.get('avg_duration', pd.Series()), lambda s: s.mean()),
        'total_trips': int(df_monthly['trip_count'].sum()) if not df_monthly.empty else 0
    }

    app.layout = html.Div([
        html.H1("NYC Yellow Taxi Dashboard", style={'textAlign': 'center', 'marginBottom': 30, 'color': '#FFD700'}),

        # KPI Cards
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
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '40px', 'flexWrap': 'wrap'}),

        # Charts Row 1
        html.Div([
            html.Div([dcc.Graph(id='monthly-duration-chart')], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            html.Div([dcc.Graph(id='weekday-duration-chart')], style={'width': '48%', 'display': 'inline-block'}),
        ]),

        # Charts Row 2
        html.Div([
            html.Div([dcc.Graph(id='heatmap-chart')], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            html.Div([dcc.Graph(id='fare-weekday-chart')], style={'width': '48%', 'display': 'inline-block'}),
        ]),

        # Charts Row 3
        html.Div([
            html.Div([dcc.Graph(id='hourly-pickups-chart')], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            html.Div([dcc.Graph(id='monthly-trips-chart')], style={'width': '48%', 'display': 'inline-block'}),
        ]),

        # Charts Row 4
        html.Div([
            html.Div([dcc.Graph(id='top-pickup-locations-chart')], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            html.Div([dcc.Graph(id='vendor-distribution-chart')], style={'width': '48%', 'display': 'inline-block'}),
        ]),

        # Charts Row 5 (Weather)
        html.Div([
            html.Div([dcc.Graph(id='weather-condition-chart')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='weather-scatter-chart')], style={'width': '100%', 'display': 'inline-block'}),
        ]),

    ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#fafafa'})

    # Callbacks
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
            fig.add_trace(go.Bar(
                x=df_top_pickup['pulocationid'].astype(str),
                y=df_top_pickup['trip_count'],
                marker_color=COLORS['primary'],
                hovertemplate='PULocationID: %{x}<br>Trips: %{y:,}<extra></extra>'
            ))
        fig.update_layout(**create_chart_layout('Top 10 Pickup Locations', 'Pickup Location ID', 'Number of Trips'))
        return fig

    @callback(Output('vendor-distribution-chart', 'figure'), Input('vendor-distribution-chart', 'id'))
    def update_payment_chart(_):
        fig = go.Figure()
        if not df_payment.empty:
            fig.add_trace(go.Pie(
                labels=df_payment['payment_type'],
                values=df_payment['trip_count'],
                hole=0.4,
                hovertemplate='%{label}: %{value:,} trips<extra></extra>'
            ))
        fig.update_layout(title_text='Payment Type Distribution', height=400)
        return fig

    # ---------------- Weather Charts ---------------- #
    @callback(Output('weather-monthly-chart', 'figure'), Input('weather-monthly-chart', 'id'))
    def update_weather_monthly_chart(_):
        fig = go.Figure()
        if not df_weather_monthly.empty:
            fig.add_trace(go.Scatter(
                x=df_weather_monthly['year_month'],
                y=df_weather_monthly['avg_temp'],
                name='Avg Temp (°C)',
                line=dict(color='red', width=3),
                mode='lines+markers'
            ))
            fig.add_trace(go.Bar(
                x=df_weather_monthly['year_month'],
                y=df_weather_monthly['trip_count'],
                name='Trips',
                marker_color='blue',
                opacity=0.5,
                yaxis='y2'
            ))
            fig.update_layout(
                **create_chart_layout('Monthly Avg Temperature vs Trips', 'Month', 'Avg Temp (°C)', 500),
                yaxis2=dict(
                    title='Trip Count',
                    overlaying='y',
                    side='right'
                ),
                legend=dict(x=0.1, y=1.1)
            )
        return fig

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

    return app

if __name__ == '__main__':
    app = create_app()
    app.run_server(host='0.0.0.0', port=8051, debug=False)
import calendar
import numpy as np
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
from sqlalchemy import text
from db_connection import DBConnection

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

def get_db_engine():
    """Returns a singleton database engine."""
    if not hasattr(get_db_engine, 'engine'):
        db = DBConnection("postgres", "password123", "localhost", 5433, "ny_taxi_dwh")
        get_db_engine.engine = db.connect()
    return get_db_engine.engine

def execute_query(query: str, engine) -> pd.DataFrame:
    """Executes a SQL query and returns the result as a DataFrame."""
    if engine is None:
        return pd.DataFrame()
    try:
        return pd.read_sql(text(query), engine)
    except Exception as e:
        print(f"Query error: {e}")
        return pd.DataFrame()

def load_data():
    """Loads all necessary data from the database, including weather."""
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
                   COUNT(*) as trip_count
            FROM fact_trips ft
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            WHERE dd.year >= {YEAR_FILTER}
            GROUP BY ft.payment_type
            ORDER BY trip_count DESC
        """,
        # -------- Weather Queries -------- #
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
        """
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
     df_weather_monthly, df_weather_conditions, df_weather_scatter) = load_data()

    # Prepare monthly dates
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

    kpis = {
        'avg_duration': safe_agg(df_monthly.get('avg_duration', pd.Series()), lambda s: s.mean()),
        'total_trips': int(df_monthly['trip_count'].sum()) if not df_monthly.empty else 0
    }

    app.layout = html.Div([
        html.H1("NYC Yellow Taxi Dashboard", style={'textAlign': 'center', 'marginBottom': 30, 'color': '#FFD700'}),

        # KPI Cards
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
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '40px', 'flexWrap': 'wrap'}),

        # Charts Row 1
        html.Div([
            html.Div([dcc.Graph(id='monthly-duration-chart')], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            html.Div([dcc.Graph(id='weekday-duration-chart')], style={'width': '48%', 'display': 'inline-block'}),
        ]),

        # Charts Row 2
        html.Div([
            html.Div([dcc.Graph(id='heatmap-chart')], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            html.Div([dcc.Graph(id='fare-weekday-chart')], style={'width': '48%', 'display': 'inline-block'}),
        ]),

        # Charts Row 3
        html.Div([
            html.Div([dcc.Graph(id='hourly-pickups-chart')], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            html.Div([dcc.Graph(id='monthly-trips-chart')], style={'width': '48%', 'display': 'inline-block'}),
        ]),

        # Charts Row 4
        html.Div([
            html.Div([dcc.Graph(id='top-pickup-locations-chart')], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            html.Div([dcc.Graph(id='vendor-distribution-chart')], style={'width': '48%', 'display': 'inline-block'}),
        ]),

        # Charts Row 5 (Weather)
        html.Div([
            html.Div([dcc.Graph(id='weather-monthly-chart')], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            html.Div([dcc.Graph(id='weather-condition-chart')], style={'width': '48%', 'display': 'inline-block'}),
        ]),

        html.Div([
            html.Div([dcc.Graph(id='weather-scatter-chart')], style={'width': '100%', 'display': 'inline-block'}),
        ])
    ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#fafafa'})

    # ---------------- Callbacks ---------------- #
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
            fig.add_trace(go.Bar(
                x=df_top_pickup['pulocationid'].astype(str),
                y=df_top_pickup['trip_count'],
                marker_color=COLORS['primary'],
                hovertemplate='PULocationID: %{x}<br>Trips: %{y:,}<extra></extra>'
            ))
        fig.update_layout(**create_chart_layout('Top 10 Pickup Locations', 'Pickup Location ID', 'Number of Trips'))
        return fig

    @callback(Output('vendor-distribution-chart', 'figure'), Input('vendor-distribution-chart', 'id'))
    def update_payment_chart(_):
        fig = go.Figure()
        if not df_payment.empty:
            fig.add_trace(go.Pie(
                labels=df_payment['payment_type'],
                values=df_payment['trip_count'],
                hole=0.4,
                hovertemplate='%{label}: %{value:,} trips<extra></extra>'
            ))
        fig.update_layout(title_text='Payment Type Distribution', height=400)
        return fig

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

    return app

if __name__ == '__main__':
    app = create_app()
    app.run_server(host='0.0.0.0', port=8051, debug=False)
