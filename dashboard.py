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
DAY_NAMES = ['Sonntag', 'Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag']
WEEKEND_DAYS = {'Samstag', 'Sonntag'}
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'weekday': '#4ECDC4',
    'weekend': '#FF6B6B'
}

def execute_query(query: str) -> pd.DataFrame:
    """Executes a SQL query and returns the result as a DataFrame."""
    db = DBConnection("postgres", "password123", "localhost", 5433, "ny_taxi_dwh")
    engine = db.connect()

    if engine is None:
        return pd.DataFrame()

    try:
        return pd.read_sql(text(query), engine)
    except Exception as e:
        print(f"Query error: {e}")
        return pd.DataFrame()

def load_data():
    """Loads all necessary data from the database."""
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
        SELECT ft."PULocationID" as PULocationID,
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
        """
    }

    return tuple(execute_query(query) for query in queries.values())

def create_chart_layout(title: str, xaxis: str, yaxis: str, height: int = 450):
    """Creates a consistent chart layout."""
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
    """Safely aggregate series with fallback."""
    return func(series) if not series.empty and not series.isna().all() else default

def create_app():
    """Creates and configures the Dash application."""
    app = dash.Dash(__name__)

    # Load and prepare data
    df_monthly, df_weekday, df_heatmap, df_fare_weekday, df_hourly, df_top_pickup, df_payment = load_data()

    # Prepare monthly data
    if not df_monthly.empty:
        df_monthly['year_month'] = pd.to_datetime(
            df_monthly[['year', 'month']].assign(day=1)
        )

    # Prepare weekday data
    if not df_weekday.empty:
        df_weekday = df_weekday.sort_values('day_of_week')
        df_weekday['day_name'] = df_weekday['day_of_week'].astype(int).map(lambda x: DAY_NAMES[x])

    # Prepare fare weekday data
    if not df_fare_weekday.empty:
        df_fare_weekday = df_fare_weekday.sort_values('day_of_week')
        df_fare_weekday['day_name'] = df_fare_weekday['day_of_week'].astype(int).map(lambda x: DAY_NAMES[x])

    # Calculate KPIs
    kpis = {
        'avg_duration': safe_agg(df_monthly.get('avg_duration', pd.Series()), lambda s: s.mean()),
        'total_trips': int(df_monthly['trip_count'].sum()) if not df_monthly.empty else 0
    }

    # Dashboard Layout
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
                ("Ø Fahrtdauer", f"{kpis['avg_duration']:.1f} Min", COLORS['primary']),
                ("Gesamt Fahrten", f"{kpis['total_trips']:,}", COLORS['danger']),
            ]]
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '40px', 'flexWrap': 'wrap'}),

        # Charts Row 1
        html.Div([
            html.Div([dcc.Graph(id='monthly-duration-chart')],
                     style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            html.Div([dcc.Graph(id='weekday-duration-chart')],
                     style={'width': '48%', 'display': 'inline-block'}),
        ]),

        # Charts Row 2
        html.Div([
            html.Div([dcc.Graph(id='heatmap-chart')],
                     style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            html.Div([dcc.Graph(id='fare-weekday-chart')],
                     style={'width': '48%', 'display': 'inline-block'}),
        ]),

        # Charts Row 3
        html.Div([
            html.Div([dcc.Graph(id='hourly-pickups-chart')],
                     style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            html.Div([dcc.Graph(id='monthly-trips-chart')],
                     style={'width': '48%', 'display': 'inline-block'}),
        ]),

        # Charts Row 4
        html.Div([
            html.Div([dcc.Graph(id='top-pickup-locations-chart')],
                     style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            html.Div([dcc.Graph(id='vendor-distribution-chart')],
                     style={'width': '48%', 'display': 'inline-block'}),
        ]),
    ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#fafafa'})

    # Callbacks
    @callback(Output('monthly-duration-chart', 'figure'),
              Input('monthly-duration-chart', 'id'))
    def update_monthly_chart(_):
        fig = go.Figure()
        if not df_monthly.empty:
            fig.add_trace(go.Scatter(
                x=df_monthly['year_month'],
                y=df_monthly['avg_duration'],
                mode='lines+markers',
                name='Ø Fahrtdauer',
                line=dict(color=COLORS['primary'], width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor=f"rgba(31, 119, 180, 0.2)",
                hovertemplate='<b>%{x|%B %Y}</b><br>Ø Dauer: %{y:.1f} Min<extra></extra>'
            ))
            for i in range(len(df_monthly)):
                fig.add_annotation(
                    x=df_monthly.iloc[i]['year_month'],
                    y=df_monthly.iloc[i]['avg_duration'],
                    text=f"{df_monthly.iloc[i]['avg_duration']:.1f}",
                    showarrow=False,
                    yshift=10,
                    font=dict(size=9, color='#333')
                )
        fig.update_layout(**create_chart_layout(
            'Durchschnittliche Fahrtdauer pro Monat', 'Monat', 'Fahrtdauer (Minuten)'
        ))
        return fig

    @callback(Output('weekday-duration-chart', 'figure'),
              Input('weekday-duration-chart', 'id'))
    def update_weekday_chart(_):
        fig = go.Figure()
        if not df_weekday.empty:
            colors = [COLORS['weekend'] if day in WEEKEND_DAYS else COLORS['weekday']
                     for day in df_weekday['day_name']]

            fig.add_trace(go.Bar(
                x=df_weekday['day_name'],
                y=df_weekday['avg_duration'],
                name='Ø Fahrtdauer',
                marker_color=colors,
                hovertemplate='<b>%{x}</b><br>Ø Dauer: %{y:.1f} Min<extra></extra>'
            ))

            for i, (day, duration) in enumerate(zip(df_weekday['day_name'], df_weekday['avg_duration'])):
                fig.add_annotation(
                    x=day,
                    y=duration,
                    text=f"{duration:.1f}",
                    showarrow=False,
                    yshift=10,
                    font=dict(size=10, color='#333')
                )

        fig.update_layout(**create_chart_layout(
            'Durchschnittliche Fahrtdauer nach Wochentag', 'Wochentag', 'Fahrtdauer (Minuten)'
        ))
        return fig

    @callback(Output('heatmap-chart', 'figure'),
              Input('heatmap-chart', 'id'))
    def update_heatmap_chart(_):
        fig = go.Figure()

        if not df_heatmap.empty:
            # Create pivot table
            pivot_data = df_heatmap.pivot(index='day_of_week', columns='month', values='trip_count')
            pivot_data = pivot_data.reindex(range(7)).fillna(0)
            pivot_data.index = DAY_NAMES
            pivot_data.columns = [calendar.month_name[i] for i in pivot_data.columns if i <= 12]

            fig.add_trace(go.Heatmap(
                z=pivot_data.values,
                x=list(pivot_data.columns),
                y=list(pivot_data.index),
                colorscale='Viridis',
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br><b>%{x}</b><br>Fahrten: %{z:,.0f}<extra></extra>'
            ))

        fig.update_layout(**create_chart_layout(
            'Fahrten-Heatmap: Wochentag vs Monat', 'Monat', 'Wochentag', 500
        ))
        return fig

    @callback(Output('fare-weekday-chart', 'figure'),
              Input('fare-weekday-chart', 'id'))
    def update_fare_weekday_chart(_):
        fig = go.Figure()

        if not df_fare_weekday.empty:
            fig.add_trace(go.Bar(
                x=df_fare_weekday['day_name'],
                y=df_fare_weekday['total_revenue'],
                name='Gesamtumsatz',
                marker_color=COLORS['success'],
                hovertemplate='<b>%{x}</b><br>Umsatz: $%{y:,.2f}<extra></extra>'
            ))

            for i, (day, revenue) in enumerate(zip(df_fare_weekday['day_name'], df_fare_weekday['total_revenue'])):
                fig.add_annotation(
                    x=day,
                    y=revenue,
                    text=f"${revenue:,.0f}",
                    showarrow=False,
                    yshift=10,
                    font=dict(size=9, color='#333')
                )

        fig.update_layout(**create_chart_layout(
            'Gesamtumsatz nach Wochentag', 'Wochentag', 'Umsatz ($)'
        ))
        return fig

    @callback(Output('hourly-pickups-chart', 'figure'),
              Input('hourly-pickups-chart', 'id'))
    def update_hourly_chart(_):
        fig = go.Figure()

        if not df_hourly.empty:
            fig.add_trace(go.Scatter(
                x=df_hourly['hour'],
                y=df_hourly['trip_count'],
                mode='lines+markers',
                name='Fahrten',
                line=dict(color=COLORS['secondary'], width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor=f"rgba(255, 127, 14, 0.2)",
                hovertemplate='<b>%{x}:00</b><br>Fahrten: %{y:,.0f}<extra></extra>'
            ))

        fig.update_layout(**create_chart_layout(
            'Fahrten nach Tageszeit', 'Stunde', 'Anzahl Fahrten'
        ))
        fig.update_xaxes(tickmode='linear', dtick=2)
        return fig

    @callback(Output('monthly-trips-chart', 'figure'),
              Input('monthly-trips-chart', 'id'))
    def update_monthly_trips_chart(_):
        fig = go.Figure()

        if not df_monthly.empty:
            fig.add_trace(go.Bar(
                x=df_monthly['year_month'],
                y=df_monthly['trip_count'],
                name='Fahrten',
                marker_color=COLORS['danger'],
                hovertemplate='<b>%{x|%B %Y}</b><br>Fahrten: %{y:,.0f}<extra></extra>'
            ))

            for i, (date, count) in enumerate(zip(df_monthly['year_month'], df_monthly['trip_count'])):
                fig.add_annotation(
                    x=date,
                    y=count,
                    text=f"{count:,.0f}",
                    showarrow=False,
                    yshift=10,
                    font=dict(size=9, color='#333')
                )

        fig.update_layout(**create_chart_layout(
            'Fahrten pro Monat', 'Monat', 'Anzahl Fahrten'
        ))
        return fig

    @callback(Output('top-pickup-locations-chart', 'figure'),
              Input('top-pickup-locations-chart', 'id'))
    def update_top_pickup_chart(_):
        fig = go.Figure()

        if not df_top_pickup.empty:
            # NYC Taxi Zone Mapping (Top Zonen)
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

            # Sortiere und füge Zone-Namen hinzu
            df_sorted = df_top_pickup.copy()
            df_sorted['zone_name'] = df_sorted['pulocationid'].map(zone_names).fillna('Zone ' + df_sorted['pulocationid'].astype(str))
            df_sorted = df_sorted.sort_values('trip_count', ascending=True)

            fig.add_trace(go.Bar(
                y=df_sorted['zone_name'],
                x=df_sorted['trip_count'],
                orientation='h',
                name='Fahrten',
                marker_color=COLORS['primary'],
                text=[f"{count:,.0f}" for count in df_sorted['trip_count']],
                textposition='inside',
                textfont=dict(color='white', size=11),
                hovertemplate='<b>%{y}</b><br>Fahrten: %{x:,.0f}<extra></extra>'
            ))


        fig.update_layout(
            title='Top 10 Pickup-Zonen',
            xaxis_title='Anzahl Fahrten',
            yaxis_title='Zone',
            template='plotly_white',
            height=450,
            margin=dict(l=200, r=50, t=50, b=50)
        )
        return fig

    @callback(Output('vendor-distribution-chart', 'figure'),
              Input('vendor-distribution-chart', 'id'))
    def update_payment_type_chart(_):
        fig = go.Figure()

        if not df_payment.empty:
            # Payment type mapping basierend auf NYC TLC Daten
            payment_type_names = {
                1: 'Kreditkarte',
                2: 'Bargeld',
                3: 'Keine Gebühr',
                4: 'Streit',
                5: 'Unbekannt',
                6: 'Annullierte Fahrt'
            }
            df_payment_copy = df_payment.copy()
            df_payment_copy['payment_name'] = df_payment_copy['payment_type'].map(payment_type_names).fillna('Sonstige')

            fig.add_trace(go.Pie(
                labels=df_payment_copy['payment_name'],
                values=df_payment_copy['trip_count'],
                hole=0.3,
                textposition='inside',
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Fahrten: %{value:,.0f}<br>Anteil: %{percent}<extra></extra>'
            ))

        fig.update_layout(
            title='Zahlungsarten-Verteilung',
            template='plotly_white',
            height=450
        )
        return fig

    return app