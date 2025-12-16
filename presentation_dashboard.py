import dash
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd
from sqlalchemy import text

# uses your existing DB connector
from db_connection import DBConnection


DAY_NAMES = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']


class TaxiOpsPresentationDashboard:
    """
    Merged operations dashboard for taxi companies
    Section A: Time-based demand (from file 1)
    Section C: Weather-driven demand (from file 2)
    """

    def __init__(self, year_filter: int = 2025, db_name: str = "ny_taxi_dwh"):
        self.year_filter = year_filter
        self.db_name = db_name
        self._engine = None

    # -------------------- DB helpers --------------------
    def _get_db_engine(self):
        if self._engine is None:
            db = DBConnection("postgres", "password123", "localhost", 5433, self.db_name)
            self._engine = db.connect()
        return self._engine

    @staticmethod
    def _execute_query(query: str, engine) -> pd.DataFrame:
        if engine is None:
            return pd.DataFrame()
        try:
            return pd.read_sql(text(query), engine)
        except Exception as e:
            print(f"[PRESENTATION_DASHBOARD] Query error: {e}")
            return pd.DataFrame()

    # -------------------- data loading --------------------
    def load_data(self) -> dict[str, pd.DataFrame]:
        engine = self._get_db_engine()

        q_hourly = f"""
            SELECT dd.hour,
                   COUNT(*) as trip_count
            FROM fact_trips ft
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            WHERE dd.year >= {self.year_filter}
            GROUP BY dd.hour
            ORDER BY dd.hour
        """

        q_weekday = f"""
            SELECT EXTRACT(DOW FROM dd.full_datetime)::int as day_of_week,
                   COUNT(*) as trip_count
            FROM fact_trips ft
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            WHERE dd.year >= {self.year_filter}
            GROUP BY day_of_week
            ORDER BY day_of_week
        """

        # Weather conditions query from file 2
        q_weather_conditions = f"""
            SELECT 
                CASE 
                    WHEN LOWER(dw.conditions) IN ('partially cloudy', 'clear', 'overcast') 
                    THEN 'Good Weather'
                    WHEN LOWER(dw.conditions) IN ('rain, overcast', 'rain, partially cloudy', 'snow, rain, overcast')
                    THEN 'Bad Weather'
                    ELSE 'Other'
                END AS weather_group,
                COUNT(ft.datetime_id) AS total_trips,
                COUNT(DISTINCT dw.datetime_id) AS hours_with_condition,
                CASE 
                    WHEN COUNT(DISTINCT dw.datetime_id) > 0 
                    THEN COUNT(ft.datetime_id)::float / COUNT(DISTINCT dw.datetime_id)
                    ELSE 0 
                END AS avg_trips_per_hour
            FROM dim_weather dw
            LEFT JOIN fact_trips ft ON ft.datetime_id = dw.datetime_id
            JOIN dim_datetime dd ON dw.datetime_id = dd.datetime_id
            WHERE dd.year >= {self.year_filter} AND dw.conditions IS NOT NULL
            GROUP BY weather_group
            HAVING COUNT(DISTINCT dw.datetime_id) > 0
            ORDER BY avg_trips_per_hour DESC
        """

        q_temp_trips = f"""
            SELECT dw.temp,
                   COUNT(*) as trip_count
            FROM fact_trips ft
            JOIN dim_weather dw ON ft.datetime_id = dw.datetime_id
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            WHERE dd.year >= {self.year_filter}
              AND dw.temp IS NOT NULL
            GROUP BY dw.temp
            ORDER BY dw.temp
        """

        df_hourly = self._execute_query(q_hourly, engine)
        df_weekday = self._execute_query(q_weekday, engine)
        df_weather = self._execute_query(q_weather_conditions, engine)
        df_temp = self._execute_query(q_temp_trips, engine)

        # Enrichment
        if not df_weekday.empty:
            df_weekday["day_name"] = df_weekday["day_of_week"].astype(int).map(lambda x: DAY_NAMES[x])

        return {
            "hourly": df_hourly,
            "weekday": df_weekday,
            "weather_conditions": df_weather,
            "temp_trips": df_temp,
        }

    # -------------------- Section A figures (from file 1) --------------------
    def _fig_trips_by_hour(self, df_hourly: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if df_hourly.empty:
            fig.add_annotation(text="No data available", showarrow=False)
            return fig

        # Create color mapping based on demand zones
        colors = []
        for hour in df_hourly["hour"]:
            if 17 <= hour <= 19:  # Rush Hour (Peak)
                colors.append('#ffd700')  # Gold
            elif 7 <= hour <= 16:  # Growth Zone (includes 7am morning commute)
                colors.append('#667eea')  # Blue-Purple
            elif (20 <= hour <= 23) or (hour == 0):  # Evening/Nightlife (includes midnight)
                colors.append('#9b59b6')  # Purple
            elif 3 <= hour <= 6:  # Dead Zone (Lowest demand)
                colors.append('#95a5a6')  # Gray
            else:  # Night hours 1-2 (Post-nightlife)
                colors.append('#34495e')  # Dark Gray

        fig.add_trace(go.Bar(
            x=df_hourly["hour"],
            y=df_hourly["trip_count"],
            marker=dict(
                color=colors,
                line=dict(color='white', width=1)
            ),
            name="Trips",
            hovertemplate='Hour: %{x}:00<br>Trips: %{y:,}<extra></extra>'
        ))

        fig.update_layout(
            title="Trips by Hour of Day (Color-coded by Demand Zone)",
            xaxis_title="Hour of Day",
            yaxis_title="Number of Trips",
            margin=dict(l=40, r=20, t=60, b=40),
            showlegend=False
        )
        return fig

    def _fig_trips_by_weekday(self, df_weekday: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if df_weekday.empty:
            fig.add_annotation(text="No data available", showarrow=False)
            return fig

        # Color weekdays vs weekend
        colors = []
        for day in df_weekday["day_of_week"]:
            if day in [0, 6]:  # Sunday, Saturday
                colors.append('#f093fb')  # Pink/Purple (weekend)
            else:  # Weekdays
                colors.append('#667eea')  # Blue-Purple

        fig.add_trace(go.Bar(
            x=df_weekday["day_name"],
            y=df_weekday["trip_count"],
            marker=dict(
                color=colors,
                line=dict(color='white', width=1)
            ),
            name="Trips",
            hovertemplate='%{x}<br>Trips: %{y:,}<extra></extra>'
        ))

        fig.update_layout(
            title="Trips by Day of Week (Weekdays vs Weekend)",
            xaxis_title="Day of Week",
            yaxis_title="Number of Trips",
            margin=dict(l=40, r=20, t=60, b=40),
            showlegend=False
        )
        return fig

    # -------------------- Section C figures (from file 2) --------------------
    def _fig_weather_conditions(self, df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if df.empty:
            fig.add_annotation(text="No data available", showarrow=False)
            return fig

        df_plot = df[df['weather_group'] != 'Other'].copy()

        colors = []
        for group in df_plot["weather_group"]:
            if group == "Bad Weather":
                colors.append("#d32f2f")
            elif group == "Good Weather":
                colors.append("#388e3c")
            else:
                colors.append("#999999")

        fig.add_trace(go.Bar(
            x=df_plot["weather_group"],
            y=df_plot["avg_trips_per_hour"],
            marker_color=colors,
            text=df_plot["avg_trips_per_hour"].apply(lambda x: f"{int(x):,}"),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>' +
                         'Avg Trips/Hour: %{y:,.0f}<br>' +
                         'Hours: %{customdata}<br>' +
                         '<extra></extra>',
            customdata=df_plot["hours_with_condition"]
        ))

        fig.update_layout(
            title="Demand by Weather Category (Normalized)",
            xaxis_title="Weather Category",
            yaxis_title="Average Trips per Hour",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            showlegend=False,
            yaxis=dict(range=[0, df_plot["avg_trips_per_hour"].max() * 1.2])
        )
        return fig

    def _fig_temp_vs_trips(self, df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if df.empty:
            fig.add_annotation(text="No data available", showarrow=False)
            return fig

        fig.add_trace(go.Scatter(
            x=df["temp"],
            y=df["trip_count"],
            mode="markers",
            marker=dict(
                size=7,
                color=df["temp"],
                colorscale="RdBu",
                showscale=True,
                colorbar=dict(title="Temperature")
            )
        ))

        fig.update_layout(
            title="Temperature Impact on Trip Volume",
            xaxis_title="Temperature",
            yaxis_title="Number of Trips",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff"
        )
        return fig

    # -------------------- KPIs --------------------
    @staticmethod
    def _kpi_peak_hours(df_hourly: pd.DataFrame) -> str:
        if df_hourly.empty:
            return "n/a"
        # top 3 hours
        top = df_hourly.sort_values("trip_count", ascending=False).head(3)["hour"].tolist()
        top = sorted([int(x) for x in top])
        return ", ".join([f"{h:02d}:00" for h in top])

    # -------------------- app layout --------------------
    def create_app(self) -> dash.Dash:
        app = dash.Dash(__name__)

        data = self.load_data()
        df_hourly = data["hourly"]
        df_weekday = data["weekday"]
        df_weather = data["weather_conditions"]
        df_temp = data["temp_trips"]

        peak_hours = self._kpi_peak_hours(df_hourly)

        fig_hour = self._fig_trips_by_hour(df_hourly)
        fig_weekday = self._fig_trips_by_weekday(df_weekday)
        fig_weather = self._fig_weather_conditions(df_weather)
        fig_temp = self._fig_temp_vs_trips(df_temp)

        def kpi_card(title: str, value: str):
            return html.Div(
                style={
                    "flex": "1",
                    "background": "#f5f5f5",
                    "borderRadius": "10px",
                    "padding": "14px 16px",
                    "textAlign": "center",
                },
                children=[
                    html.Div(title, style={"fontSize": "12px", "opacity": "0.75"}),
                    html.Div(value, style={"fontSize": "22px", "fontWeight": "700", "marginTop": "6px"}),
                ],
            )

        app.layout = html.Div(
            style={"fontFamily": "Arial, sans-serif", "padding": "18px"},
            children=[
                html.H1("NYC Yellow Taxi – Fleet Optimization Dashboard", style={"textAlign": "center", "marginBottom": "10px"}),
                html.P(
                    "Objective: Reduce idle time and maximize fleet utilization through data-driven shift planning "
                    "based on temporal demand patterns and weather conditions.",
                    style={"textAlign": "center", "maxWidth": "900px", "margin": "0 auto 6px auto", "fontSize": "16px"}
                ),
                html.P(
                    "Data-driven insights: Demand variation between peak (18:00) and low (04:00) hours",
                    style={"textAlign": "center", "maxWidth": "900px", "margin": "0 auto 10px auto",
                           "fontSize": "14px", "color": "#666", "fontStyle": "italic"}
                ),

                # KPI Row
                html.Div(
                    style={"display": "flex", "gap": "12px", "marginTop": "14px"},
                    children=[
                        kpi_card("Peak Hours (Top 3)", peak_hours),
                    ],
                ),

                # ---- Section A: Time (from file 1) ----
                html.H2("Time-based Demand (Operations Scheduling)", style={"marginTop": "22px"}),

                html.Div(
                    style={"background": "#ffffff",
                           "borderRadius": "12px", "padding": "16px", "border": "2px solid #e0e0e0"},
                    children=[
                        html.Div(style={"fontSize": "20px", "fontWeight": "bold", "marginBottom": "12px", "color": "#333"},
                                children=["Demand Variation: Match Fleet to Hourly Demand"]),

                        html.Div(style={"display": "flex", "gap": "12px", "marginTop": "12px"}, children=[
                            html.Div(style={"flex": "1", "background": "#fffaeb",
                                           "borderRadius": "8px", "padding": "12px", "border": "2px solid #ffd700"}, children=[
                                html.Div("RUSH HOUR", style={"fontSize": "14px", "fontWeight": "bold", "marginBottom": "6px", "color": "#333"}),
                                html.Div("17:00-19:00", style={"fontSize": "24px", "fontWeight": "bold", "color": "#333"}),
                                html.Div("496k trips/hr", style={"fontSize": "14px", "opacity": "0.7", "color": "#333"}),
                                html.Div("Deploy 100%", style={"fontSize": "12px", "marginTop": "4px", "background": "#ffd700",
                                        "color": "#333", "padding": "2px 6px", "borderRadius": "4px", "display": "inline-block"}),
                            ]),

                            html.Div(style={"flex": "1", "background": "#f0f4ff",
                                           "borderRadius": "8px", "padding": "12px", "border": "1px solid #667eea"}, children=[
                                html.Div("GROWTH ZONE", style={"fontSize": "14px", "fontWeight": "bold", "marginBottom": "6px", "color": "#333"}),
                                html.Div("07:00-16:00", style={"fontSize": "24px", "fontWeight": "bold", "color": "#333"}),
                                html.Div("+177% growth", style={"fontSize": "14px", "opacity": "0.7", "color": "#333"}),
                                html.Div("Gradual scaling", style={"fontSize": "12px", "marginTop": "4px", "background": "#e8eeff",
                                        "color": "#333", "padding": "2px 6px", "borderRadius": "4px", "display": "inline-block"}),
                            ]),

                            html.Div(style={"flex": "1", "background": "#f9f0ff",
                                           "borderRadius": "8px", "padding": "12px", "border": "1px solid #9b59b6"}, children=[
                                html.Div("NIGHTLIFE", style={"fontSize": "14px", "fontWeight": "bold", "marginBottom": "6px", "color": "#333"}),
                                html.Div("20:00-00:00", style={"fontSize": "24px", "fontWeight": "bold", "color": "#333"}),
                                html.Div("~400k trips/hr", style={"fontSize": "14px", "opacity": "0.7", "color": "#333"}),
                                html.Div("High demand", style={"fontSize": "12px", "marginTop": "4px", "background": "#f0e6ff",
                                        "color": "#333", "padding": "2px 6px", "borderRadius": "4px", "display": "inline-block"}),
                            ]),

                            html.Div(style={"flex": "1", "background": "#f5f5f5",
                                           "borderRadius": "8px", "padding": "12px", "border": "1px solid #95a5a6"}, children=[
                                html.Div("DEAD ZONE", style={"fontSize": "14px", "fontWeight": "bold", "marginBottom": "6px", "color": "#333"}),
                                html.Div("01:00-06:00", style={"fontSize": "24px", "fontWeight": "bold", "color": "#333"}),
                                html.Div("~250k trips total", style={"fontSize": "14px", "opacity": "0.7", "color": "#333"}),
                                html.Div("Minimal fleet", style={"fontSize": "12px", "marginTop": "4px", "background": "#e8e8e8",
                                        "color": "#333", "padding": "2px 6px", "borderRadius": "4px", "display": "inline-block"}),
                            ]),
                        ]),

                        html.Div(style={"marginTop": "14px", "padding": "10px", "background": "#fffaeb",
                                       "borderRadius": "6px", "borderLeft": "4px solid #ffd700"}, children=[
                            html.Div("Actionable Strategy:", style={"fontWeight": "bold", "marginBottom": "6px", "color": "#333"}),
                            html.Div("Demand varies between peak (496k trips at 6pm) and lowest demand (1-6am with ~250k total trips). " +
                                    "Scale fleet accordingly: fewer vehicles during low-demand hours reduces operational costs, " +
                                    "more vehicles during peak hours maximizes revenue capture. Strategic shift planning improves driver satisfaction.",
                                    style={"fontSize": "14px", "lineHeight": "1.4", "color": "#333"}),
                        ]),
                    ],
                ),

                html.Div(
                    style={"display": "flex", "gap": "12px", "marginTop": "12px"},
                    children=[
                        html.Div(style={"flex": "1"}, children=[dcc.Graph(figure=fig_hour)]),
                        html.Div(style={"flex": "1"}, children=[dcc.Graph(figure=fig_weekday)]),
                    ],
                ),

                # ---- Section C: Weather (from file 2) ----
                html.H2("Weather-Driven Demand (Dynamic Planning)", style={"marginTop": "32px"}),

                html.Div(
                    style={
                        "background": "#ffffff",
                        "borderRadius": "12px",
                        "padding": "16px",
                        "border": "2px solid #e0e0e0",
                        "marginBottom": "16px"
                    },
                    children=[
                        html.Div(style={"fontSize": "18px", "fontWeight": "bold", "marginBottom": "12px", "color": "#333"},
                                children=["Weather Impact on Fleet Planning"]),

                        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}, children=[
                            html.Div(style={"background": "#e8f5e9", "borderRadius": "8px", "padding": "12px"}, children=[
                                html.Div("Good Weather", style={"fontSize": "16px", "fontWeight": "bold", "color": "#2d7a2d"}),
                                html.Div("Standard fleet deployment", style={"fontSize": "14px", "color": "#333"}),
                            ]),

                            html.Div(style={"background": "#fff3e0", "borderRadius": "8px", "padding": "12px", "border": "1px solid #ff9800"}, children=[
                                html.Div("Bad Weather", style={"fontSize": "16px", "fontWeight": "bold", "color": "#e65100"}),
                                html.Div("Increase +15% (more taxi demand)", style={"fontSize": "14px", "color": "#333"}),
                            ]),
                        ]),

                        html.Div(style={"marginTop": "12px", "padding": "10px", "background": "#fff3e0",
                                       "borderRadius": "6px", "borderLeft": "4px solid #ff9800"}, children=[
                            html.Div("Strategy:", style={"fontWeight": "bold", "marginBottom": "4px", "color": "#e65100"}),
                            html.Div("Rain/snow increases taxi demand as people avoid walking/cycling. " +
                                    "Use weather forecasts to pre-position extra vehicles in core areas before bad weather hits. " +
                                    "Focus: Manhattan, transport hubs, business districts.",
                                    style={"fontSize": "14px", "color": "#333", "lineHeight": "1.4"}),
                        ]),
                    ],
                ),

                html.Div(
                    style={"display": "flex", "gap": "16px", "marginTop": "16px"},
                    children=[
                        html.Div(style={"flex": "1"}, children=[
                            dcc.Graph(figure=fig_weather)
                        ]),
                        html.Div(style={"flex": "1"}, children=[
                            dcc.Graph(figure=fig_temp)
                        ]),
                    ],
                ),

                html.Div(style={"height": "18px"}),
                html.P("Data Source: NYC Yellow Cab + Weather | Year Filter: "
                       f"{self.year_filter}", style={"opacity": "0.6", "textAlign": "center"}),
            ]
        )

        return app