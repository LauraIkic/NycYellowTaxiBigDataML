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
    Executive operations dashboard for taxi companies
    Focus:
    (A) Time-based demand
    (C) Weather-driven demand as a planning lever
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
                   COUNT(*) AS trip_count
            FROM fact_trips ft
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            WHERE dd.year >= {self.year_filter}
            GROUP BY dd.hour
            ORDER BY dd.hour
        """

        q_weekday = f"""
            SELECT EXTRACT(DOW FROM dd.full_datetime)::int AS day_of_week,
                   COUNT(*) AS trip_count
            FROM fact_trips ft
            JOIN dim_datetime dd ON ft.datetime_id = dd.datetime_id
            WHERE dd.year >= {self.year_filter}
            GROUP BY day_of_week
            ORDER BY day_of_week
        """

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
                   COUNT(*) AS trip_count
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

        if not df_weekday.empty:
            df_weekday["day_name"] = df_weekday["day_of_week"].map(lambda x: DAY_NAMES[int(x)])

        return {
            "hourly": df_hourly,
            "weekday": df_weekday,
            "weather_conditions": df_weather,
            "temp_trips": df_temp,
        }

    # -------------------- figures --------------------
    def _fig_trips_by_hour(self, df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if df.empty:
            fig.add_annotation(text="No data available", showarrow=False)
            return fig

        fig.add_trace(go.Scatter(
            x=df["hour"],
            y=df["trip_count"],
            mode="lines+markers"
        ))

        fig.update_layout(
            title="Trips by Hour of Day",
            xaxis_title="Hour of Day",
            yaxis_title="Number of Trips",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff"
        )
        return fig

    def _fig_trips_by_weekday(self, df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if df.empty:
            fig.add_annotation(text="No data available", showarrow=False)
            return fig

        fig.add_trace(go.Bar(
            x=df["day_name"],
            y=df["trip_count"],
            marker_color="#1976d2"
        ))

        fig.update_layout(
            title="Trips by Day of Week",
            xaxis_title="Day of Week",
            yaxis_title="Number of Trips",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff"
        )
        return fig

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
    def _kpi_peak_hours(df: pd.DataFrame) -> str:
        if df.empty:
            return "n/a"
        hours = df.sort_values("trip_count", ascending=False).head(3)["hour"]
        return ", ".join([f"{int(h):02d}:00" for h in sorted(hours)])

    # -------------------- UI helpers --------------------
    @staticmethod
    def _kpi_card(title: str, value: str):
        return html.Div(
            style={
                "flex": "1",
                "background": "#f5f5f5",
                "borderRadius": "10px",
                "padding": "14px",
                "textAlign": "center",
            },
            children=[
                html.Div(title, style={"fontSize": "12px", "opacity": "0.7"}),
                html.Div(value, style={"fontSize": "22px", "fontWeight": "700"}),
            ],
        )

    # -------------------- app layout --------------------
    def create_app(self) -> dash.Dash:
        app = dash.Dash(__name__)
        data = self.load_data()

        app.layout = html.Div(
            style={"fontFamily": "Arial, sans-serif", "padding": "20px"},
            children=[
                html.H1("NYC Taxi Operations – Executive Insights", style={"textAlign": "center"}),

                # KPI row
                html.Div(
                    style={"display": "flex", "gap": "12px", "marginTop": "14px"},
                    children=[
                        self._kpi_card(
                            "Peak Demand Hours",
                            self._kpi_peak_hours(data["hourly"])
                        ),
                    ],
                ),

                # -------- Section A --------
                html.H2("A) Time-Based Demand", style={"marginTop": "26px"}),

                html.Div(
                    style={"display": "flex", "gap": "16px"},
                    children=[
                        html.Div(style={"flex": "1"}, children=[
                            dcc.Graph(figure=self._fig_trips_by_hour(data["hourly"]))
                        ]),
                        html.Div(style={"flex": "1"}, children=[
                            dcc.Graph(figure=self._fig_trips_by_weekday(data["weekday"]))
                        ]),
                    ],
                ),

                # -------- Section C --------
                html.H2("Weather-Driven Demand", style={"marginTop": "32px"}),

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
                            dcc.Graph(figure=self._fig_weather_conditions(data["weather_conditions"]))
                        ]),
                        html.Div(style={"flex": "1"}, children=[
                            dcc.Graph(figure=self._fig_temp_vs_trips(data["temp_trips"]))
                        ]),
                    ],
                ),
            ],
        )

        return app
