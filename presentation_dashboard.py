import dash
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd
from sqlalchemy import text

# uses your existing DB connector (as in dashboard.py)
from db_connection import DBConnection


DAY_NAMES = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']


class TaxiOpsPresentationDashboard:
    """
    Presentation dashboard for taxi companies (5-minute pitch)
    Focus: (A) Time-based demand (hour/day of week) + (C) Weather → demand
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

        # Weather: trips per condition (via dim_weather)
        q_weather_conditions = f"""
            SELECT dw.conditions,
                   COUNT(ft.datetime_id) as trip_count
            FROM dim_weather dw
            LEFT JOIN fact_trips ft ON ft.datetime_id = dw.datetime_id
            JOIN dim_datetime dd ON dw.datetime_id = dd.datetime_id
            WHERE dd.year >= {self.year_filter}
            GROUP BY dw.conditions
            ORDER BY trip_count DESC
        """

        # Temperature vs trips (compact, very intuitive for business)
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

    # -------------------- figures --------------------
    def _fig_trips_by_hour(self, df_hourly: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if df_hourly.empty:
            fig.add_annotation(text="No data available", showarrow=False)
            return fig

        fig.add_trace(go.Scatter(
            x=df_hourly["hour"],
            y=df_hourly["trip_count"],
            mode="lines+markers",
            name="Trips"
        ))
        fig.update_layout(
            title="Trips by Hour of Day (Demand Peaks)",
            xaxis_title="Hour of Day",
            yaxis_title="Number of Trips",
            margin=dict(l=40, r=20, t=60, b=40),
        )
        return fig

    def _fig_trips_by_weekday(self, df_weekday: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if df_weekday.empty:
            fig.add_annotation(text="No data available", showarrow=False)
            return fig

        fig.add_trace(go.Bar(
            x=df_weekday["day_name"],
            y=df_weekday["trip_count"],
            name="Trips"
        ))
        fig.update_layout(
            title="Trips by Day of Week",
            xaxis_title="Day of Week",
            yaxis_title="Number of Trips",
            margin=dict(l=40, r=20, t=60, b=40),
        )
        return fig

    def _fig_weather_conditions(self, df_weather: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if df_weather.empty:
            fig.add_annotation(text="No data available", showarrow=False)
            return fig

        # Top N to keep it pitch-ready
        top_n = 8
        df_plot = df_weather.head(top_n).copy()

        fig.add_trace(go.Bar(
            x=df_plot["conditions"],
            y=df_plot["trip_count"],
            name="Trips"
        ))
        fig.update_layout(
            title="Trips by Weather Conditions (Top Categories)",
            xaxis_title="Weather Condition",
            yaxis_title="Number of Trips",
            margin=dict(l=40, r=20, t=60, b=80),
        )
        return fig

    def _fig_temp_vs_trips(self, df_temp: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if df_temp.empty:
            fig.add_annotation(text="No data available", showarrow=False)
            return fig

        fig.add_trace(go.Scatter(
            x=df_temp["temp"],
            y=df_temp["trip_count"],
            mode="markers",
            name="Trips"
        ))
        fig.update_layout(
            title="Temperature vs Trip Count",
            xaxis_title="Temperature",
            yaxis_title="Trips",
            margin=dict(l=40, r=20, t=60, b=40),
        )
        return fig

    # -------------------- KPIs / insights --------------------
    @staticmethod
    def _kpi_peak_hours(df_hourly: pd.DataFrame) -> str:
        if df_hourly.empty:
            return "n/a"
        # top 3 hours
        top = df_hourly.sort_values("trip_count", ascending=False).head(3)["hour"].tolist()
        top = sorted([int(x) for x in top])
        return ", ".join([f"{h:02d}:00" for h in top])

    @staticmethod
    def _kpi_weather_split(df_weather: pd.DataFrame) -> dict[str, str]:
        """
        Very simple business categorization:
        - 'bad' if condition contains Rain or Snow
        - otherwise 'good'
        """
        if df_weather.empty:
            return {"good_pct": "n/a", "bad_pct": "n/a"}

        df = df_weather.copy()
        df["bucket"] = df["conditions"].fillna("Unknown").astype(str).apply(
            lambda s: "bad" if ("Rain" in s or "Snow" in s) else "good"
        )
        total = df["trip_count"].sum()
        if total <= 0:
            return {"good_pct": "n/a", "bad_pct": "n/a"}

        good = df.loc[df["bucket"] == "good", "trip_count"].sum()
        bad = df.loc[df["bucket"] == "bad", "trip_count"].sum()

        return {
            "good_pct": f"{(good / total) * 100:.1f}%",
            "bad_pct": f"{(bad / total) * 100:.1f}%"
        }

    # -------------------- app layout --------------------
    def create_app(self) -> dash.Dash:
        app = dash.Dash(__name__)

        data = self.load_data()
        df_hourly = data["hourly"]
        df_weekday = data["weekday"]
        df_weather = data["weather_conditions"]
        df_temp = data["temp_trips"]

        peak_hours = self._kpi_peak_hours(df_hourly)
        weather_split = self._kpi_weather_split(df_weather)

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
                html.H1("NYC Taxi Ops – Insights Dashboard", style={"textAlign": "center"}),
                html.P(
                    "Goal: Fewer empty trips, higher utilization – through better shift and fleet planning "
                    "(time + weather as control variables).",
                    style={"textAlign": "center", "maxWidth": "900px", "margin": "0 auto 10px auto"}
                ),

                # KPI Row
                html.Div(
                    style={"display": "flex", "gap": "12px", "marginTop": "14px"},
                    children=[
                        kpi_card("Peak Hours (Top 3)", peak_hours),
                        kpi_card("Trips in 'Good' Weather", weather_split["good_pct"]),
                        kpi_card("Trips in 'Bad' Weather (Rain/Snow)", weather_split["bad_pct"]),
                    ],
                ),

                # ---- Section A: Time ----
                html.H2("A) Time-based Demand (Operations Scheduling)", style={"marginTop": "22px"}),

                html.Div(
                    style={"background": "#ffffff", "borderRadius": "12px", "padding": "12px 14px"},
                    children=[
                        html.P(
                            "Finding: Demand is not evenly distributed. "
                            "There are clear peaks in the morning (commuters) and in the evening (after work).",
                            style={"marginBottom": "8px"}
                        ),
                        html.Ul([
                            html.Li("More vehicles/shifts between approx. 07–10 and 16–19."),
                            html.Li("Reduced fleet at night (approx. 02–05) saves costs and empty trips."),
                            html.Li("Control via dispatching and shift planning (forecasting optional)."),
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

                # ---- Section C: Weather ----
                html.H2("C) Weather-driven Demand (Dynamic Planning)", style={"marginTop": "22px"}),

                html.Div(
                    style={"background": "#ffffff", "borderRadius": "12px", "padding": "12px 14px"},
                    children=[
                        html.P(
                            "Finding: Weather is an external driver – clearly visible in trips per weather condition. "
                            "This enables dynamic planning (e.g. the day before or in the morning).",
                            style={"marginBottom": "8px"}
                        ),
                        html.Ul([
                            html.Li("Most trips occur during clear or partially cloudy conditions."),
                            html.Li("Demand drops significantly during rain/snow combinations."),
                            html.Li("Recommendation: Integrate weather forecasts into daily fleet decisions."),
                        ]),
                    ],
                ),

                html.Div(
                    style={"display": "flex", "gap": "12px", "marginTop": "12px"},
                    children=[
                        html.Div(style={"flex": "1"}, children=[dcc.Graph(figure=fig_weather)]),
                        html.Div(style={"flex": "1"}, children=[dcc.Graph(figure=fig_temp)]),
                    ],
                ),

                # ---- Final Recommendation ----
                html.H2("Actionable Recommendations (What we would do tomorrow)", style={"marginTop": "22px"}),
                html.Div(
                    style={"background": "#f5f5f5", "borderRadius": "12px", "padding": "12px 14px"},
                    children=[
                        html.Ol([
                            html.Li("Align shift planning with demand peaks (07–10, 16–19)."),
                            html.Li("Reduce night fleet (02–05) or deploy selectively (airport/hotspots)."),
                            html.Li("Use weather forecasts as a trigger: good weather → more vehicles, bad weather → focus on core zones."),
                            html.Li("Optional next step: forecasting model for trips per hour as decision support."),
                        ])
                    ],
                ),

                html.Div(style={"height": "18px"}),
                html.P(
                    "Data Source: NYC Yellow Cab + Weather | Year Filter: "
                    f"{self.year_filter}",
                    style={"opacity": "0.6", "textAlign": "center"}
                ),
            ]
        )

        return app
