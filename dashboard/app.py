"""
AdaptAction Climate Resilience Dashboard
=========================================
Interactive Dash application for visualizing territorial CRI scores,
trajectories, and degradation alerts.

Designed for Expertise France programme teams, GIZ country offices,
and Enabel climate focal points.

Run locally:
    python dashboard/app.py

Access: http://localhost:8050
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from datetime import datetime

from climate_resilience import ClimateResilienceIndex, ClimateDataLoader, DegradationAlertSystem
from climate_resilience.alerts import AlertLevel

# ---------------------------------------------------------------------------
# Bootstrap + Fonts
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap",
    ],
    title="AdaptAction CRI Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# ---------------------------------------------------------------------------
# Color palette (humanitarian + climate)
# ---------------------------------------------------------------------------
COLORS = {
    "critical":  "#d63031",
    "degrading": "#e17055",
    "watch":     "#fdcb6e",
    "normal":    "#00b894",
    "improving": "#0984e3",
    "bg_dark":   "#1a1a2e",
    "bg_card":   "#16213e",
    "accent":    "#0f3460",
    "text":      "#e0e0e0",
    "header":    "#e94560",
}

PILLAR_COLORS = {
    "exposure":     "#d63031",
    "sensitivity":  "#e17055",
    "adaptive_cap": "#6c5ce7",
    "livelihood":   "#00cec9",
    "ecosystem":    "#55efc4",
}

COUNTRIES_COVERED = ["Niger", "Mali", "Burkina Faso", "Senegal", "Mauritania",
                     "Chad", "Central African Republic", "Cameroon"]

# ---------------------------------------------------------------------------
# Data initialization
# ---------------------------------------------------------------------------
def load_dashboard_data():
    loader = ClimateDataLoader(offline_mode=True)
    raw_df = loader.generate_synthetic_dataset(n_zones=60, n_periods=8, seed=42)
    normalized_df = loader.normalize_indicators(raw_df)

    latest_df = loader.get_latest_period(normalized_df)

    cri_engine = ClimateResilienceIndex(context="sahel")
    results = cri_engine.compute_batch(latest_df, period="latest")

    # Merger uniquement latitude/longitude — les autres colonnes sont déjà dans results
    coords = latest_df[["zone_id", "latitude", "longitude"]].drop_duplicates("zone_id")
    full_df = results.merge(coords, on="zone_id", how="left")

    alert_system = DegradationAlertSystem()
    alerts = alert_system.evaluate(full_df)
    alert_df = alert_system.get_summary(alerts)
    stats = alert_system.get_statistics(alerts)

    return full_df, normalized_df, alert_df, stats, cri_engine


FULL_DF, RAW_DF, ALERT_DF, ALERT_STATS, CRI_ENGINE = load_dashboard_data()

# ---------------------------------------------------------------------------
# Helper: Alert badge
# ---------------------------------------------------------------------------
def alert_badge(level: str) -> dbc.Badge:
    color_map = {
        "CRITICAL": "danger", "DEGRADING": "warning",
        "WATCH": "warning", "NORMAL": "success", "IMPROVING": "info",
    }
    return dbc.Badge(level, color=color_map.get(level, "secondary"), className="ms-1")


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
def make_kpi_card(title, value, subtitle, icon, color):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(icon, style={"fontSize": "2rem"}),
                html.Div([
                    html.H3(str(value), style={"color": color, "margin": 0, "fontWeight": 700}),
                    html.P(title, style={"color": "#aaa", "margin": 0, "fontSize": "0.85rem"}),
                    html.Small(subtitle, style={"color": "#666"}),
                ], style={"marginLeft": "12px"}),
            ], style={"display": "flex", "alignItems": "center"}),
        ])
    ], style={"backgroundColor": COLORS["bg_card"], "border": f"1px solid {color}30"})


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = dbc.Container([

    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("🌡️ Climate Resilience Index", style={
                    "color": COLORS["header"], "fontWeight": 700, "marginBottom": 0,
                }),
                html.P("AdaptAction Programme — Territorialized Monitoring Dashboard",
                       style={"color": "#aaa", "fontSize": "0.95rem"}),
                html.Small(
                    f"West & Central Africa | {len(FULL_DF['zone_id'].unique())} zones | "
                    f"Updated: {datetime.utcnow().strftime('%Y-%m-%d')}",
                    style={"color": "#666"}
                ),
            ]),
        ], md=8),
        dbc.Col([
            html.Div([
                dbc.Badge(f"🔴 {ALERT_STATS.get('n_critical_zones', 0)} CRITICAL",
                          color="danger", className="me-2 p-2"),
                dbc.Badge(f"🟠 {ALERT_STATS.get('n_degrading_zones', 0)} DEGRADING",
                          color="warning", className="me-2 p-2"),
                dbc.Badge(f"📡 LIVE", color="success", className="p-2"),
            ], style={"textAlign": "right", "paddingTop": "1rem"}),
        ], md=4),
    ], className="mb-4 mt-3"),

    # KPI Row
    dbc.Row([
        dbc.Col(make_kpi_card(
            "Avg. CRI Score", f"{FULL_DF['cri_score'].mean():.1f}/100",
            "Regional average", "📊", COLORS["normal"]
        ), md=3),
        dbc.Col(make_kpi_card(
            "Critical Zones", ALERT_STATS.get("n_critical_zones", 0),
            "CRI < 25 — urgent action", "🔴", COLORS["critical"]
        ), md=3),
        dbc.Col(make_kpi_card(
            "Zones Monitored", len(FULL_DF["zone_id"].unique()),
            f"Across {len(FULL_DF['country'].unique())} countries", "🗺️", COLORS["improving"]
        ), md=3),
        dbc.Col(make_kpi_card(
            "Watch or Above", f"{ALERT_STATS.get('pct_watch_or_above', 0):.0f}%",
            "Zones needing attention", "⚠️", COLORS["watch"]
        ), md=3),
    ], className="mb-4"),

    # Filters
    dbc.Row([
        dbc.Col([
            html.Label("Country", style={"color": "#aaa", "fontSize": "0.85rem"}),
            dcc.Dropdown(
                id="country-filter",
                options=[{"label": "All Countries", "value": "ALL"}] +
                        [{"label": c, "value": c} for c in sorted(FULL_DF["country"].unique())],
                value="ALL",
                clearable=False,
                style={"backgroundColor": COLORS["bg_card"]},
            ),
        ], md=4),
        dbc.Col([
            html.Label("Alert Level", style={"color": "#aaa", "fontSize": "0.85rem"}),
            dcc.Dropdown(
                id="alert-filter",
                options=[
                    {"label": "All Levels", "value": "ALL"},
                    {"label": "🔴 Critical", "value": "CRITICAL"},
                    {"label": "🟠 Degrading", "value": "DEGRADING"},
                    {"label": "🟡 Watch", "value": "WATCH"},
                    {"label": "🟢 Normal", "value": "NORMAL"},
                    {"label": "🔵 Improving", "value": "IMPROVING"},
                ],
                value="ALL",
                clearable=False,
            ),
        ], md=4),
        dbc.Col([
            html.Label("Period", style={"color": "#aaa", "fontSize": "0.85rem"}),
            dcc.Dropdown(
                id="period-filter",
                options=[{"label": "Latest", "value": "latest"}],
                value="latest",
                clearable=False,
            ),
        ], md=4),
    ], className="mb-4"),

    # Map + Alert table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🗺️ Territorial CRI Map", style={"fontWeight": 600}),
                dbc.CardBody([
                    dcc.Graph(id="map-chart", style={"height": "450px"}),
                ])
            ], style={"backgroundColor": COLORS["bg_card"]}),
        ], md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🚨 Active Alerts", style={"fontWeight": 600}),
                dbc.CardBody([
                    html.Div(id="alert-table", style={"overflowY": "auto", "maxHeight": "400px"}),
                ]),
            ], style={"backgroundColor": COLORS["bg_card"]}),
        ], md=4),
    ], className="mb-4"),

    # Pillar radar + Trajectory
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🕷️ Pillar Analysis — Radar View", style={"fontWeight": 600}),
                dbc.CardBody([
                    dcc.Graph(id="radar-chart", style={"height": "350px"}),
                ]),
            ], style={"backgroundColor": COLORS["bg_card"]}),
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📈 CRI Trajectories by Country", style={"fontWeight": 600}),
                dbc.CardBody([
                    dcc.Graph(id="trajectory-chart", style={"height": "350px"}),
                ]),
            ], style={"backgroundColor": COLORS["bg_card"]}),
        ], md=6),
    ], className="mb-4"),

    # Distribution
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 CRI Score Distribution", style={"fontWeight": 600}),
                dbc.CardBody([
                    dcc.Graph(id="distribution-chart", style={"height": "300px"}),
                ]),
            ], style={"backgroundColor": COLORS["bg_card"]}),
        ], md=12),
    ], className="mb-4"),

    # Footer
    html.Hr(style={"borderColor": "#333"}),
    html.P([
        "AdaptAction Climate Resilience Index | ",
        html.A("GitHub", href="https://github.com/your-username/Climate-Resilience-Index",
               style={"color": COLORS["header"]}),
        " | Expertise France · GIZ · Enabel | Data: NASA POWER, ERA5, CHIRPS",
    ], style={"textAlign": "center", "color": "#555", "fontSize": "0.85rem"}),

], fluid=True, style={"backgroundColor": COLORS["bg_dark"], "minHeight": "100vh",
                      "fontFamily": "Inter, sans-serif", "color": COLORS["text"]})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("map-chart", "figure"),
    [Input("country-filter", "value"), Input("alert-filter", "value")],
)
def update_map(country, alert_level):
    try:
        df = FULL_DF.copy()
        if country != "ALL":
            df = df[df["country"] == country]
        if not ALERT_DF.empty and alert_level != "ALL":
            alert_zones = ALERT_DF[ALERT_DF["level"] == alert_level]["zone_id"].tolist()
            df = df[df["zone_id"].isin(alert_zones)]
        if df.empty:
            return go.Figure()

        print("DEBUG - shape:", df.shape)
        print("DEBUG - cri_score nulls:", df["cri_score"].isna().sum())
        print("DEBUG - lat nulls:", df["latitude"].isna().sum())
        print("DEBUG - sample:", df[["zone_name","cri_score","latitude","longitude"]].head(2).to_string())

        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            lat=df["latitude"],
            lon=df["longitude"],
            mode="markers",
            marker=dict(
                size=12,
                color=df["cri_score"],
                colorscale=[
                    [0.0, "#d63031"], [0.25, "#e17055"],
                    [0.4,  "#fdcb6e"], [0.65, "#00b894"],
                    [1.0,  "#0984e3"]
                ],
                cmin=0, cmax=100,
                colorbar=dict(
    title=dict(text="CRI Score", font=dict(color="white")),
    tickfont=dict(color="white"),
),
                showscale=True,
            ),
            text=df["zone_name"] + "<br>Country: " + df["country"] +
                 "<br>CRI: " + df["cri_score"].round(1).astype(str),
            hoverinfo="text",
        ))
        fig.update_layout(
            geo=dict(
                scope="africa",
                bgcolor=COLORS["bg_dark"],
                landcolor="#2d3436",
                showland=True,
                showcountries=True,
                countrycolor="#636e72",
                showocean=True,
                oceancolor="#1a1a2e",
                projection_type="natural earth",
            ),
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            paper_bgcolor=COLORS["bg_card"],
        )
        return fig
    except Exception as e:
        print("MAP ERROR:", e)
        import traceback
        traceback.print_exc()
        return go.Figure()
@app.callback(
    Output("alert-table", "children"),
    Input("country-filter", "value"),
)
def update_alert_table(country):
    df = ALERT_DF.copy() if not ALERT_DF.empty else pd.DataFrame()
    if df.empty:
        return html.P("No alerts", style={"color": "#666"})

    if country != "ALL":
        df = df[df["country"] == country]

    df = df[df["level"].isin(["CRITICAL", "DEGRADING", "WATCH"])].head(12)

    rows = []
    for _, row in df.iterrows():
        icon = {"CRITICAL": "🔴", "DEGRADING": "🟠", "WATCH": "🟡"}.get(row["level"], "🟢")
        rows.append(html.Div([
            html.Div([
                html.Span(icon, style={"marginRight": "6px"}),
                html.Strong(row.get("zone_name", row["zone_id"]), style={"fontSize": "0.85rem"}),
            ]),
            html.Div([
                html.Span(f"CRI: {row['cri_score']:.1f}", style={"color": "#aaa", "fontSize": "0.78rem"}),
                html.Span(f" | {row['country']}", style={"color": "#666", "fontSize": "0.78rem"}),
            ]),
        ], style={
            "padding": "8px", "borderBottom": "1px solid #333",
            "borderLeft": "3px solid " + {'CRITICAL': '#d63031', 'DEGRADING': '#e17055', 'WATCH': '#fdcb6e'}.get(row['level'], '#00b894'),
            "marginBottom": "4px",
        }))

    return rows if rows else html.P("No alerts for selected filters", style={"color": "#666"})


@app.callback(
    Output("radar-chart", "figure"),
    Input("country-filter", "value"),
)
def update_radar(country):
    df = FULL_DF.copy()
    if country != "ALL":
        df = df[df["country"] == country]

    pillar_cols = ["pillar_exposure", "pillar_sensitivity", "pillar_adaptive_cap",
                   "pillar_livelihood", "pillar_ecosystem"]
    available = [c for c in pillar_cols if c in df.columns]

    if not available:
        return go.Figure()

    countries_plot = df["country"].unique()[:5]
    fig = go.Figure()

    categories = [c.replace("pillar_", "").replace("_", " ").title() for c in available]
    categories.append(categories[0])  # Close the radar

    for i, ctry in enumerate(countries_plot):
        ctry_df = df[df["country"] == ctry]
        values = [ctry_df[c].mean() for c in available]
        values.append(values[0])

        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories, name=ctry,
            fill="toself", opacity=0.6,
            line=dict(color=px.colors.qualitative.Plotly[i % 10], width=2),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#aaa")),
            angularaxis=dict(tickfont=dict(color="#ddd")),
            bgcolor=COLORS["bg_card"],
        ),
        paper_bgcolor=COLORS["bg_card"],
        plot_bgcolor=COLORS["bg_card"],
        legend=dict(font=dict(color="#ddd")),
        margin=dict(l=40, r=40, t=20, b=20),
        showlegend=True,
    )
    return fig


@app.callback(
    Output("trajectory-chart", "figure"),
    Input("country-filter", "value"),
)
def update_trajectory(country):
    df = RAW_DF.copy()
    if "cri_score" not in df.columns:
        # Re-compute CRI for trajectory
        results = CRI_ENGINE.compute_batch(df)
        df = df.merge(results[["zone_id", "cri_score"]], on="zone_id", how="left")

    if country != "ALL":
        df = df[df["country"] == country]

    # Aggregate by country + period
    traj = df.groupby(["country", "period"])["cri_score"].mean().reset_index()
    traj = traj.sort_values("period")

    fig = px.line(
        traj, x="period", y="cri_score", color="country",
        markers=True,
        labels={"cri_score": "Mean CRI Score", "period": "Period"},
    )
    fig.add_hline(y=25, line_dash="dot", line_color=COLORS["critical"],
                  annotation_text="Critical threshold (25)", annotation_font_color="#aaa")
    fig.add_hline(y=40, line_dash="dot", line_color=COLORS["watch"],
                  annotation_text="Watch threshold (40)", annotation_font_color="#aaa")
    fig.update_layout(
        paper_bgcolor=COLORS["bg_card"], plot_bgcolor=COLORS["bg_card"],
        font=dict(color="#ddd"),
        legend=dict(font=dict(color="#ddd"), bgcolor=COLORS["bg_dark"]),
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333", range=[0, 100]),
    )
    return fig


@app.callback(
    Output("distribution-chart", "figure"),
    Input("country-filter", "value"),
)
def update_distribution(country):
    df = FULL_DF.copy()
    if country != "ALL":
        df = df[df["country"] == country]

    fig = px.histogram(
        df, x="cri_score", nbins=25, color="country",
        labels={"cri_score": "CRI Score", "count": "Number of Zones"},
        barmode="overlay", opacity=0.7,
    )
    for threshold, label, color in [(25, "Critical", COLORS["critical"]), (40, "Watch", COLORS["watch"])]:
        fig.add_vline(x=threshold, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_font_color=color)

    fig.update_layout(
        paper_bgcolor=COLORS["bg_card"], plot_bgcolor=COLORS["bg_card"],
        font=dict(color="#ddd"),
        legend=dict(font=dict(color="#ddd"), bgcolor=COLORS["bg_dark"]),
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor="#333", range=[0, 100]),
        yaxis=dict(gridcolor="#333"),
    )
    return fig


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    debug = os.environ.get("SPACE_ID", None) is None
    print("\n🌡️  AdaptAction CRI Dashboard starting...")
    print(f"📍  http://localhost:{port}\n")
    app.run(debug=debug, host="0.0.0.0", port=port)
