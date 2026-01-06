import dash_bootstrap_components as dbc
from dash import Dash
from flask_caching import Cache

from mndot_dash.callbacks import register_callbacks
from mndot_dash.ui import build_layout

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# In-memory cache (good for dev/single-process). For production, prefer Redis.
cache = Cache(server, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 3600})

app.layout = build_layout()
register_callbacks(app, cache)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8053, debug=True)
