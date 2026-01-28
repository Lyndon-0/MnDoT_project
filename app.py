import dash_bootstrap_components as dbc
from dash import Dash, html
from flask_caching import Cache

from mndot_dash.callbacks import register_callbacks
from mndot_dash.ui import build_layout

APP_CSS = """
  :root {
    --app-bg: #f6f7fb;
    --card-bg: #ffffff;
    --border: #e5e7eb;
    --text: #111827;
    --muted: #6b7280;
    --shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    --radius: 14px;
  }

  html, body {
    height: 100%;
    background: var(--app-bg);
    color: var(--text);
  }

  /* Page shell */
  .app-root {
    min-height: 100vh;
    padding: 22px 24px;
    box-sizing: border-box;
  }

  /* Center the app on very wide screens */
  .app-root .container-fluid {
    max-width: 1700px;
    margin: 0 auto;
    padding-left: 0;
    padding-right: 0;
  }

  /* Increase gutter between the main panels */
  .app-root .container-fluid > .row {
    --bs-gutter-x: 18px;
    --bs-gutter-y: 18px;
  }

  /* Turn the two main columns into cards */
  .app-root .container-fluid > .row > .col-3,
  .app-root .container-fluid > .row > .col-9 {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 16px 16px 18px 16px;
  }

  /* Typography cleanup */
  .app-root h3, .app-root h4, .app-root h5 {
    margin-top: 0.2rem;
  }
  .app-root hr {
    margin: 10px 0 14px 0;
    border-top: 1px solid var(--border);
    opacity: 1;
  }

  /* Inputs: soften borders a bit */
  .app-root .Select-control,
  .app-root .DateInput_input,
  .app-root input.form-control {
    border-color: var(--border) !important;
    border-radius: 10px !important;
    box-shadow: none !important;
  }

  /* Graph containers */
  .app-root #map-graph,
  .app-root #ts-graph {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    background: #ffffff;
  }
  .app-root #map-graph {
    margin-top: 6px;
  }

  /* Modal: match the card styling */
  .app-root .modal-content {
    border-radius: var(--radius);
    border: 1px solid var(--border);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.18);
  }

  /* Filters: consistent vertical rhythm */
  .app-root .filters-panel .filter-block {
    margin-bottom: 14px;
  }
  .app-root .filters-panel label {
    margin-bottom: 6px;
    font-weight: 600;
    color: var(--text);
  }
  .app-root .filters-panel .alert {
    margin: 0 0 14px 0;
  }
  .app-root .filters-panel .accordion {
    margin-top: 2px;
  }
"""

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="MnDOT Detector Monitor")
server = app.server

# In-memory cache (good for dev/single-process). For production, prefer Redis.
cache = Cache(server, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 3600})

app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>{APP_CSS}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""

app.layout = html.Div(build_layout(), className="app-root")
register_callbacks(app, cache)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8053, debug=True)
