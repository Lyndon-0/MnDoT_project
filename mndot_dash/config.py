CH_HOST, CH_PORT, CH_DB = "127.0.0.1", 8123, "sensors"
DAILY_METRICS_TABLE = "daily_metrics"

ROUTE_OPTIONS = ["I-94", "I-494", "I-35E", "I-35W", "I-694"]
DIRECTION_OPTIONS = ["EB", "NB", "SB", "WB"]

CORRIDOR_OPTIONS = [
    "I-35E NB",
    "I-35E SB",
    "I-35W NB",
    "I-35W SB",
    "I-494 EB",
    "I-494 WB",
    "I-694 EB",
    "I-694 WB",
    "I-94 EB",
    "I-94 WB",
]

SENSOR_LABELS = ["V30 (volume)", "C30 (occupancy)", "S30 (speed)"]
UI2DB_SENSOR = {"V30 (volume)": "v30", "C30 (occupancy)": "c30", "S30 (speed)": "s30"}

BLUE = "#2563EB"
GRAY = "#9CA3AF"
