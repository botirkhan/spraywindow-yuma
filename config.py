"""SprayWindow Yuma - shared settings."""
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
for _d in (DATA, LOGS):
    os.makedirs(_d, exist_ok=True)

# AZMet stations in the Yuma area (coordinates from the AZMet station list)
STATIONS = {
    "az02": {"name": "Yuma Valley", "lat": 32.7118, "lon": -114.7041, "elev_m": 36},
    "az14": {"name": "Yuma N.Gila", "lat": 32.7804, "lon": -114.5163, "elev_m": 43},
    "az36": {"name": "Yuma South",  "lat": 32.5655, "lon": -114.7440, "elev_m": 28},
    "az42": {"name": "Wellton ETo", "lat": 32.6932, "lon": -114.0674, "elev_m": 73},
    "az24": {"name": "Roll",        "lat": 32.7629, "lon": -113.9675, "elev_m": 81},
}
BOUNDS = {"lat_min": 32.40, "lat_max": 32.95, "lon_min": -114.90, "lon_max": -113.80}
DEFAULT_POINT = {"lat": 32.6700, "lon": -114.6200}

AZMET = "https://api.azmet.arizona.edu/v1/observations"
OM_FC = "https://api.open-meteo.com/v1/forecast"
OM_HIST = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OM_PREV = "https://previous-runs-api.open-meteo.com/v1/forecast"

# Open-Meteo variable -> short name
OMV = {
    "temperature_2m": "t2", "relative_humidity_2m": "rh", "dew_point_2m": "td",
    "wind_speed_10m": "ws10", "wind_direction_10m": "wd10", "wind_gusts_10m": "gust",
    "cloud_cover": "cloud", "shortwave_radiation": "rad", "surface_pressure": "pres",
    "temperature_80m": "t80", "wind_speed_80m": "ws80", "wind_direction_80m": "wd80",
    "boundary_layer_height": "blh",
}
HISTORY_DAYS = 365
TEST_DAYS = 60
PREV_DAYS = 92
HEIGHT_FACTOR = 0.79   # log-law factor, 10 m wind -> 3 m station height (z0 = 3 cm)
