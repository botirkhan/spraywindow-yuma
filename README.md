# SprayWindow Yuma

Field-specific pesticide spray timing for the Yuma, Arizona area. Pick any point on the map and get Go / Caution / Stop for right now, plus the safe spray windows for the next 72 hours, each with a confidence percentage.

Built in one day at Hack Arizona x AgX Arizona Summit for the "Spray or Wait?" challenge.

## The problem

A Yuma grower told us the nearby weather stations all read differently, so he cannot be confident what the wind will be at his own field, or when. On the afternoon we built this, five AZMet stations within 40 miles of each other read from 2.5 mph to 9.8 mph at the same moment.

## What it does

1. Pulls the hourly weather-model forecast for the exact field coordinates (2 m, 10 m and 80 m above ground) from Open-Meteo.
2. Corrects it with what five AZMet stations have taught a small machine-learning model about how real wind and temperature differ from the forecast at each hour of the day, weighted by distance to the field.
3. Predicts a low, likely and high wind speed, plus temperature and wind direction, for each hour.
4. Turns that into the chance the wind stays inside the applicator's limits (default 3-10 mph), combines it with an inversion estimate from the 80 m vs 2 m temperature difference, and lists the safe windows.
5. Flags wind shear between heights and wind blowing toward a marked sensitive area.
6. Logs every decision with a time stamp, the readings and what triggered the status (CSV download).
7. Asks for the crop and pesticide type. Herbicides turn "wind toward a sensitive area" into Stop; high-volatility products get extra caution in heat (85 F+) and when an inversion may follow the spray window.
8. Lets the grower mark any number of sensitive areas (neighboring crops, homes, water) by clicking the map, and flags hours when the wind blows toward any of them.
9. Stores past sprays in a local SQLite database (`data/sprays.db`) with the conditions at the time and the outcome.
10. Includes a Google Gemini advisor that answers questions in plain language. It is given the current forecast, station readings, tested accuracy and the grower's spray records, and it reports real token usage.
11. Sends phone push alerts (status change, window opening, window closing, inversion expected) through the free ntfy app: install ntfy, subscribe to the random topic shown in the app, and switch on live alerts. A phone preview on screen shows what goes out and when, and the page refreshes itself every 5 minutes.

## Measured accuracy

Trained on 43,795 station-hours (Nov 2025 - Nov 2026). Numbers below are from `results/metrics_2026-11-15.json`.

**Location with no station** (each station hidden in turn and predicted from the others):

| Wind speed error | mph |
|---|---|
| SprayWindow model | 1.52 |
| Forecast, adjusted to station height | 1.86 |
| Nearest station's reading at the same moment | 1.87 |

- Temperature error: 2.1 F (forecast 2.6 F).
- "Wind in range" calls correct: 73% (forecast 64%, nearest station 69%).
- When the model says 80%+ confidence, it was right 88% of the time (18% of hours).

**By lead time** (trained on the past, tested on the last 60 days using forecasts as they were actually issued):

| Days ahead | Model wind error (mph) | Forecast wind error (mph) | In-range calls correct |
|---|---|---|---|
| Same day | 1.66 | 1.98 | 72% |
| 1 day | 2.12 | 2.39 | 64% |
| 2 days | 2.59 | 2.81 | 59% |
| 3 days | 2.67 | 2.90 | 58% |

More history helps: wind error falls from 1.90 to 1.71 mph going from 1 to 10 months of training data.

## Limits

- AZMet stations measure temperature at one height. The inversion chance is a rule-based estimate from forecast temperature at 2 m and 80 m, not a measurement. A two-height sensor at the field would replace it.
- Two to three days ahead the in-range calls are only modestly better than chance; the confidence figure widens to reflect that.
- Tested on five stations. A field far from all of them will be less accurate.
- The Advice box is written by templates. The Gemini advisor needs your own API key; without one the app still works. The product rules (herbicide, volatility) are simple general guidance, not label requirements.
- This tool supports the applicator's decision. It does not replace pesticide labels, on-site checks or regulations.

## Run it (Windows)

1. Install Python 3.12 (Microsoft Store or python.org).
2. Double-click `1_setup.bat` (installs packages).
3. Double-click `2_train.bat` (downloads a year of data and trains; about 2 minutes).
4. Double-click `3_run.bat` and open http://127.0.0.1:8000.
5. Optional, for the advisor: copy `gemini_key.example.txt` to `gemini_key.txt` and paste your Gemini API key after the `=` sign. That file is git-ignored.

On macOS or Linux: `pip install -r requirements.txt`, `python pipeline.py`, `python server.py`.

## Files

| File | Purpose |
|---|---|
| `config.py` | Stations, area bounds, data source addresses |
| `datasrc.py` | AZMet and Open-Meteo downloads |
| `core.py` | Features, the model, probability maths |
| `pipeline.py` | Download, honest tests, final training |
| `engine.py` | Live forecast, statuses, spray windows, log |
| `store.py` | SQLite database of past sprays |
| `advisor.py` | Gemini advisor (context building, API call, token usage) |
| `notify.py` | Phone push alerts through ntfy.sh |
| `server.py` | Local web server |
| `static/index.html` | The app screen |

## Data sources

- [AZMet](https://azmet.arizona.edu/), Arizona Meteorological Network, University of Arizona (station observations).
- [Open-Meteo](https://open-meteo.com/) forecast, historical-forecast and previous-runs APIs (CC BY 4.0).
- Map tiles: OpenStreetMap contributors.

The repository includes the downloaded data and the trained model in `data/`, so the app runs without step 3. The model file was saved with scikit-learn on Python 3.12; if it fails to load, run step 3.

## Next steps

Field sensor upload for per-field calibration, a two-height temperature sensor for measured inversions, text alerts, pesticide volatility profiles, and terrain and satellite inputs.
