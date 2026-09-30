"""
prediction_weather.py

Core forecasting logic for the Weather Detector project.

This module talks to the free Open-Meteo API (no API key required) and turns
raw hourly forecast data into human-readable "rain windows" and "sunny
windows" for one or many cities/townships at once -- i.e. answers to:

    - If it's going to rain, exactly when does it start, and how long
      will it last?
    - If it's sunny, how long will the sunny stretch last?

MULTI-CITY DESIGN:
Open-Meteo supports requesting many locations in a single call by passing
comma-separated latitude/longitude lists -- it returns one forecast object
per location, in the same order. This module uses that to fetch dozens or
hundreds of cities/townships efficiently, splitting into batches
(api.batch_size in the config) so URLs stay a safe length.

IMPORTANT HONESTY NOTE:
No weather model can promise a fixed "90% accuracy" on exact rain timing --
that number depends on the day, the season, and how far out you're
forecasting. Instead of faking a constant accuracy figure, this module
computes a real confidence score for every window, based on how strong and
how consistent the underlying hourly data is. Treat "confidence" as
"how much the model agrees with itself", not a guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any

import os

import requests
from dateutil import parser as dateparser


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class HourlyPoint:
    time: datetime
    temperature_c: float
    precipitation_probability: float   # 0-100
    precipitation_mm: float
    cloud_cover: float                 # 0-100
    weather_code: int
    humidity: float = 0.0              # relative humidity, 0-100
    wind_speed: float = 0.0            # km/h
    pressure: float = 0.0              # hPa
    model_agreement: float = 50.0      # 0-100: how much independent models agree (see ensemble fetch). 50 = unknown/neutral.


@dataclass
class Window:
    start: datetime
    end: datetime
    kind: str                # "rain" or "sunny"
    avg_value: float         # avg precip probability (rain) or avg (100-cloud) (sunny)
    confidence_percent: float
    confidence_label: str

    @property
    def duration_minutes(self) -> int:
        return int((self.end - self.start).total_seconds() // 60)

    def duration_str(self) -> str:
        mins = self.duration_minutes
        hours, minutes = divmod(mins, 60)
        parts = []
        if hours:
            parts.append(f"{hours}h")
        if minutes or not parts:
            parts.append(f"{minutes}m")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "duration_minutes": self.duration_minutes,
            "confidence_percent": self.confidence_percent,
            "confidence_label": self.confidence_label,
        }


HOURLY_VARS = [
    "temperature_2m",
    "precipitation_probability",
    "precipitation",
    "cloud_cover",
    "weather_code",
    "relative_humidity_2m",
    "wind_speed_10m",
    "surface_pressure",
]


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

def _parse_hourly_block(hourly: Dict[str, Any]) -> List[HourlyPoint]:
    times = hourly["time"]
    temps = hourly["temperature_2m"]
    precip_prob = hourly["precipitation_probability"]
    precip_mm = hourly["precipitation"]
    cloud_cover = hourly["cloud_cover"]
    weather_code = hourly["weather_code"]
    humidity = hourly.get("relative_humidity_2m", [0.0] * len(times))
    wind_speed = hourly.get("wind_speed_10m", [0.0] * len(times))
    pressure = hourly.get("surface_pressure", [0.0] * len(times))

    points: List[HourlyPoint] = []
    for i, t in enumerate(times):
        points.append(HourlyPoint(
            time=dateparser.parse(t),
            temperature_c=temps[i],
            precipitation_probability=precip_prob[i],
            precipitation_mm=precip_mm[i],
            cloud_cover=cloud_cover[i],
            weather_code=weather_code[i],
            humidity=humidity[i],
            wind_speed=wind_speed[i],
            pressure=pressure[i],
        ))
    return points


def _fetch_model_agreement_batch(batch: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Fetch precipitation from several independent weather models (ECMWF, GFS,
    ICON, GEM by default) for this batch of locations, and compute how much
    those models AGREE, hour by hour -- this is a genuine confidence signal
    (multi-model consensus is a standard meteorological technique), unlike
    a single model's own internal variance.

    Returns {} entirely (never raises) on any failure -- this is a pure
    enhancement layer; if it's unavailable, callers fall back to the
    original single-model confidence calculation instead of crashing.
    """
    ens_cfg = config.get("ensemble", {})
    if not ens_cfg.get("enabled", False):
        return {}

    models = ens_cfg.get("models", ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless"])
    threshold_mm = ens_cfg.get("rain_threshold_mm", 0.1)
    api_cfg = config["api"]

    lats = ",".join(str(loc["latitude"]) for loc in batch)
    lons = ",".join(str(loc["longitude"]) for loc in batch)

    params = {
        "latitude": lats,
        "longitude": lons,
        "timezone": config.get("timezone", "auto"),
        "forecast_days": api_cfg.get("forecast_days", 2),
        "hourly": "precipitation",
        "models": ",".join(models),
    }

    try:
        resp = requests.get(api_cfg["base_url"], params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return {}

    if isinstance(data, dict):
        data = [data]

    results: Dict[str, List[float]] = {}
    for loc, entry in zip(batch, data):
        hourly = entry.get("hourly")
        if not hourly:
            continue

        model_arrays = []
        for m in models:
            key = f"precipitation_{m}"
            arr = hourly.get(key)
            if arr is not None:
                model_arrays.append(arr)

        if not model_arrays:
            continue

        n = len(hourly.get("time", []))
        agreement: List[float] = []
        for i in range(n):
            votes = []
            for arr in model_arrays:
                if i < len(arr) and arr[i] is not None:
                    votes.append(arr[i] >= threshold_mm)
            if votes:
                yes = sum(1 for v in votes if v)
                no = len(votes) - yes
                agreement.append(max(yes, no) / len(votes) * 100)
            else:
                agreement.append(50.0)  # no data for this hour -- neutral

        results[loc["name"]] = agreement

    return results


CURRENT_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "surface_pressure",
]


def _apply_live_current(entry: Dict[str, Any], points: List[HourlyPoint]) -> None:
    """
    Open-Meteo's "current" block is based on the latest real analysis --
    closer to an actual live reading than picking the nearest cell out of
    the hourly FORECAST array. This overwrites the matching hour's point
    in-place with those live values (temperature/humidity/wind/pressure
    only -- rain probability still comes from the hourly forecast, since
    "current" doesn't include it).
    """
    current = entry.get("current")
    if not current or not points:
        return

    try:
        current_time = dateparser.parse(current["time"])
    except (KeyError, ValueError, TypeError):
        return

    current_hour = current_time.replace(minute=0, second=0, microsecond=0)
    for p in points:
        if p.time.replace(minute=0, second=0, microsecond=0) == current_hour:
            if current.get("temperature_2m") is not None:
                p.temperature_c = current["temperature_2m"]
            if current.get("relative_humidity_2m") is not None:
                p.humidity = current["relative_humidity_2m"]
            if current.get("wind_speed_10m") is not None:
                p.wind_speed = current["wind_speed_10m"]
            if current.get("surface_pressure") is not None:
                p.pressure = current["surface_pressure"]
            break


def _fetch_batch(batch: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, List[HourlyPoint]]:
    """Fetch one batch of locations in a single Open-Meteo request."""
    api_cfg = config["api"]

    lats = ",".join(str(loc["latitude"]) for loc in batch)
    lons = ",".join(str(loc["longitude"]) for loc in batch)

    params = {
        "latitude": lats,
        "longitude": lons,
        "timezone": config.get("timezone", "auto"),
        "forecast_days": api_cfg.get("forecast_days", 2),
        "hourly": ",".join(HOURLY_VARS),
        "current": ",".join(CURRENT_VARS),
    }

    try:
        resp = requests.get(api_cfg["base_url"], params=params, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch forecast data: {exc}") from exc

    data = resp.json()

    # Open-Meteo returns a single object when only one location is requested,
    # but a LIST of objects (same order as input) when multiple are requested.
    if isinstance(data, dict):
        data = [data]

    results: Dict[str, List[HourlyPoint]] = {}
    for loc, entry in zip(batch, data):
        hourly = entry.get("hourly")
        points = _parse_hourly_block(hourly) if hourly else []
        _apply_live_current(entry, points)  # closer-to-real "right now" values
        results[loc["name"]] = points

    # Layer in multi-model agreement, if enabled. Matched by hour INDEX --
    # both calls use identical timezone/forecast_days, so the hourly time
    # grids line up.
    agreement_by_name = _fetch_model_agreement_batch(batch, config)
    for name, points in results.items():
        agreement = agreement_by_name.get(name)
        if not agreement:
            continue
        for i, p in enumerate(points):
            if i < len(agreement):
                p.model_agreement = agreement[i]

    return results


def fetch_forecast_multi(config: Dict[str, Any]) -> Dict[str, List[HourlyPoint]]:
    """
    Fetch hourly forecast data for every location listed in config['locations'].
    Automatically splits into batches (config['api']['batch_size']) so this
    scales to a long list of cities/townships without hitting URL limits.

    Returns: dict mapping location name -> list of HourlyPoint.
    """
    locations = config["locations"]
    if not locations:
        raise ValueError("Config 'locations' list is empty -- add at least one city.")

    batch_size = config["api"].get("batch_size", 50)
    results: Dict[str, List[HourlyPoint]] = {}

    for i in range(0, len(locations), batch_size):
        batch = locations[i:i + batch_size]
        results.update(_fetch_batch(batch, config))

    return results


# --------------------------------------------------------------------------
# Confidence scoring
# --------------------------------------------------------------------------

def _confidence_from_values(values: List[float], model_agreement_values: List[float] = None):
    """
    Turn a list of 0-100 'signal strength' values (e.g. rain probabilities
    for the hours in a window) into a confidence score.

    Rewards a strong average signal and penalizes inconsistency (variance)
    across the window. If model_agreement_values is provided (how much
    independent weather models -- ECMWF, GFS, ICON, GEM -- agree hour by
    hour), it's blended in as a second, genuinely independent confidence
    signal: multi-model consensus is a standard meteorological technique,
    distinct from a single model's own internal hour-to-hour variance.
    Capped at 95% -- forecasts are never a sure thing.
    """
    if not values:
        return 0.0, "no data"

    avg = sum(values) / len(values)
    variance = sum((v - avg) ** 2 for v in values) / len(values)
    consistency_penalty = min(variance ** 0.5, 30)

    raw_confidence = avg - consistency_penalty

    if model_agreement_values:
        avg_agreement = sum(model_agreement_values) / len(model_agreement_values)
        # Equal-weight blend: our own signal-strength estimate, and how much
        # independent models agree with each other.
        raw_confidence = (raw_confidence + avg_agreement) / 2

    confidence = max(5.0, min(95.0, raw_confidence))

    if confidence >= 80:
        label = "high confidence"
    elif confidence >= 60:
        label = "moderate confidence"
    elif confidence >= 40:
        label = "low confidence"
    else:
        label = "very low confidence"

    return round(confidence, 1), label


# --------------------------------------------------------------------------
# Window detection
# --------------------------------------------------------------------------

def _merge_into_windows(points, is_match_fn, value_fn, kind: str, min_window_minutes: int) -> List[Window]:
    windows: List[Window] = []
    current_group: List[HourlyPoint] = []

    def flush():
        if not current_group:
            return
        start = current_group[0].time
        end = current_group[-1].time + timedelta(hours=1)

        duration_minutes = int((end - start).total_seconds() // 60)
        if duration_minutes < min_window_minutes:
            return

        values = [value_fn(p) for p in current_group]
        # Only rain windows get the model-agreement blend -- it's computed
        # from precipitation consensus, so it's meaningful for rain but not
        # a meaningful signal for "is it sunny".
        agreement_values = [p.model_agreement for p in current_group] if kind == "rain" else None
        confidence, label = _confidence_from_values(values, agreement_values)
        windows.append(Window(
            start=start,
            end=end,
            kind=kind,
            avg_value=sum(values) / len(values),
            confidence_percent=confidence,
            confidence_label=label,
        ))

    for p in points:
        if is_match_fn(p):
            current_group.append(p)
        else:
            flush()
            current_group = []
    flush()

    return windows


def find_rain_windows(points: List[HourlyPoint], config: Dict[str, Any]) -> List[Window]:
    threshold = config["thresholds"]["rain_probability_percent"]
    min_minutes = config["thresholds"]["min_window_minutes"]

    return _merge_into_windows(
        points,
        is_match_fn=lambda p: p.precipitation_probability >= threshold,
        value_fn=lambda p: p.precipitation_probability,
        kind="rain",
        min_window_minutes=min_minutes,
    )


def find_sunny_windows(points: List[HourlyPoint], config: Dict[str, Any]) -> List[Window]:
    threshold = config["thresholds"]["sunny_cloud_cover_percent"]
    min_minutes = config["thresholds"]["min_window_minutes"]

    def is_sunny(p: HourlyPoint) -> bool:
        # weather_code 0/1 = clear/mainly clear (Open-Meteo WMO codes)
        return p.cloud_cover <= threshold and p.weather_code in (0, 1)

    return _merge_into_windows(
        points,
        is_match_fn=is_sunny,
        value_fn=lambda p: 100 - p.cloud_cover,
        kind="sunny",
        min_window_minutes=min_minutes,
    )


# --------------------------------------------------------------------------
# Telegram alerts
# --------------------------------------------------------------------------

def send_telegram_message(text: str, config: Dict[str, Any], chat_id: str = None) -> bool:
    """
    Send a message via the Telegram Bot API.

    The bot token (and the default chat ID, when `chat_id` isn't given) are
    read from environment variables -- in GitHub Actions these come from
    repository Secrets, so nothing sensitive is ever committed to the repo.
    Pass `chat_id` explicitly to reply to whoever texted the bot, instead of
    the configured default chat/group.

    Returns True if the message was sent successfully, False otherwise
    (missing credentials or a failed request are logged, not raised, so
    a Telegram outage never breaks the rest of the run).
    """
    tg_cfg = config.get("telegram", {})
    if not tg_cfg.get("enabled", False):
        return False

    token = os.environ.get(tg_cfg.get("bot_token_env", "TELEGRAM_BOT_TOKEN"))
    target_chat_id = chat_id or os.environ.get(tg_cfg.get("chat_id_env", "TELEGRAM_CHAT_ID"))

    if not token or not target_chat_id:
        print("Telegram alert skipped: bot token or chat ID not set in environment.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": target_chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, data=payload, timeout=15)
        resp.raise_for_status()
        return True
    except requests.RequestException as exc:
        print(f"Telegram alert failed to send: {exc}")
        return False


def get_and_confirm_telegram_updates(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fetch any new incoming Telegram messages (e.g. a user texting a city
    name) since the last time this was called, then immediately confirm
    them with Telegram so the same message is never processed twice --
    even though nothing needs to be persisted between separate runs.
    """
    tg_cfg = config.get("telegram", {})
    if not tg_cfg.get("enabled", False):
        return []

    token = os.environ.get(tg_cfg.get("bot_token_env", "TELEGRAM_BOT_TOKEN"))
    if not token:
        print("Telegram updates skipped: bot token not set in environment.")
        return []

    url = f"https://api.telegram.org/bot{token}/getUpdates"

    try:
        resp = requests.get(url, params={"timeout": 0}, timeout=15)
        resp.raise_for_status()
        updates = resp.json().get("result", [])
    except requests.RequestException as exc:
        print(f"Failed to fetch Telegram updates: {exc}")
        return []

    if updates:
        max_update_id = max(u["update_id"] for u in updates)
        try:
            # Confirming with offset = last_id + 1 tells Telegram these are
            # handled, so they won't be returned again on the next run.
            requests.get(url, params={"offset": max_update_id + 1, "timeout": 0}, timeout=15)
        except requests.RequestException:
            pass  # not fatal -- worst case, these get processed again next run

    return updates


# --------------------------------------------------------------------------
# Optional: Gemini-generated weather tips
# --------------------------------------------------------------------------

def generate_ai_tip(city_name: str, current: Dict[str, Any], config: Dict[str, Any]) -> str:
    """
    Ask Gemini for a short, natural-language weather tip based on current
    conditions. Returns None (never raises) if the feature is disabled, the
    API key is missing, or the request fails for any reason -- callers
    should fall back to a static tip in that case, since this is a purely
    cosmetic enhancement and must never break the actual weather report.
    """
    gem_cfg = config.get("gemini", {})
    if not gem_cfg.get("enabled", False):
        return None

    api_key = os.environ.get(gem_cfg.get("api_key_env", "GEMINI_API_KEY"))
    if not api_key:
        print("Gemini tip skipped: API key not set in environment.")
        return None

    model = gem_cfg.get("model", "gemini-2.0-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    prompt = (
        f"Write ONE short, friendly, practical weather tip (max 20 words, "
        f"no greeting, no city name) for someone in {city_name} right now. "
        f"Conditions: {current.get('temperature_c', 0):.1f}C, "
        f"{current.get('humidity', 0):.0f}% humidity, "
        f"{current.get('wind_speed', 0):.1f} km/h wind, "
        f"{current.get('precipitation_probability', 0):.0f}% chance of rain. "
        f"Just the tip, nothing else."
    )

    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return text if text else None
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        print(f"Gemini tip failed, falling back to static tip: {exc}")
        return None
