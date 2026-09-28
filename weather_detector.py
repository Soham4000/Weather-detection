"""
weather_detector.py

Weather Detector
----------------
- Current temperature/humidity/pressure/wind are fetched separately
  from the current-weather endpoint.
- Rain and sunny windows continue to come from the hourly forecast.
- No AI.
- Telegram support preserved.
- JSON output preserved.
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import yaml

from prediction_weather import (
    fetch_forecast_multi,
    find_rain_windows,
    find_sunny_windows,
    send_telegram_message,
    get_and_confirm_telegram_updates,
    Window,
)


# ============================================================
# CONFIG
# ============================================================

OPEN_METEO_CURRENT_URL = "https://api.open-meteo.com/v1/forecast"


# ============================================================
# CONFIG LOADING
# ============================================================

def load_config(path: str) -> Dict[str, Any]:

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError("Config file is empty.")

    # Load locations from external JSON if configured.
    if "locations_file" in config:

        locations_path = config["locations_file"]

        try:
            with open(
                locations_path,
                "r",
                encoding="utf-8"
            ) as f:
                config["locations"] = json.load(f)

        except (
            FileNotFoundError,
            json.JSONDecodeError
        ) as exc:

            raise ValueError(
                f"Could not load locations_file "
                f"'{locations_path}': {exc}"
            ) from exc

    required_top_level = [
        "locations",
        "api",
        "thresholds",
        "output",
    ]

    for key in required_top_level:

        if key not in config:
            raise ValueError(
                f"Config file is missing required section: '{key}'"
            )

    return config


# ============================================================
# COMMAND LINE OVERRIDES
# ============================================================

def apply_overrides(
    config: Dict[str, Any],
    args: argparse.Namespace
) -> None:

    if args.lat is not None and args.lon is not None:

        config["locations"] = [{
            "name": args.name or "Custom Location",
            "country": "",
            "latitude": args.lat,
            "longitude": args.lon,
        }]


# ============================================================
# WINDOW PRINTING
# ============================================================

def print_windows(
    title: str,
    windows: List[Window],
    config: Dict[str, Any]
) -> None:

    time_fmt = config["output"].get(
        "time_format",
        "%I:%M %p"
    )

    print(f"  {title}:")

    if not windows:

        print(
            "    None predicted in the forecast period."
        )

        return

    for w in windows:

        start_str = w.start.strftime(time_fmt)
        end_str = w.end.strftime(time_fmt)

        date_str = w.start.strftime(
            "%a %d %b"
        )

        print(
            f"    {date_str}: "
            f"{start_str} -> {end_str} "
            f"(duration: {w.duration_str()}) "
            f"-- {w.confidence_label} "
            f"({w.confidence_percent}%)"
        )


# ============================================================
# HOURLY DETAIL
# ============================================================

def print_hourly_detail(
    points,
    config: Dict[str, Any]
) -> None:

    time_fmt = config["output"].get(
        "time_format",
        "%I:%M %p"
    )

    for p in points:

        print(
            f"    "
            f"{p.time.strftime('%a %d %b ' + time_fmt)} | "
            f"temp: {p.temperature_c:>5.1f}C | "
            f"rain chance: "
            f"{p.precipitation_probability:>3.0f}% | "
            f"precip: "
            f"{p.precipitation_mm:>4.1f}mm | "
            f"cloud cover: "
            f"{p.cloud_cover:>3.0f}%"
        )


# ============================================================
# DISCLAIMER
# ============================================================

def print_disclaimer() -> None:

    print("\nNote on accuracy")
    print("-" * 17)

    print(
        "  Forecast confidence naturally drops the further "
        "out a window is.\n"
        "  Treat 'confidence' above as the strength and "
        "consistency of the forecast signal,\n"
        "  not as a guaranteed accuracy rate."
    )


# ============================================================
# CURRENT WEATHER
# ============================================================

def fetch_current_weather(
    latitude: float,
    longitude: float,
    timezone: str = "auto"
) -> Dict[str, Any]:

    """
    Fetch CURRENT weather separately from the hourly forecast.

    This is important because points[0] is a forecast hour and
    should not automatically be treated as "current weather".
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,

        # Dedicated current conditions.
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "showers",
            "weather_code",
            "cloud_cover",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
        ]),

        "timezone": timezone,

        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }

    url = (
        OPEN_METEO_CURRENT_URL
        + "?"
        + urllib.parse.urlencode(params)
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                "WeatherDetector/2.0"
        },
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=30
        ) as response:

            data = json.loads(
                response.read().decode("utf-8")
            )

    except Exception as exc:

        raise RuntimeError(
            f"Could not fetch current weather: {exc}"
        ) from exc

    current = data.get("current")

    if not current:
        raise RuntimeError(
            "Current weather data was not returned."
        )

    def number(
        key: str,
        default: float = 0.0
    ) -> float:

        value = current.get(key)

        try:
            return float(value)
        except (
            TypeError,
            ValueError
        ):
            return default

    def integer(
        key: str,
        default: int = 0
    ) -> int:

        value = current.get(key)

        try:
            return int(value)
        except (
            TypeError,
            ValueError
        ):
            return default

    return {
        "time": current.get(
            "time"
        ),

        "temperature_c": number(
            "temperature_2m"
        ),

        "feels_like_c": number(
            "apparent_temperature"
        ),

        "humidity": number(
            "relative_humidity_2m"
        ),

        "wind_speed": number(
            "wind_speed_10m"
        ),

        "wind_direction": number(
            "wind_direction_10m"
        ),

        "pressure": number(
            "surface_pressure"
        ),

        "precipitation_mm": number(
            "precipitation"
        ),

        "rain_mm": number(
            "rain"
        ),

        "showers_mm": number(
            "showers"
        ),

        "cloud_cover": number(
            "cloud_cover"
        ),

        "weather_code": integer(
            "weather_code"
        ),

        # This value is the source timestamp,
        # not the computer's local time.
        "source": "Open-Meteo current",
    }


# ============================================================
# CURRENT POINT VALIDATION
# ============================================================

def get_nearest_forecast_point(
    points
):
    """
    Used only as a fallback if current-weather API fails.
    """

    if not points:
        return None

    point_tz = points[0].time.tzinfo

    try:
        now = datetime.now(point_tz)
    except Exception:
        now = datetime.now()

    return min(
        points,
        key=lambda p: abs(
            (
                p.time - now
            ).total_seconds()
        )
    )


# ============================================================
# ALERT MESSAGE
# ============================================================

def build_alert_message(
    all_results: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> str:

    tg_cfg = config.get(
        "telegram",
        {}
    )

    alert_kinds = set(
        tg_cfg.get(
            "alert_on",
            ["rain"]
        )
    )

    min_confidence = tg_cfg.get(
        "min_confidence_percent",
        60
    )

    time_fmt = config["output"].get(
        "time_format",
        "%I:%M %p"
    )

    lines: List[str] = []

    for city in all_results:

        city_lines = []

        for kind_key, label in (
            ("rain_windows", "Rain"),
            ("sunny_windows", "Sunny")
        ):

            kind_name = (
                "rain"
                if label == "Rain"
                else "sunny"
            )

            if kind_name not in alert_kinds:
                continue

            for w in city[kind_key]:

                if (
                    w["confidence_percent"]
                    < min_confidence
                ):
                    continue

                start = datetime.fromisoformat(
                    w["start"]
                )

                end = datetime.fromisoformat(
                    w["end"]
                )

                city_lines.append(
                    f"  {label}: "
                    f"{start.strftime(time_fmt)} "
                    f"-> "
                    f"{end.strftime(time_fmt)} "
                    f"({w['confidence_percent']}% "
                    f"{w['confidence_label']})"
                )

        if city_lines:

            lines.append(
                f"<b>{city['name']}</b>"
            )

            lines.extend(
                city_lines
            )

    if not lines:
        return ""

    header = "\u26c5 Weather Alert\n"

    return (
        header
        + "\n".join(lines)
    )


# ============================================================
# SUMMARY MESSAGE
# ============================================================

def build_summary_message(
    all_results: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> str:

    time_fmt = config["output"].get(
        "time_format",
        "%I:%M %p"
    )

    lines = [
        "\U0001F324 Weather Update"
    ]

    for city in all_results:

        lines.append(
            f"\n<b>{city['name']}</b>"
        )

        current = city.get(
            "current",
            {}
        )

        # ----------------------------------------------------
        # CURRENT CONDITIONS
        # ----------------------------------------------------

        temp = current.get(
            "temperature_c"
        )

        humidity = current.get(
            "humidity"
        )

        pressure = current.get(
            "pressure"
        )

        wind = current.get(
            "wind_speed"
        )

        if temp is not None:

            lines.append(
                f"  🌡 Temperature: "
                f"{temp:.1f}°C"
            )

        if humidity is not None:

            lines.append(
                f"  💧 Humidity: "
                f"{humidity:.0f}%"
            )

        if pressure is not None:

            lines.append(
                f"  🔵 Pressure: "
                f"{pressure:.1f} hPa"
            )

        if wind is not None:

            lines.append(
                f"  🌬 Wind: "
                f"{wind:.1f} km/h"
            )

        # ----------------------------------------------------
        # RAIN
        # ----------------------------------------------------

        if city["rain_windows"]:

            w = city[
                "rain_windows"
            ][0]

            start = datetime.fromisoformat(
                w["start"]
            )

            end = datetime.fromisoformat(
                w["end"]
            )

            lines.append(
                f"  🌧 Rain: "
                f"{start.strftime(time_fmt)} "
                f"-> "
                f"{end.strftime(time_fmt)} "
                f"({w['confidence_percent']}% "
                f"{w['confidence_label']})"
            )

        else:

            lines.append(
                "  🌧 Rain: none predicted "
                "in the forecast period."
            )

        # ----------------------------------------------------
        # SUNNY
        # ----------------------------------------------------

        if city["sunny_windows"]:

            w = city[
                "sunny_windows"
            ][0]

            start = datetime.fromisoformat(
                w["start"]
            )

            end = datetime.fromisoformat(
                w["end"]
            )

            lines.append(
                f"  ☀️ Sunny: "
                f"{start.strftime(time_fmt)} "
                f"-> "
                f"{end.strftime(time_fmt)} "
                f"({w['confidence_percent']}% "
                f"{w['confidence_label']})"
            )

        else:

            lines.append(
                "  ☀️ Sunny: none predicted "
                "in the forecast period."
            )

    return "\n".join(lines)


# ============================================================
# SAVE JSON
# ============================================================

def save_results_json(
    all_results: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> None:

    path = config["output"].get(
        "results_path",
        "results/latest.json"
    )

    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "generated_at":
                    datetime.now().isoformat(),

                "cities":
                    all_results,
            },
            f,
            indent=2
        )

    print(
        f"\nSaved machine-readable "
        f"results to: {path}"
    )


# ============================================================
# CITY MATCHING
# ============================================================

def match_cities(
    query: str,
    all_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    q = query.strip().lower()

    exact = [
        c
        for c in all_results
        if c["name"].strip().lower() == q
    ]

    if exact:
        return exact

    return [
        c
        for c in all_results
        if (
            q in c["name"].lower()
            or c["name"].lower() in q
       
