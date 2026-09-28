"""
prediction_weather.py

Non-AI weather prediction engine.

Features:
- Open-Meteo hourly forecast
- Multiple weather signals for rain detection
- Multiple weather signals for sunny/clear detection
- Neighbor-hour confirmation to reduce false positives
- Short-gap merging
- Rain intensity classification
- Dynamic confidence score
- Telegram support
- Batch forecasting for many locations

No AI / Gemini required.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class WeatherPoint:
    time: datetime

    temperature_c: float
    humidity: float
    wind_speed: float
    pressure: float

    precipitation_probability: float
    precipitation_mm: float
    rain_mm: float

    cloud_cover: float
    weather_code: int

    # Optional values if available from API
    showers_mm: float = 0.0
    snowfall_mm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time.isoformat(),
            "temperature_c": self.temperature_c,
            "humidity": self.humidity,
            "wind_speed": self.wind_speed,
            "pressure": self.pressure,
            "precipitation_probability": self.precipitation_probability,
            "precipitation_mm": self.precipitation_mm,
            "rain_mm": self.rain_mm,
            "cloud_cover": self.cloud_cover,
            "weather_code": self.weather_code,
            "showers_mm": self.showers_mm,
            "snowfall_mm": self.snowfall_mm,
        }


@dataclass
class Window:
    start: datetime
    end: datetime

    confidence_percent: int
    confidence_label: str

    duration_minutes: int

    max_rain_probability: float = 0.0
    total_precipitation_mm: float = 0.0
    max_precipitation_mm: float = 0.0

    def duration_str(self) -> str:
        hours, minutes = divmod(self.duration_minutes, 60)

        if hours and minutes:
            return f"{hours}h {minutes}m"

        if hours:
            return f"{hours}h"

        return f"{minutes}m"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "duration_minutes": self.duration_minutes,
            "confidence_percent": self.confidence_percent,
            "confidence_label": self.confidence_label,
            "max_rain_probability": round(self.max_rain_probability, 1),
            "total_precipitation_mm": round(
                self.total_precipitation_mm, 2
            ),
            "max_precipitation_mm": round(
                self.max_precipitation_mm, 2
            ),
        }


# ============================================================
# OPEN-METEO
# ============================================================

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation_probability",
    "precipitation",
    "rain",
    "showers",
    "snowfall",
    "cloud_cover",
    "surface_pressure",
    "wind_speed_10m",
    "weather_code",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def fetch_forecast(
    latitude: float,
    longitude: float,
    timezone: str = "auto",
    forecast_days: int = 7,
) -> List[WeatherPoint]:

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(HOURLY_VARIABLES),
        "forecast_days": forecast_days,
        "timezone": timezone,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }

    url = OPEN_METEO_URL + "?" + urllib.parse.urlencode(params)

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "WeatherDetector/2.0"
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

    except Exception as exc:
        raise RuntimeError(
            f"Unable to fetch weather forecast: {exc}"
        ) from exc

    hourly = data.get("hourly")

    if not hourly:
        raise RuntimeError("Open-Meteo returned no hourly forecast data.")

    times = hourly.get("time", [])

    points: List[WeatherPoint] = []

    for i, time_string in enumerate(times):

        try:
            dt = datetime.fromisoformat(time_string)
        except ValueError:
            continue

        points.append(
            WeatherPoint(
                time=dt,

                temperature_c=_safe_float(
                    hourly.get("temperature_2m", [0])[i]
                ),

                humidity=_safe_float(
                    hourly.get("relative_humidity_2m", [0])[i]
                ),

                wind_speed=_safe_float(
                    hourly.get("wind_speed_10m", [0])[i]
                ),

                pressure=_safe_float(
                    hourly.get("surface_pressure", [0])[i]
                ),

                precipitation_probability=_safe_float(
                    hourly.get(
                        "precipitation_probability",
                        [0]
                    )[i]
                ),

                precipitation_mm=_safe_float(
                    hourly.get("precipitation", [0])[i]
                ),

                rain_mm=_safe_float(
                    hourly.get("rain", [0])[i]
                ),

                cloud_cover=_safe_float(
                    hourly.get("cloud_cover", [0])[i]
                ),

                weather_code=_safe_int(
                    hourly.get("weather_code", [0])[i]
                ),

                showers_mm=_safe_float(
                    hourly.get("showers", [0])[i]
                ),

                snowfall_mm=_safe_float(
                    hourly.get("snowfall", [0])[i]
                ),
            )
        )

    return points


# ============================================================
# BATCH FORECAST
# ============================================================

def fetch_forecast_multi(
    config: Dict[str, Any]
) -> Dict[str, List[WeatherPoint]]:

    locations = config["locations"]

    api_cfg = config.get("api", {})

    forecast_days = int(
        api_cfg.get("forecast_days", 7)
    )

    timezone = config.get(
        "timezone",
        api_cfg.get("timezone", "auto")
    )

    results: Dict[str, List[WeatherPoint]] = {}

    for location in locations:

        name = location["name"]

        latitude = float(location["latitude"])
        longitude = float(location["longitude"])

        print(
            f"Fetching forecast for {name} "
            f"({latitude}, {longitude})..."
        )

        try:
            results[name] = fetch_forecast(
                latitude=latitude,
                longitude=longitude,
                timezone=timezone,
                forecast_days=forecast_days,
            )

        except Exception as exc:
            print(
                f"Warning: failed to fetch {name}: {exc}"
            )

            results[name] = []

    return results


# ============================================================
# WEATHER CODE HELPERS
# ============================================================

# WMO weather codes
RAIN_CODES = {
    51, 53, 55,       # drizzle
    56, 57,           # freezing drizzle
    61, 63, 65,       # rain
    66, 67,           # freezing rain
    80, 81, 82,       # rain showers
    95, 96, 99,       # thunderstorm
}


HEAVY_RAIN_CODES = {
    65,
    67,
    82,
    95,
    96,
    99,
}


CLEAR_CODES = {
    0,
    1,
}


PARTLY_CLOUDY_CODES = {
    2,
}


CLOUDY_CODES = {
    3,
}


def is_rain_code(code: int) -> bool:
    return code in RAIN_CODES


def is_heavy_rain_code(code: int) -> bool:
    return code in HEAVY_RAIN_CODES


def is_clear_code(code: int) -> bool:
    return code in CLEAR_CODES


# ============================================================
# RAIN SIGNAL
# ============================================================

def calculate_rain_signal(
    point: WeatherPoint,
    config: Dict[str, Any],
) -> float:

    """
    Calculates a rain signal from 0-100.

    This is NOT an accuracy percentage.

    It represents the strength of evidence that rain is occurring
    or likely during this hour.
    """

    thresholds = config.get("thresholds", {})

    probability_threshold = float(
        thresholds.get(
            "rain_probability_percent",
            40
        )
    )

    precipitation_threshold = float(
        thresholds.get(
            "rain_amount_mm",
            0.10
        )
    )

    probability = point.precipitation_probability
    precipitation = point.precipitation_mm
    rain_mm = point.rain_mm

    signal = 0.0

    # --------------------------------------------------------
    # Probability signal
    # --------------------------------------------------------

    if probability >= 90:
        signal += 35

    elif probability >= 75:
        signal += 30

    elif probability >= 60:
        signal += 24

    elif probability >= 50:
        signal += 18

    elif probability >= probability_threshold:
        signal += 12

    elif probability >= 25:
        signal += 6

    # --------------------------------------------------------
    # Actual precipitation signal
    # --------------------------------------------------------

    if precipitation >= 5:
        signal += 35

    elif precipitation >= 2:
        signal += 30

    elif precipitation >= 1:
        signal += 25

    elif precipitation >= 0.5:
        signal += 20

    elif precipitation >= precipitation_threshold:
        signal += 15

    elif precipitation > 0:
        signal += 8

    # --------------------------------------------------------
    # Rain-specific amount
    # --------------------------------------------------------

    if rain_mm >= 5:
        signal += 15

    elif rain_mm >= 2:
        signal += 12

    elif rain_mm >= 1:
        signal += 10

    elif rain_mm >= 0.5:
        signal += 7

    elif rain_mm > 0:
        signal += 4

    # --------------------------------------------------------
    # Weather code
    # --------------------------------------------------------

    if is_heavy_rain_code(point.weather_code):
        signal += 15

    elif is_rain_code(point.weather_code):
        signal += 10

    # --------------------------------------------------------
    # Cloud support
    # --------------------------------------------------------

    if point.cloud_cover >= 90:
        signal += 5

    elif point.cloud_cover >= 75:
        signal += 3

    # --------------------------------------------------------
    # Cap
    # --------------------------------------------------------

    return min(100.0, signal)


# ============================================================
# SUNNY SIGNAL
# ============================================================

def calculate_sunny_signal(
    point: WeatherPoint,
    config: Dict[str, Any],
) -> float:

    """
    Calculates strength of clear/sunny conditions.

    NOT an accuracy percentage.
    """

    signal = 0.0

    # Clear WMO code
    if point.weather_code == 0:
        signal += 45

    elif point.weather_code == 1:
        signal += 35

    elif point.weather_code == 2:
        signal += 20

    # Cloud cover
    if point.cloud_cover <= 10:
        signal += 35

    elif point.cloud_cover <= 25:
        signal += 28

    elif point.cloud_cover <= 40:
        signal += 18

    elif point.cloud_cover <= 55:
        signal += 8

    # No measurable precipitation
    if point.precipitation_mm <= 0.05:
        signal += 10

    # Low rain probability
    if point.precipitation_probability <= 10:
        signal += 10

    elif point.precipitation_probability <= 20:
        signal += 7

    elif point.precipitation_probability <= 30:
        signal += 4

    return min(100.0, signal)


# ============================================================
# CONFIDENCE
# ============================================================

def confidence_label(score: float) -> str:

    if score >= 85:
        return "Very Strong"

    if score >= 70:
        return "Strong"

    if score >= 55:
        return "Moderate"

    if score >= 40:
        return "Possible"

    return "Weak"


# ============================================================
# NEIGHBOR CONFIRMATION
# ============================================================

def neighbor_rain_support(
    points: List[WeatherPoint],
    index: int,
) -> float:

    """
    Looks at nearby hours.

    This prevents a single isolated forecast hour from
    immediately becoming a strong rain window.
    """

    scores = []

    for offset in (-2, -1, 0, 1, 2):

        j = index + offset

        if 0 <= j < len(points):

            scores.append(
                calculate_rain_signal(
                    points[j],
                    {}
                )
            )

    if not scores:
        return 0.0

    # Weighted central importance
    weights = []

    for offset in (-2, -1, 0, 1, 2):

        j = index + offset

        if 0 <= j < len(points):

            if offset == 0:
                weights.append(1.5)
            elif abs(offset) == 1:
                weights.append(1.0)
            else:
                weights.append(0.5)

    weighted_sum = 0.0
    weight_total = 0.0

    idx = 0

    for offset in (-2, -1, 0, 1, 2):

        j = index + offset

        if 0 <= j < len(points):

            weighted_sum += scores[idx] * weights[idx]
            weight_total += weights[idx]

            idx += 1

    if weight_total == 0:
        return 0.0

    return weighted_sum / weight_total


# ============================================================
# RAIN WINDOW DETECTION
# ============================================================

def find_rain_windows(
    points: List[WeatherPoint],
    config: Dict[str, Any],
) -> List[Window]:

    if not points:
        return []

    thresholds = config.get("thresholds", {})

    probability_threshold = float(
        thresholds.get(
            "rain_probability_percent",
            40
        )
    )

    amount_threshold = float(
        thresholds.get(
            "rain_amount_mm",
            0.10
        )
    )

    signal_threshold = float(
        thresholds.get(
            "rain_signal_threshold",
            35
        )
    )

    # --------------------------------------------------------
    # First pass
    # --------------------------------------------------------

    rain_indices: List[int] = []

    signals: Dict[int, float] = {}

    for i, point in enumerate(points):

        signal = calculate_rain_signal(
            point,
            config
        )

        signals[i] = signal

        actual_rain = (
            point.precipitation_mm >= amount_threshold
            or point.rain_mm >= amount_threshold
        )

        probable_rain = (
            point.precipitation_probability
            >= probability_threshold
        )

        weather_code_rain = is_rain_code(
            point.weather_code
        )

        if (
            signal >= signal_threshold
            or actual_rain
            or (
                probable_rain
                and weather_code_rain
            )
        ):
            rain_indices.append(i)

    # --------------------------------------------------------
    # Neighbor confirmation
    # --------------------------------------------------------

    confirmed: List[int] = []

    for i in rain_indices:

        point = points[i]

        actual_rain = (
            point.precipitation_mm >= amount_threshold
            or point.rain_mm >= amount_threshold
        )

        strong_probability = (
            point.precipitation_probability >= 70
        )

        code_support = is_rain_code(
            point.weather_code
        )

        left_support = (
            i > 0
            and signals.get(i - 1, 0) >= 30
        )

        right_support = (
            i < len(points) - 1
            and signals.get(i + 1, 0) >= 30
        )

        # Actual precipitation doesn't need a neighbor.
        if actual_rain:
            confirmed.append(i)
            continue

        # Strong forecast + rain code.
        if strong_probability and code_support:
            confirmed.append(i)
            continue

        # Neighbor-supported shower.
        if left_support or right_support:
            confirmed.append(i)
            continue

    rain_indices = sorted(set(confirmed))

    if not rain_indices:
        return []

    # --------------------------------------------------------
    # Convert indices to windows
    # --------------------------------------------------------

    groups: List[List[int]] = []

    current_group = [rain_indices[0]]

    for index in rain_indices[1:]:

        previous = current_group[-1]

        gap_hours = (
            points[index].time -
            points[previous].time
        ).total_seconds() / 3600

        # Merge consecutive hours.
        if gap_hours <= 1.01:

            current_group.append(index)

        # Also merge a short one-hour gap.
        elif gap_hours <= 2.01:

            middle = previous + 1

            if middle < len(points):

                middle_signal = signals.get(
                    middle,
                    calculate_rain_signal(
                        points[middle],
                        config
                    )
                )

                if middle_signal >= 20:

                    current_group.append(index)

                else:

                    groups.append(current_group)
                    current_group = [index]

            else:

                groups.append(current_group)
                current_group = [index]

        else:

            groups.append(current_group)
            current_group = [index]

    groups.append(current_group)

    # --------------------------------------------------------
    # Build windows
    # --------------------------------------------------------

    windows: List[Window] = []

    for group in groups:

        first = group[0]
        last = group[-1]

        start = points[first].time

        # Forecast points are hourly.
        end = points[last].time + timedelta(hours=1)

        group_points = [
            points[i]
            for i in group
        ]

        probabilities = [
            p.precipitation_probability
            for p in group_points
        ]

        precipitation = [
            p.precipitation_mm
            for p in group_points
        ]

        group_signals = [
            signals[i]
            for i in group
        ]

        base_score = max(group_signals)

        # Consistency bonus
        strong_count = sum(
            1
            for s in group_signals
            if s >= 50
        )

        if strong_count >= 2:
            base_score += 5

        # Actual precipitation bonus
        actual_precip_count = sum(
            1
            for p in group_points
            if p.precipitation_mm > 0
        )

        if actual_precip_count >= 2:
            base_score += 5

        confidence = int(
            max(
                0,
                min(
                    98,
                    round(base_score)
                )
            )
        )

        duration_minutes = int(
            (end - start).total_seconds() / 60
        )

        windows.append(
            Window(
                start=start,
                end=end,
                duration_minutes=duration_minutes,
                confidence_percent=confidence,
                confidence_label=confidence_label(
                    confidence
                ),
                max_rain_probability=max(
                    probabilities,
                    default=0.0
                ),
                total_precipitation_mm=sum(
                    precipitation
                ),
                max_precipitation_mm=max(
                    precipitation,
                    default=0.0
                ),
            )
        )

    return windows


# ============================================================
# SUNNY WINDOW DETECTION
# ============================================================

def find_sunny_windows(
    points: List[WeatherPoint],
    config: Dict[str, Any],
) -> List[Window]:

    if not points:
        return []

    thresholds = config.get("thresholds", {})

    sunny_cloud_threshold = float(
        thresholds.get(
            "sunny_cloud_cover_percent",
            45
        )
    )

    sunny_rain_probability = float(
        thresholds.get(
            "sunny_rain_probability_percent",
            25
        )
    )

    sunny_signal_threshold = float(
        thresholds.get(
            "sunny_signal_threshold",
            55
        )
    )

    sunny_indices: List[int] = []

    signals: Dict[int, float] = {}

    for i, point in enumerate(points):

        signal = calculate_sunny_signal(
            point,
            config
        )

        signals[i] = signal

        acceptable_cloud = (
            point.cloud_cover
            <= sunny_cloud_threshold
        )

        low_rain_probability = (
            point.precipitation_probability
            <= sunny_rain_probability
        )

        no_rain = (
            point.precipitation_mm <= 0.05
            and point.rain_mm <= 0.05
        )

        clear_code = is_clear_code(
            point.weather_code
        )

        if (
            signal >= sunny_signal_threshold
            and acceptable_cloud
            and low_rain_probability
            and no_rain
        ):

            sunny_indices.append(i)

        elif (
            clear_code
            and acceptable_cloud
            and low_rain_probability
            and no_rain
        ):

            sunny_indices.append(i)

    if not sunny_indices:
        return []

    # --------------------------------------------------------
    # Group consecutive sunny hours
    # --------------------------------------------------------

    groups: List[List[int]] = []

    current_group = [sunny_indices[0]]

    for index in sunny_indices[1:]:

        previous = current_group[-1]

        gap_hours = (
            points[index].time -
            points[previous].time
        ).total_seconds() / 3600

        if gap_hours <= 1.01:

            current_group.append(index)

        elif gap_hours <= 2.01:

            middle = previous + 1

            if middle < len(points):

                middle_point = points[middle]

                if (
                    middle_point.cloud_cover <= 60
                    and middle_point.precipitation_probability <= 35
                    and middle_point.precipitation_mm <= 0.10
                ):
                    current_group.append(index)

                else:
                    groups.append(current_group)
                    current_group = [index]

            else:
                groups.append(current_group)
                current_group = [index]

        else:

            groups.append(current_group)
            current_group = [index]

    groups.append(current_group)

    # --------------------------------------------------------
    # Build windows
    # --------------------------------------------------------

    windows: List[Window] = []

    for group in groups:

        first = group[0]
        last = group[-1]

        start = points[first].time
        end = points[last].time + timedelta(hours=1)

        group_points = [
            points[i]
            for i in group
        ]

        group_signals = [
            signals[i]
            for i in group
        ]

        probabilities = [
            p.precipitation_probability
            for p in group_points
        ]

        confidence = int(
            max(
                0,
                min(
                    98,
                    round(
                        sum(group_signals)
                        / len(group_signals)
                    )
                )
            )
        )

        duration_minutes = int(
            (end - start).total_seconds() / 60
        )

        windows.append(
            Window(
                start=start,
                end=end,
                duration_minutes=duration_minutes,
                confidence_percent=confidence,
                confidence_label=confidence_label(
                    confidence
                ),
                max_rain_probability=max(
                    probabilities,
                    default=0.0
                ),
                total_precipitation_mm=0.0,
                max_precipitation_mm=0.0,
            )
        )

    return windows


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram_message(
    message: str,
    config: Dict[str, Any],
    chat_id: Optional[str] = None,
) -> bool:

    tg_cfg = config.get("telegram", {})

    if not tg_cfg.get("enabled", False):
        return False

    token = (
        os.getenv("TELEGRAM_BOT_TOKEN")
        or tg_cfg.get("bot_token")
    )

    if not token:
        print("Telegram token not configured.")
        return False

    if chat_id is None:
        chat_id = (
            os.getenv("TELEGRAM_CHAT_ID")
            or tg_cfg.get("chat_id")
        )

    if not chat_id:
        print("Telegram chat ID not configured.")
        return False

    url = (
        f"https://api.telegram.org/bot{token}/sendMessage"
    )

    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    data = urllib.parse.urlencode(
        payload
    ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type":
                "application/x-www-form-urlencoded"
        },
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=20
        ) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )

        return bool(
            result.get("ok", False)
        )

    except Exception as exc:

        print(
            f"Telegram error: {exc}"
        )

        return False


# ============================================================
# TELEGRAM UPDATES
# ============================================================

def get_and_confirm_telegram_updates(
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:

    tg_cfg = config.get("telegram", {})

    token = (
        os.getenv("TELEGRAM_BOT_TOKEN")
        or tg_cfg.get("bot_token")
    )

    if not token:
        return []

    offset_path = tg_cfg.get(
        "update_offset_path",
        "data/telegram_offset.json"
    )

    offset = 0

    if os.path.exists(offset_path):

        try:

            with open(
                offset_path,
                "r"
            ) as f:

                stored = json.load(f)

                offset = int(
                    stored.get(
                        "offset",
                        0
                    )
                )

        except Exception:
            offset = 0

    url = (
        f"https://api.telegram.org/bot{token}/getUpdates"
        f"?timeout=5&offset={offset}"
    )

    try:

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent":
                    "WeatherDetector/2.0"
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=15
        ) as response:

            data = json.loads(
                response.read().decode("utf-8")
            )

    except Exception as exc:

        print(
            f"Telegram update error: {exc}"
        )

        return []

    if not data.get("ok"):
        return []

    updates = data.get(
        "result",
        []
    )

    if updates:

        latest_update_id = max(
            u.get(
                "update_id",
                0
            )
            for u in updates
        )

        new_offset = latest_update_id + 1

        os.makedirs(
            os.path.dirname(offset_path)
            or ".",
            exist_ok=True
        )

        try:

            with open(
                offset_path,
                "w"
            ) as f:

                json.dump(
                    {
                        "offset":
                            new_offset
                    },
                    f,
                    indent=2
                )

        except OSError:
            pass

    return updates
