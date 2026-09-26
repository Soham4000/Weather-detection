"""
prediction_weather.py

Core forecasting logic for the Weather Detector project.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any

import os

import requests
from dateutil import parser as dateparser


@dataclass
class HourlyPoint:
    time: datetime
    temperature_c: float
    precipitation_probability: float
    precipitation_mm: float
    cloud_cover: float
    weather_code: int


@dataclass
class Window:
    start: datetime
    end: datetime
    kind: str
    avg_value: float
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
]


def _parse_hourly_block(hourly: Dict[str, Any]) -> List[HourlyPoint]:
    times = hourly["time"]
    temps = hourly["temperature_2m"]
    precip_prob = hourly["precipitation_probability"]
    precip_mm = hourly["precipitation"]
    cloud_cover = hourly["cloud_cover"]
    weather_code = hourly["weather_code"]

    points: List[HourlyPoint] = []
    for i, t in enumerate(times):
        points.append(HourlyPoint(
            time=dateparser.parse(t),
            temperature_c=temps[i],
            precipitation_probability=precip_prob[i],
            precipitation_mm=precip_mm[i],
            cloud_cover=cloud_cover[i],
            weather_code=weather_code[i],
        ))
    return points


def _fetch_batch(batch: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, List[HourlyPoint]]:
    api_cfg = config["api"]
    lats = ",".join(str(loc["latitude"]) for loc in batch)
    lons = ",".join(str(loc["longitude"]) for loc in batch)

    params = {
        "latitude": lats,
        "longitude": lons,
        "timezone": config.get("timezone", "auto"),
        "forecast_days": api_cfg.get("forecast_days", 2),
        "hourly": ",".join(HOURLY_VARS),
    }

    try:
        resp = requests.get(api_cfg["base_url"], params=params, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch forecast data: {exc}") from exc

    data = resp.json()
    if isinstance(data, dict):
        data = [data]

    results: Dict[str, List[HourlyPoint]] = {}
    for loc, entry in zip(batch, data):
        hourly = entry.get("hourly")
        results[loc["name"]] = _parse_hourly_block(hourly) if hourly else []
    return results


def fetch_forecast_multi(config: Dict[str, Any]) -> Dict[str, List[HourlyPoint]]:
    locations = config["locations"]
    if not locations:
        raise ValueError("Config 'locations' list is empty -- add at least one city.")

    batch_size = config["api"].get("batch_size", 50)
    results: Dict[str, List[HourlyPoint]] = {}

    for i in range(0, len(locations), batch_size):
        batch = locations[i:i + batch_size]
        results.update(_fetch_batch(batch, config))

    return results


def _confidence_from_values(values: List[float]):
    if not values:
        return 0.0, "no data"

    avg = sum(values) / len(values)
    variance = sum((v - avg) ** 2 for v in values) / len(values)
    consistency_penalty = min(variance ** 0.5, 30)

    raw_confidence = avg - consistency_penalty
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
        confidence, label = _confidence_from_values(values)
        windows.append(Window(
            start=start, end=end, kind=kind,
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
        kind="rain", min_window_minutes=min_minutes,
    )


def find_sunny_windows(points: List[HourlyPoint], config: Dict[str, Any]) -> List[Window]:
    threshold = config["thresholds"]["sunny_cloud_cover_percent"]
    min_minutes = config["thresholds"]["min_window_minutes"]

    def is_sunny(p: HourlyPoint) -> bool:
        return p.cloud_cover <= threshold and p.weather_code in (0, 1)

    return _merge_into_windows(
        points,
        is_match_fn=is_sunny,
        value_fn=lambda p: 100 - p.cloud_cover,
        kind="sunny", min_window_minutes=min_minutes,
    )


# --- Telegram alerts: token/chat ID come ONLY from environment variables ---

def send_telegram_message(text: str, config: Dict[str, Any], chat_id: str = None) -> bool:
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
            requests.get(url, params={"offset": max_update_id + 1, "timeout": 0}, timeout=15)
        except requests.RequestException:
            pass

    return updates
