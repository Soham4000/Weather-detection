"""
weather_detector.py

Command-line entry point for the Weather Detector project.

Usage:
    python weather_detector.py
    python weather_detector.py --config rain_prediction.yml
    python weather_detector.py --lat 23.5204 --lon 87.3119 --name "Test Town"
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

from prediction_weather import (
    fetch_forecast_multi,
    find_rain_windows,
    find_sunny_windows,
    send_telegram_message,
    get_and_confirm_telegram_updates,
    Window,
)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    required_top_level = ["locations", "api", "thresholds", "output"]
    for key in required_top_level:
        if key not in config:
            raise ValueError(f"Config file is missing required section: '{key}'")

    return config


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.lat is not None and args.lon is not None:
        config["locations"] = [{
            "name": args.name or "Custom Location",
            "country": "",
            "latitude": args.lat,
            "longitude": args.lon,
        }]


def print_windows(title: str, windows: List[Window], config: Dict[str, Any]) -> None:
    time_fmt = config["output"].get("time_format", "%I:%M %p")
    print(f"  {title}:")
    if not windows:
        print("    None predicted in the forecast period.")
        return
    for w in windows:
        start_str = w.start.strftime(time_fmt)
        end_str = w.end.strftime(time_fmt)
        date_str = w.start.strftime("%a %d %b")
        print(
            f"    {date_str}: {start_str} -> {end_str} "
            f"(duration: {w.duration_str()}) "
            f"-- {w.confidence_label} ({w.confidence_percent}%)"
        )


def print_hourly_detail(points, config: Dict[str, Any]) -> None:
    time_fmt = config["output"].get("time_format", "%I:%M %p")
    for p in points:
        print(
            f"    {p.time.strftime('%a %d %b ' + time_fmt)} | "
            f"temp: {p.temperature_c:>5.1f}C | "
            f"rain chance: {p.precipitation_probability:>3.0f}% | "
            f"precip: {p.precipitation_mm:>4.1f}mm | "
            f"cloud cover: {p.cloud_cover:>3.0f}%"
        )


def print_disclaimer() -> None:
    print("\nNote on accuracy")
    print("-" * 17)
    print(
        "  Forecast confidence naturally drops the further out a window is.\n"
        "  Treat 'confidence' above as how strong and consistent the signal\n"
        "  is in the forecast data, not a guaranteed accuracy rate. No\n"
        "  weather tool -- this one included -- can promise a fixed 90%\n"
        "  accuracy on exact rain timing."
    )


def build_alert_message(all_results: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
    tg_cfg = config.get("telegram", {})
    alert_kinds = set(tg_cfg.get("alert_on", ["rain"]))
    min_confidence = tg_cfg.get("min_confidence_percent", 60)
    time_fmt = config["output"].get("time_format", "%I:%M %p")

    lines: List[str] = []

    for city in all_results:
        city_lines = []
        for kind_key, label in (("rain_windows", "Rain"), ("sunny_windows", "Sunny")):
            kind_name = "rain" if label == "Rain" else "sunny"
            if kind_name not in alert_kinds:
                continue
            for w in city[kind_key]:
                if w["confidence_percent"] < min_confidence:
                    continue
                start = datetime.fromisoformat(w["start"])
                end = datetime.fromisoformat(w["end"])
                city_lines.append(
                    f"  {label}: {start.strftime(time_fmt)} -> {end.strftime(time_fmt)} "
                    f"({w['confidence_percent']}% {w['confidence_label']})"
                )
        if city_lines:
            lines.append(f"<b>{city['name']}</b>")
            lines.extend(city_lines)

    if not lines:
        return ""

    header = "\u26c5 Weather Alert\n"
    return header + "\n".join(lines)


def save_results_json(all_results: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
    path = config["output"].get("results_path", "results/latest.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "cities": all_results,
        }, f, indent=2)
    print(f"\nSaved machine-readable results to: {path}")


def match_cities(query: str, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    q = query.strip().lower()
    exact = [c for c in all_results if c["name"].strip().lower() == q]
    if exact:
        return exact
    return [c for c in all_results if q in c["name"].lower() or c["name"].lower() in q]


def build_city_reply(city: Dict[str, Any], config: Dict[str, Any]) -> str:
    time_fmt = config["output"].get("time_format", "%I:%M %p")
    lines = [f"<b>{city['name']}</b>"]

    for key, label in (("rain_windows", "Rain"), ("sunny_windows", "Sunny")):
        windows = city[key]
        if not windows:
            lines.append(f"{label}: none predicted in the forecast period.")
            continue
        lines.append(f"{label}:")
        for w in windows:
            start = datetime.fromisoformat(w["start"])
            end = datetime.fromisoformat(w["end"])
            lines.append(
                f"  {start.strftime(time_fmt)} -> {end.strftime(time_fmt)} "
                f"({w['confidence_percent']}% {w['confidence_label']})"
            )

    return "\n".join(lines)


def handle_city_requests(all_results: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
    updates = get_and_confirm_telegram_updates(config)
    if not updates:
        return

    for update in updates:
        message = update.get("message") or update.get("edited_message")
        if not message or "text" not in message:
            continue

        chat_id = message["chat"]["id"]
        text = message["text"]
        matches = match_cities(text, all_results)

        if len(matches) == 1:
            reply = build_city_reply(matches[0], config)
        elif len(matches) > 1:
            names = ", ".join(m["name"] for m in matches)
            reply = f"That matches more than one place: {names}. Please send the exact name."
        else:
            shown = [c["name"] for c in all_results[:20]]
            extra = f" (+{len(all_results) - 20} more)" if len(all_results) > 20 else ""
            reply = (
                "I don't recognize that place. Just text the name of your city "
                f"or township. Available: {', '.join(shown)}{extra}"
            )

        sent = send_telegram_message(reply, config, chat_id=chat_id)
        print(f"Replied to chat {chat_id}: {sent}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Weather Detector: rain & sunny window predictor for many cities")
    parser.add_argument("--config", default="rain_prediction.yml", help="Path to config YAML file")
    parser.add_argument("--lat", type=float, default=None, help="Override: run for a single ad-hoc latitude")
    parser.add_argument("--lon", type=float, default=None, help="Override: run for a single ad-hoc longitude")
    parser.add_argument("--name", type=str, default=None, help="Override: display name for the ad-hoc location")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        return 1

    apply_overrides(config, args)
    locations = config["locations"]

    print(f"Weather Detector -- {len(locations)} location(s)")
    print("=" * 60)

    try:
        forecasts = fetch_forecast_multi(config)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    all_results: List[Dict[str, Any]] = []

    for loc in locations:
        name = loc["name"]
        points = forecasts.get(name, [])
        print(f"\n{name}")
        print("-" * len(name))

        if not points:
            print("  No data returned for this location.")
            continue

        rain_windows = find_rain_windows(points, config)
        sunny_windows = find_sunny_windows(points, config)

        print_windows("Rain windows", rain_windows, config)
        print_windows("Sunny windows", sunny_windows, config)

        if config["output"].get("verbose", False):
            print_hourly_detail(points, config)

        all_results.append({
            "name": name,
            "country": loc.get("country", ""),
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "rain_windows": [w.to_dict() for w in rain_windows],
            "sunny_windows": [w.to_dict() for w in sunny_windows],
        })

    if config["output"].get("save_results_json", True):
        save_results_json(all_results, config)

    if config.get("telegram", {}).get("enabled", False):
        message = build_alert_message(all_results, config)
        if message:
            sent = send_telegram_message(message, config)
            print(f"\nTelegram alert sent: {sent}")
        elif config["telegram"].get("always_send_summary", False):
            send_telegram_message("No significant weather alerts this run.", config)
            print("\nTelegram summary sent (no alerts).")
        else:
            print("\nNo qualifying alerts -- Telegram message not sent.")

        if config["telegram"].get("respond_to_city_requests", False):
            handle_city_requests(all_results, config)

    if config["output"].get("show_disclaimer", True):
        print_disclaimer()

    return 0


if __name__ == "__main__":
    sys.exit(main())
