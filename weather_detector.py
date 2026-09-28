"""
weather_detector.py

Command-line entry point for the Weather Detector project.

Usage:
python weather_detector.py
python weather_detector.py --config rain_prediction.yml
python weather_detector.py --lat 23.5204 --lon 87.3119 --name "Test Town"

Reads settings from rain_prediction.yml -- including a list of any number of
cities/townships -- fetches the hourly forecast for all of them (batched),
and for each one prints:
- Rain windows: exact predicted start time, end time, and duration
- Sunny windows: exact predicted start time, end time, and duration
- A confidence score per window (see prediction_weather.py for how this
is calculated -- it is NOT a fixed accuracy claim)

If output.save_results_json is enabled, also writes a combined machine-
readable summary to output.results_path (default: results/latest.json),
so other automation steps (a notification script, a dashboard, a GitHub
Action that commits history) can consume the results without re-parsing
console output.
"""

import argparse
import json
import os
import sys
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

def load_config(path: str) -> Dict[str, Any]:
with open(path, "r") as f:
config = yaml.safe_load(f)

# The city list can either be inline under "locations" (legacy), or
# loaded from a separate JSON file via "locations_file" -- the latter
# is the recommended setup so the city list can grow freely without
# touching this YAML file.
if "locations_file" in config:
    locations_path = config["locations_file"]
    try:
        with open(locations_path, "r") as f:
            config["locations"] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load locations_file '{locations_path}': {exc}") from exc

required_top_level = ["locations", "api", "thresholds", "output"]
for key in required_top_level:
    if key not in config:
        raise ValueError(f"Config file is missing required section: '{key}'")

return config

def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
"""If --lat/--lon are passed, run for just that one ad-hoc location
instead of the full list in the config file (handy for quick tests)."""
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
"""
Build a single Telegram message covering every city that has a
qualifying alert (matching telegram.alert_on kinds, at or above
telegram.min_confidence_percent). Returns "" if nothing qualifies.
"""
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
            from datetime import datetime as _dt
            start = _dt.fromisoformat(w["start"])
            end = _dt.fromisoformat(w["end"])
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

def build_summary_message(all_results: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
"""
Build a Telegram message covering EVERY run, regardless of confidence
threshold -- the nearest upcoming rain window and sunny window for each
city, so you get a real weather digest every time the workflow runs,
not just when something crosses the alert threshold.
"""
time_fmt = config["output"].get("time_format", "%I:%M %p")
lines: List[str] = ["\U0001F324 Weather Update"]

for city in all_results:
    lines.append(f"\n<b>{city['name']}</b>")

    if city["rain_windows"]:
        w = city["rain_windows"][0]
        start = datetime.fromisoformat(w["start"])
        end = datetime.fromisoformat(w["end"])
        lines.append(
            f"  Rain: {start.strftime(time_fmt)} -> {end.strftime(time_fmt)} "
            f"({w['confidence_percent']}% {w['confidence_label']})"
        )
    else:
        lines.append("  Rain: none predicted in the forecast period.")

    if city["sunny_windows"]:
        w = city["sunny_windows"][0]
        start = datetime.fromisoformat(w["start"])
        end = datetime.fromisoformat(w["end"])
        lines.append(
            f"  Sunny: {start.strftime(time_fmt)} -> {end.strftime(time_fmt)} "
            f"({w['confidence_percent']}% {w['confidence_label']})"
        )
    else:
        lines.append("  Sunny: none predicted in the forecast period.")

return "\n".join(lines)

def save_results_json(all_results: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
path = config["output"].get("results_path", "results/latest.json")
os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
with open(path, "w") as f:
json.dump({
"generated_at": datetime.now().isoformat(),
"cities": all_results,
}, f, indent=2)
print(f"\nSaved machine-readable results to: {path}")

--------------------------------------------------------------------------

On-demand "which city am I in" replies

--------------------------------------------------------------------------

def match_cities(query: str, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
"""Match a user's texted city/township name against the configured list.
Tries an exact (case-insensitive) match first, then falls back to a
substring match so "durgapur" matches "Durgapur" and typos/partials
still have a chance."""
q = query.strip().lower()
exact = [c for c in all_results if c["name"].strip().lower() == q]
if exact:
return exact
return [c for c in all_results if q in c["name"].lower() or c["name"].lower() in q]

def generate_ai_tip(city_name: str, current: Dict[str, Any], config: Dict[str, Any]) -> str:
"""
Local weather tip fallback.

This keeps weather_detector.py independent of a missing
generate_ai_tip() function in prediction_weather.py.
If prediction_weather.py later provides an AI tip function,
this function can be replaced with that implementation.
"""
try:
    rain_chance = float(current.get("precipitation_probability", 0.0))
    temp = float(current.get("temperature_c", 0.0))
    wind = float(current.get("wind_speed", 0.0))
except (TypeError, ValueError):
    return ""

tips = []

if rain_chance >= 70:
    tips.append("High rain chance. Carry an umbrella and plan outdoor activities carefully.")
elif rain_chance >= 40:
    tips.append("Moderate rain chance. Carry an umbrella if you are going outdoors.")
else:
    tips.append("Low rain chance. Conditions are generally suitable for outdoor activities.")

if temp >= 35:
    tips.append("It is hot, so stay hydrated and avoid prolonged direct sunlight.")
elif temp <= 15:
    tips.append("It is relatively cool, so consider carrying an extra layer.")

if wind >= 35:
    tips.append("Strong winds are possible; use extra caution outdoors.")

return " ".join(tips)

def build_daily_report(city: Dict[str, Any], config: Dict[str, Any]) -> str:
"""
Build a formatted daily weather report for one city, matching this style:

    WEATHER REPORT
    CITY NAME
    Date, Time
    ----------------------------
    Temperature : ..C
    Humidity    : ..%
    Wind Speed  : .. km/h
    Pressure    : .. hPa
    ----------------------------
    Rain Chance : ..%
    Prediction  : RAIN PREDICTED / CLEAR SKIES
    Tip
    ----------------------------
    Sent by GitHub Actions

Uses the "current" snapshot (nearest upcoming hour) attached to each
city's entry in all_results.
"""
tz_name = config.get("timezone", "UTC")
try:
    tz = ZoneInfo(tz_name)
except Exception:
    tz = ZoneInfo("UTC")
now_local = datetime.now(tz)
date_str = now_local.strftime("%d %B %Y, %I:%M %p")

current = city.get("current", {})
temp = current.get("temperature_c", 0.0)
humidity = current.get("humidity", 0.0)
wind = current.get("wind_speed", 0.0)
pressure = current.get("pressure", 0.0)
rain_chance = current.get("precipitation_probability", 0.0)

threshold = config["thresholds"]["rain_probability_percent"]
if rain_chance >= threshold:
    prediction = "\U0001F327\uFE0F RAIN PREDICTED"
    static_tip = "Carry an umbrella. Avoid outdoor plans if possible."
else:
    prediction = "\u2600\uFE0F CLEAR SKIES"
    static_tip = "Good conditions for outdoor plans."

# If Gemini is configured and enabled, use its generated tip instead --
# falls back to the static tip automatically on any failure.
ai_tip = generate_ai_tip(city["name"], current, config)
tip = ai_tip if ai_tip else static_tip

divider = "\u2500" * 28
time_fmt = config["output"].get("time_format", "%I:%M %p")

def format_window_lines(windows: List[Dict[str, Any]]) -> List[str]:
    if not windows:
        return ["  None predicted in the forecast period."]
    today = now_local.date()
    tomorrow = today + timedelta(days=1)
    now_naive = now_local.replace(tzinfo=None)
    out = []
    for w in windows:
        start = datetime.fromisoformat(w["start"])
        end = datetime.fromisoformat(w["end"])
        hours, minutes = divmod(w["duration_minutes"], 60)
        dur = f"{hours}h" + (f" {minutes}m" if minutes else "")

        if start.date() == today:
            day_label = "Today"
        elif start.date() == tomorrow:
            day_label = "Tomorrow"
        else:
            day_label = start.strftime("%A, %d %b")

        same_day_end = " (into next day)" if end.date() != start.date() else ""

        if start <= now_naive <= end:
            remaining = int((end - now_naive).total_seconds() // 60)
            rh, rm = divmod(remaining, 60)
            countdown = f"\U0001F534 Happening now -- ends in {rh}h {rm}m"
        elif start > now_naive:
            until = int((start - now_naive).total_seconds() // 60)
            uh, um = divmod(until, 60)
            countdown = f"\u23F3 Starts in {uh}h {um}m"
        else:
            countdown = "Already passed"

        out.append(
            f"  {day_label}\n"
            f"    {start.strftime(time_fmt)} \u2192 {end.strftime(time_fmt)}{same_day_end}\n"
            f"    Duration: {dur} | Confidence: {w['confidence_percent']}% ({w['confidence_label']})\n"
            f"    {countdown}"
        )
    return out

lines = [
    "\U0001F326\uFE0F DAILY WEATHER REPORT",
    f"\U0001F4CD {city['name'].upper()}",
    f"\U0001F4C5 {date_str}",
    divider,
    "",
    f"\U0001F321\uFE0F Temperature : {temp:.1f}\u00B0C",
    f"\U0001F4A7 Humidity    : {humidity:.0f}%",
    f"\U0001F32C\uFE0F Wind Speed  : {wind:.1f} km/h",
    f"\U0001F535 Pressure    : {pressure:.1f} hPa",
    divider,
    "",
    f"\U0001F4CA Rain Chance : {rain_chance:.1f}%",
    f"\U0001F52E Prediction  : {prediction}",
    "",
    f"\U0001F4AC {tip}",
    divider,
    "\u23F1\uFE0F Rain Windows:",
]
lines.extend(format_window_lines(city.get("rain_windows", [])))
lines.append("")
lines.append("\u2600\uFE0F Sunny Windows:")
lines.extend(format_window_lines(city.get("sunny_windows", [])))
lines.append(divider)
lines.append("\U0001F916 Sent by GitHub Actions")

return "\n".join(lines)

def build_city_reply(city: Dict[str, Any], config: Dict[str, Any]) -> str:
"""Build the on-demand forecast reply text for a single city."""
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

def load_subscriptions(path: str) -> Dict[str, str]:
"""Load the chat_id -> city_name subscription map from disk.
Returns an empty dict if the file doesn't exist yet (first run ever)."""
if not os.path.exists(path):
return {}
try:
with open(path, "r") as f:
return json.load(f)
except (json.JSONDecodeError, OSError):
return {}

def save_subscriptions(path: str, subscriptions: Dict[str, str]) -> None:
os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
with open(path, "w") as f:
json.dump(subscriptions, f, indent=2)

def handle_subscriptions(all_results: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
"""
Per-user city subscriptions:
- Texting a city name subscribes that chat to updates for that city,
replacing any previous subscription (one active city per chat).
- Every run, every currently subscribed chat gets that city's forecast.
- This keeps happening automatically, run after run, until the user
texts a different (valid) city name to switch.

The subscription map persists in a JSON file that gets committed back
to the repo after each run (see the workflow's git-auto-commit step),
since GitHub Actions runs don't share memory between invocations.
"""
sub_path = config["telegram"].get("subscriptions_path", "data/subscriptions.json")
subscriptions = load_subscriptions(sub_path)
results_by_name = {c["name"]: c for c in all_results}

updates = get_and_confirm_telegram_updates(config)
just_replied_to: set = set()

for update in updates:
    message = update.get("message") or update.get("edited_message")
    if not message or "text" not in message:
        continue

    chat_id = str(message["chat"]["id"])
    text = message["text"]
    matches = match_cities(text, all_results)

    if len(matches) == 1:
        city = matches[0]
        subscriptions[chat_id] = city["name"]
        reply = (
            f"Subscribed! You'll get {city['name']} weather updates every "
            f"~5 minutes. Text another city name anytime to switch.\n\n"
            + build_daily_report(city, config)
        )
        send_telegram_message(reply, config, chat_id=chat_id)
        just_replied_to.add(chat_id)
    elif len(matches) > 1:
        names = ", ".join(m["name"] for m in matches)
        send_telegram_message(
            f"That matches more than one place: {names}. Please send the exact name.",
            config, chat_id=chat_id,
        )
    else:
        shown = [c["name"] for c in all_results[:20]]
        extra = f" (+{len(all_results) - 20} more)" if len(all_results) > 20 else ""
        send_telegram_message(
            "I don't recognize that place. Text the exact name of your city "
            f"or township to subscribe. Available: {', '.join(shown)}{extra}",
            config, chat_id=chat_id,
        )

# Recurring update for everyone already subscribed (skip anyone we just
# replied to above, since their subscribe-confirmation already included
# the current forecast -- no need to send it twice in the same run).
for chat_id, city_name in list(subscriptions.items()):
    if chat_id in just_replied_to:
        continue
    city = results_by_name.get(city_name)
    if not city:
        continue  # subscribed city no longer in rain_prediction.yml -- skip quietly
    sent = send_telegram_message(build_daily_report(city, config), config, chat_id=chat_id)
    print(f"Recurring update sent to chat {chat_id} ({city_name}): {sent}")

save_subscriptions(sub_path, subscriptions)

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

    current = points[0]  # nearest upcoming hour -- used as "right now" conditions

    all_results.append({
        "name": name,
        "country": loc.get("country", ""),
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "rain_windows": [w.to_dict() for w in rain_windows],
        "sunny_windows": [w.to_dict() for w in sunny_windows],
        "current": {
            "time": current.time.isoformat(),
            "temperature_c": current.temperature_c,
            "humidity": current.humidity,
            "wind_speed": current.wind_speed,
            "pressure": current.pressure,
            "precipitation_probability": current.precipitation_probability,
        },
    })

if config["output"].get("save_results_json", True):
    save_results_json(all_results, config)

if config.get("telegram", {}).get("enabled", False):
    tg_cfg = config["telegram"]

    if tg_cfg.get("always_send_summary", False):
        # Full digest every run, regardless of confidence threshold.
        message = build_summary_message(all_results, config)
    else:
        # Only message when something crosses the confidence threshold.
        message = build_alert_message(all_results, config)

    if message:
        sent = send_telegram_message(message, config)
        print(f"\nTelegram message sent: {sent}")
    else:
        print("\nNo qualifying alerts -- Telegram message not sent.")

    if tg_cfg.get("respond_to_city_requests", False):
        handle_subscriptions(all_results, config)

if config["output"].get("show_disclaimer", True):
    print_disclaimer()

return 0

if name == "main":
sys.exit(main())    make it more and more accurate
