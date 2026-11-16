import csv
import json
import os
from datetime import datetime

import redis
import requests

BROADCASTER_API = "http://172.172.1.13:8050/api/v1"
REDIS_HOST = "172.172.1.13"
REDIS_PORT = 6379
MESSAGE_CODE = 1501  # Touchline

EXCHANGE_SEGMENT = 1
INSTRUMENT_ID = 26000
SYMBOL = str(INSTRUMENT_ID)
CSV_FILE = r"data\\live_data_ohlc.csv"

EXCHANGE_SEGMENT_MAP = {
    "NSECM": 1,
    "NSEFO": 2,
    "BSECM": 11,
    "BSEFO": 12,
    "MCXFO": 51,
}


def resolve_segment(raw) -> int:
    if raw is None:
        return 0
    if isinstance(raw, int):
        return raw
    s = str(raw).strip()
    if s.lstrip("-").isdigit():
        return int(s)
    return EXCHANGE_SEGMENT_MAP.get(s, 0)


def subscribe_instrument() -> None:
    payload = {
        "instruments": [
            {
                "exchangeSegment": EXCHANGE_SEGMENT,
                "exchangeInstrumentID": INSTRUMENT_ID,
                "symbol": SYMBOL,
                "subscribedCodes": [MESSAGE_CODE],
            }
        ]
    }
    r = requests.post(f"{BROADCASTER_API}/subscriptions", json=payload, timeout=10)
    r.raise_for_status()
    print(f"Subscribed to segment={EXCHANGE_SEGMENT} token={INSTRUMENT_ID}")


def get_last_tick():
    try:
        r = requests.get(
            f"{BROADCASTER_API}/last-tick/{EXCHANGE_SEGMENT}/{INSTRUMENT_ID}",
            params={"code": MESSAGE_CODE},
            timeout=5,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"Could not fetch last tick: {exc}")
        return None


def tick_matches_instrument(tick: dict) -> bool:
    seg_raw = tick.get("ExchangeSegment", tick.get("exchangeSegment"))
    tok_raw = tick.get("ExchangeInstrumentID", tick.get("exchangeInstrumentID"))
    if seg_raw is None or tok_raw is None:
        return False
    try:
        return (
            resolve_segment(seg_raw) == EXCHANGE_SEGMENT
            and int(tok_raw) == INSTRUMENT_ID
        )
    except (TypeError, ValueError):
        return False


def process_tick(tick: dict, current_minute, candle):
    touchline = tick.get("Touchline")
    if not touchline:
        return current_minute, candle

    ltp = touchline.get("LastTradedPrice")
    if ltp is None:
        return current_minute, candle

    ltp = float(ltp)
    total_volume = int(touchline.get("TotalTradedQuantity", 0))
    minute = datetime.now().replace(second=0, microsecond=0)

    if current_minute is None:
        return minute, {
            "open": ltp,
            "high": ltp,
            "low": ltp,
            "close": ltp,
            "volume": total_volume,
        }

    if minute != current_minute:
        print(
            f"{current_minute} "
            f"O:{candle['open']} "
            f"H:{candle['high']} "
            f"L:{candle['low']} "
            f"C:{candle['close']} "
            f"V:{candle['volume']}"
        )

        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    current_minute.strftime("%Y-%m-%d %H:%M:%S"),
                    candle["open"],
                    candle["high"],
                    candle["low"],
                    candle["close"],
                    candle["volume"],
                ]
            )

        return minute, {
            "open": ltp,
            "high": ltp,
            "low": ltp,
            "close": ltp,
            "volume": total_volume,
        }

    candle["high"] = max(candle["high"], ltp)
    candle["low"] = min(candle["low"], ltp)
    candle["close"] = ltp
    candle["volume"] = total_volume
    return current_minute, candle


# --------------------------------------------------
# CREATE CSV IF NOT EXISTS
# --------------------------------------------------

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])

# --------------------------------------------------
# SUBSCRIBE VIA BROADCASTER API
# --------------------------------------------------

subscribe_instrument()

# --------------------------------------------------
# REDIS (listen on broadcast-all channel)
# --------------------------------------------------

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
ps = r.pubsub()
ps.subscribe("xts:marketdata:all")

print("Listening on xts:marketdata:all")

current_minute = None
candle = None

last_tick = get_last_tick()
if last_tick and tick_matches_instrument(last_tick):
    current_minute, candle = process_tick(last_tick, current_minute, candle)

for msg in ps.listen():
    if msg["type"] != "message":
        continue

    try:
        tick = json.loads(msg["data"])
    except Exception:
        continue

    if not isinstance(tick, dict) or "Touchline" not in tick:
        continue

    if not tick_matches_instrument(tick):
        continue

    current_minute, candle = process_tick(tick, current_minute, candle)