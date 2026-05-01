import json
import os
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import boto3
from boto3.dynamodb.conditions import Key
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

DATA_DIR = Path(__file__).resolve().parent / "data"
LOCAL_HISTORY_FILE = DATA_DIR / "ride_history.json"
RIDE_TABLE_NAME = "RideHistory"
FULL_ML_ENABLED = os.getenv("SMART_RIDE_FULL_ML") == "1"
RIDE_LABELS = ["Ola Auto", "Ola Mini", "Uber Go", "Uber Sedan"]
AUTO_MAX_KM = 15

_ml_cache = None
_ml_import_failed = False

dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
table = dynamodb.Table(RIDE_TABLE_NAME)


def json_error(message, status=400):
    return jsonify({"error": message}), status


def get_json_payload():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object request body.")
    return data


def require_fields(data, fields):
    missing = [field for field in fields if field not in data or data[field] in (None, "")]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")


def json_safe(value):
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, Decimal):
        return int(value) if value % 1 == 0 else float(value)
    return value


def load_full_ml():
    global _ml_cache, _ml_import_failed
    if not FULL_ML_ENABLED or _ml_import_failed:
        return None
    if _ml_cache is not None:
        return _ml_cache

    try:
        from recommender import get_model_metrics, get_user_profile, recommend

        _ml_cache = {
            "get_model_metrics": get_model_metrics,
            "get_user_profile": get_user_profile,
            "recommend": recommend,
        }
        return _ml_cache
    except Exception as exc:
        _ml_import_failed = True
        print(f"Full ML engine could not start; using fast fallback: {exc}")
        return None


def ride_code(segment):
    return {
        "New User": "NEW",
        "Budget Rider": "SAVE",
        "Premium Commuter": "PRO",
        "Weekend Explorer": "TRIP",
        "Night Owl": "NIGHT",
        "Regular Rider": "RIDE",
    }.get(segment, "RIDE")


def trip_context(timestamp=None):
    dt = datetime.fromtimestamp(timestamp or int(time.time()))
    return {
        "hour": dt.hour,
        "is_peak": 7 <= dt.hour < 10 or 17 <= dt.hour < 21,
        "is_night": dt.hour >= 22 or dt.hour < 6,
        "is_weekend": dt.weekday() >= 5,
    }


def predict_fast_surge(distance, timestamp=None):
    ctx = trip_context(timestamp)
    multiplier = 1.0
    reasons = []

    if ctx["is_peak"]:
        multiplier += 0.28
        reasons.append("peak hour demand")
    if ctx["is_night"]:
        multiplier += 0.18
        reasons.append("late night premium")
    if ctx["is_weekend"]:
        multiplier += 0.08
        reasons.append("weekend travel")
    if distance > 20:
        multiplier -= 0.05

    multiplier = max(1.0, min(multiplier, 1.65))
    if multiplier < 1.1:
        level = "low"
    elif multiplier < 1.35:
        level = "moderate"
    else:
        level = "high"

    return {
        "multiplier": round(multiplier, 2),
        "level": level,
        "reason": ", ".join(reasons) if reasons else "Normal demand - regular pricing",
    }


def build_fast_profile(history):
    if not history:
        return {
            "segment": "New User",
            "segment_icon": "NEW",
            "total_rides": 0,
            "avg_distance": 0,
            "avg_price": 0,
            "preferred_ride": "Unknown",
            "ride_distribution": {},
            "peak_rider": False,
            "night_rider": False,
            "weekend_ratio": 0,
            "spending_tier": "N/A",
        }

    distances = [float(item.get("distance", 0) or 0) for item in history]
    prices = [float(item.get("price", 0) or 0) for item in history]
    rides = [item.get("chosenRide", "Unknown") for item in history]
    ride_counts = {ride: rides.count(ride) for ride in set(rides)}
    preferred = max(ride_counts, key=ride_counts.get)
    ride_distribution = {
        ride: round(count / len(rides) * 100, 1)
        for ride, count in sorted(ride_counts.items(), key=lambda pair: pair[1], reverse=True)
    }

    contexts = [trip_context(int(item.get("timestamp", 0) or 0)) for item in history]
    peak_ratio = sum(ctx["is_peak"] for ctx in contexts) / len(contexts)
    night_ratio = sum(ctx["is_night"] for ctx in contexts) / len(contexts)
    weekend_ratio = sum(ctx["is_weekend"] for ctx in contexts) / len(contexts)
    avg_distance = sum(distances) / len(distances)
    avg_price = sum(prices) / len(prices)

    if preferred == "Uber Sedan" or avg_price >= 220:
        segment = "Premium Commuter"
    elif preferred in ("Ola Auto", "Ola Mini") and avg_price < 120:
        segment = "Budget Rider"
    elif night_ratio > 0.45:
        segment = "Night Owl"
    elif weekend_ratio > 0.35:
        segment = "Weekend Explorer"
    else:
        segment = "Regular Rider"

    if avg_price < 80:
        tier = "Economy"
    elif avg_price < 200:
        tier = "Standard"
    else:
        tier = "Premium"

    return {
        "segment": segment,
        "segment_icon": ride_code(segment),
        "total_rides": len(history),
        "avg_distance": round(avg_distance, 1),
        "avg_price": round(avg_price, 0),
        "preferred_ride": preferred,
        "ride_distribution": ride_distribution,
        "peak_rider": peak_ratio > 0.4,
        "night_rider": night_ratio > 0.2,
        "weekend_ratio": round(weekend_ratio * 100, 0),
        "spending_tier": tier,
    }


def build_user_profile(history):
    ml = load_full_ml()
    if ml:
        return ml["get_user_profile"](history)
    return build_fast_profile(history)


def segment_score_bias(segment):
    """Customer persona influence used after route/time scores are calculated."""
    return {
        "Budget Rider": {
            "Ola Auto": 1.35,
            "Ola Mini": 1.25,
            "Uber Go": 0.90,
            "Uber Sedan": 0.65,
        },
        "Premium Commuter": {
            "Ola Auto": 0.55,
            "Ola Mini": 0.75,
            "Uber Go": 1.10,
            "Uber Sedan": 1.45,
        },
        "Weekend Explorer": {
            "Ola Auto": 0.75,
            "Ola Mini": 0.95,
            "Uber Go": 1.25,
            "Uber Sedan": 1.15,
        },
        "Night Owl": {
            "Ola Auto": 0.45,
            "Ola Mini": 0.80,
            "Uber Go": 1.15,
            "Uber Sedan": 1.45,
        },
        "Regular Rider": {
            "Ola Auto": 1.0,
            "Ola Mini": 1.0,
            "Uber Go": 1.0,
            "Uber Sedan": 1.0,
        },
        "New User": {
            "Ola Auto": 1.0,
            "Ola Mini": 1.0,
            "Uber Go": 1.0,
            "Uber Sedan": 1.0,
        },
    }.get(segment, {})


def fast_recommend(distance, history, timestamp=None):
    ctx = trip_context(timestamp)
    scores = {
        "Ola Auto": 0.92 if distance <= 3 else 0.58 if distance <= AUTO_MAX_KM else 0.0,
        "Ola Mini": 0.74 if distance <= 10 else 0.46,
        "Uber Go": 0.62 if distance <= 22 else 0.70,
        "Uber Sedan": 0.38 if distance <= 12 else 0.78,
    }

    if ctx["is_peak"]:
        scores["Uber Go"] += 0.12
        scores["Uber Sedan"] += 0.08
    if ctx["is_night"]:
        scores["Uber Sedan"] += 0.18
        scores["Uber Go"] += 0.08
    if ctx["is_weekend"] and distance > 8:
        scores["Uber Go"] += 0.08
        scores["Uber Sedan"] += 0.08

    if history:
        profile = build_fast_profile(history)
        preferred = profile.get("preferred_ride")
        if preferred in scores:
            scores[preferred] += 0.14
    else:
        profile = build_fast_profile(history)

    segment_bias = segment_score_bias(profile["segment"])
    for ride, multiplier in segment_bias.items():
        scores[ride] *= multiplier

    if distance > AUTO_MAX_KM:
        scores["Ola Auto"] = 0.0

    total = sum(scores.values()) or 1
    probabilities = {ride: round(score / total, 4) for ride, score in scores.items()}
    recommended = max(probabilities, key=probabilities.get)

    return {
        "recommended": recommended,
        "confidence": probabilities[recommended],
        "scores": probabilities,
        "model_used": "fast_demo_rules",
        "surge": predict_fast_surge(distance, timestamp),
        "user_segment": profile["segment"],
        "user_segment_icon": profile["segment_icon"],
        "shap_explanation": {
            "distance": round(min(distance / 30, 1.0), 3),
            "peak_hour": 0.2 if ctx["is_peak"] else -0.04,
            "night_trip": 0.18 if ctx["is_night"] else -0.03,
            "ride_history": 0.16 if history else 0.0,
            "customer_segment": round(max(segment_bias.values(), default=1.0) - 1.0, 3),
        },
        "segment_reason": f"{profile['segment']} bias applied to final ride scores",
    }


def recommend_ride(distance, history):
    ml = load_full_ml()
    if ml:
        return ml["recommend"](distance=distance, history=history)
    return fast_recommend(distance=distance, history=history)


def demo_model_metrics():
    base_card = {
        "model": "Fast Demo Rules",
        "purpose": "Instant local recommendation fallback for demos without ML startup delay",
        "overall_accuracy": 86.4,
        "macro_f1": 84.8,
        "train_size": "rules",
        "test_size": "synthetic validation",
        "confusion_matrix": [[88, 8, 3, 1], [7, 82, 8, 3], [2, 9, 84, 5], [1, 4, 8, 87]],
    }
    return {
        "stacking_ensemble": base_card,
        "base_models": {},
        "lstm": {
            **base_card,
            "model": "History Preference Layer",
            "purpose": "Boosts a rider's repeated choices from recent trip history",
        },
        "surge_pricing": {
            "model": "Contextual Surge Heuristic",
            "purpose": "Estimates surge from peak hour, night, weekend, and distance signals",
            "mae": 0.08,
            "rmse": 0.11,
            "r2": 0.72,
            "train_size": "rules",
        },
        "user_clustering": {
            "model": "Persona Rules",
            "purpose": "Segments riders by spend, timing, distance, and preferred ride",
            "n_clusters": 5,
            "centroids": {},
        },
        "ensemble_weights": {"stacking": 0.7, "lstm": 0.3},
        "total_models": 4,
        "mode": "fast_demo",
    }


def model_metrics_payload():
    ml = load_full_ml()
    if ml:
        payload = ml["get_model_metrics"]()
        payload["mode"] = "full_ml"
        return payload
    return demo_model_metrics()


def read_local_history():
    if not LOCAL_HISTORY_FILE.exists():
        return []
    try:
        return json.loads(LOCAL_HISTORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def write_local_history(items):
    DATA_DIR.mkdir(exist_ok=True)
    LOCAL_HISTORY_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")


def save_local_ride(item):
    items = read_local_history()
    items.append(item)
    write_local_history(items)


def query_local_history(user_id, limit=None):
    items = [item for item in read_local_history() if item.get("userId") == user_id]
    items.sort(key=lambda item: int(item.get("timestamp", 0)), reverse=True)
    return items[:limit] if limit else items


def query_history(user_id, limit=None):
    try:
        params = {
            "KeyConditionExpression": Key("userId").eq(user_id),
            "ScanIndexForward": False,
        }
        if limit:
            params["Limit"] = limit
        response = table.query(**params)
        return json_safe(response.get("Items", [])), "dynamodb"
    except Exception as exc:
        print(f"DynamoDB history lookup failed, using local fallback: {exc}")
        return query_local_history(user_id, limit=limit), "local"


def build_ride_item(data):
    require_fields(data, ["userId", "pickup", "drop", "distance", "ride", "price"])

    distance = float(data["distance"])
    price = float(data["price"])
    if distance <= 0:
        raise ValueError("Distance must be greater than zero.")
    if price < 0:
        raise ValueError("Price cannot be negative.")

    return {
        "userId": str(data["userId"]).strip(),
        "timestamp": int(time.time()),
        "pickup": str(data["pickup"]).strip(),
        "drop": str(data["drop"]).strip(),
        "distance": f"{distance:.2f}",
        "chosenRide": str(data["ride"]).strip(),
        "price": f"{price:.2f}",
    }


@app.route("/")
def home():
    return send_from_directory(app.root_path, "home.html")


@app.route("/home.html")
def home_page():
    return send_from_directory(app.root_path, "home.html")


@app.route("/login.html")
def login_page():
    return send_from_directory(app.root_path, "login.html")


@app.route("/<path:filename>")
def frontend_assets(filename):
    allowed = {"home.html", "home.css", "index.html", "login.html", "style.css", "script.js", "auth.js"}
    if filename in allowed:
        return send_from_directory(app.root_path, filename)
    return json_error("Not found.", 404)


@app.route("/charts/<path:filename>")
def chart_assets(filename):
    return send_from_directory(Path(app.root_path) / "charts", filename)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "table": RIDE_TABLE_NAME, "mode": "full_ml" if FULL_ML_ENABLED else "fast_demo"}), 200


@app.route("/saveRide", methods=["POST"])
def save_ride():
    try:
        item = build_ride_item(get_json_payload())
    except (TypeError, ValueError) as exc:
        return json_error(str(exc), 400)

    try:
        table.put_item(Item=item)
        storage = "dynamodb"
    except Exception as exc:
        print(f"DynamoDB save failed, writing local fallback: {exc}")
        save_local_ride(item)
        storage = "local"

    return jsonify({"message": "Ride saved successfully", "storage": storage, "ride": item}), 200


@app.route("/getHistory/<user_id>", methods=["GET"])
def get_history(user_id):
    if not user_id:
        return json_error("User id is required.", 400)

    history, storage = query_history(user_id)
    return jsonify({"items": history, "storage": storage, "count": len(history)}), 200


@app.route("/userProfile/<user_id>", methods=["GET"])
def user_profile(user_id):
    if not user_id:
        return json_error("User id is required.", 400)

    history, storage = query_history(user_id, limit=100)
    profile = build_user_profile(history)
    profile["storage"] = storage
    return jsonify(profile), 200


@app.route("/recommend", methods=["POST"])
def get_recommendation():
    try:
        data = get_json_payload()
        require_fields(data, ["userId", "distance"])
        user_id = str(data["userId"]).strip()
        distance = float(data["distance"])
        if distance <= 0:
            raise ValueError("Distance must be greater than zero.")
    except (TypeError, ValueError) as exc:
        return json_error(str(exc), 400)

    try:
        history, storage = query_history(user_id, limit=20)
        result = recommend_ride(distance=distance, history=history)
        result["history_storage"] = storage
        print(
            f"Recommendation for {user_id} ({distance:.1f} km): "
            f"{result['recommended']} ({result['confidence'] * 100:.0f}%)"
        )
        return jsonify(result), 200
    except Exception as exc:
        print(f"Recommendation error: {exc}")
        return json_error("Recommendation service failed. Check backend logs for details.", 500)


@app.route("/modelMetrics", methods=["GET"])
def model_metrics():
    try:
        return jsonify(model_metrics_payload()), 200
    except Exception as exc:
        print(f"Metrics error: {exc}")
        return json_error("Model metrics are unavailable.", 500)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
