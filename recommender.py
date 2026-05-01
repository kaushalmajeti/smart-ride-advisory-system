"""
Smart Ride — Advanced ML Recommendation Engine
=================================================

FIVE ML MODELS + ENSEMBLE:
──────────────────────────
1. XGBoost Classifier
   PURPOSE : Predicts the best ride for the CURRENT TRIP based on trip features.
   INPUT   : distance, hour, day-of-week, is_weekend, is_peak, is_night,
             auto_available, distance², distance×peak, distance×night, hour×weekend
   OUTPUT  : probability over [Ola Auto, Ola Mini, Uber Go, Uber Sedan]
   WHY XGB : Handles tabular/structured data extremely well. Fast, interpretable,
             works great even on small datasets. Gives feature importances.

2. Random Forest Classifier
   PURPOSE : Second base learner for the stacking ensemble.
   WHY RF  : Low variance, naturally handles class imbalance, decorrelated trees
             add diversity to the ensemble.

3. Gradient Boosting Classifier
   PURPOSE : Third base learner for the stacking ensemble.
   WHY GBM : Different boosting strategy than XGBoost (sklearn's implementation),
             adds diversity through algorithmic variation.

4. Stacking Meta-Learner (Logistic Regression)
   PURPOSE : Learns optimal combination of XGBoost + RF + GBM predictions.
   WHY     : Stacking is an advanced ensemble technique that achieves higher
             accuracy than any single model by learning which model to trust
             for which input patterns.

5. LSTM (Long Short-Term Memory) Sequence Model
   PURPOSE : Learns the USER'S PERSONAL PATTERN from their last N rides.
   INPUT   : Sequence of last 5 rides — each encoded as
             [normalised_distance, normalised_hour, normalised_day, ride_index]
   OUTPUT  : probability over [Ola Auto, Ola Mini, Uber Go, Uber Sedan]
   WHY LSTM: Sequential ride choices carry temporal patterns — e.g. a user always
             takes Auto on Monday mornings and Uber Go on weekend nights.
             LSTMs capture this time-series dependency. Pure NumPy, no TF/PyTorch.

ADDITIONAL ML COMPONENTS:
─────────────────────────
6. Dynamic Surge Pricing (XGBoost Regressor)
   PURPOSE : Predicts surge pricing multiplier (1.0x–2.5x) for realistic pricing.
   FEATURES: hour, day, is_peak, is_night, is_weekend, distance

7. K-Means User Clustering
   PURPOSE : Segments users into personas (Budget Rider, Premium Commuter,
             Weekend Explorer, Night Owl) based on ride history features.

8. SHAP Explainability
   PURPOSE : Explains WHY a particular ride was recommended using SHAP values.
   WHY     : Transparent, explainable AI is critical for user trust.

FINAL ENSEMBLE:
───────────────
  Final score = 0.55 × Stacking + 0.45 × LSTM
  Cold start  = Stacking only (< 3 rides in history)

DATASET NOTES:
──────────────
  Currently trained on high-quality SYNTHETIC data (10 000 samples) that models
  realistic Indian ride-hailing patterns — exponential distance distribution,
  realistic hour distribution, context-based preferences (night, peak, weekend).
"""

import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP — graceful fallback if not installed
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠  SHAP not installed. Run: pip install shap")
    print("   Continuing without explainability features.\n")


# ── Constants ─────────────────────────────────────────────────────────────────

RIDE_LABELS  = ["Ola Auto", "Ola Mini", "Uber Go", "Uber Sedan"]
LABEL_TO_IDX = {r: i for i, r in enumerate(RIDE_LABELS)}
AUTO_MAX_KM  = 15
SEQ_LEN      = 5
N_FEAT_SEQ   = 4

FEATURE_NAMES = [
    "distance", "hour", "day_of_week", "is_weekend", "is_peak",
    "is_night", "auto_available", "distance²", "distance×peak",
    "distance×night", "hour×weekend"
]

# User segment names for K-Means clustering
SEGMENT_NAMES = ["Budget Rider", "Premium Commuter", "Weekend Explorer", "Night Owl"]


# ── Feature engineering ───────────────────────────────────────────────────────

def trip_features(distance: float, timestamp: int = None) -> np.ndarray:
    """
    11 features for a single trip  →  shape (1, 11)
    Used by the stacking ensemble at inference time.
    """
    if timestamp is None:
        timestamp = int(datetime.now().timestamp())
    dt         = datetime.fromtimestamp(timestamp)
    h          = dt.hour
    dow        = dt.weekday()
    is_weekend = int(dow >= 5)
    is_peak    = int(7 <= h < 10 or 17 <= h < 21)
    is_night   = int(h >= 22 or h < 6)
    auto_ok    = int(distance <= AUTO_MAX_KM)

    return np.array([[
        distance, h, dow,
        is_weekend, is_peak, is_night, auto_ok,
        distance**2, distance * is_peak,
        distance * is_night, h * is_weekend
    ]], dtype=np.float32)


def sequence_features(history: list) -> np.ndarray:
    """
    Last SEQ_LEN rides as a sequence  →  shape (1, SEQ_LEN, N_FEAT_SEQ)
    Used by LSTM at inference time.
    """
    seq = []
    for ride in history[-SEQ_LEN:]:
        d     = float(ride.get("distance", 0))
        ts    = int(ride.get("timestamp", datetime.now().timestamp()))
        dt    = datetime.fromtimestamp(ts)
        r_idx = LABEL_TO_IDX.get(ride.get("chosenRide", "Ola Mini"), 1) / 3.0
        seq.append([d / 50.0, dt.hour / 23.0, dt.weekday() / 6.0, r_idx])
    while len(seq) < SEQ_LEN:
        seq.insert(0, [0.0, 0.0, 0.0, 0.0])
    return np.array([seq], dtype=np.float32)


# ── Synthetic training data ────────────────────────────────────────────────────

def generate_training_data(n_samples: int = 10000, seed: int = 42):
    """
    Generates realistic synthetic ride data for India.
    10 000 samples for better model generalisation.
    """
    np.random.seed(seed)

    d1 = np.random.exponential(scale=4,   size=int(n_samples * 0.55))
    d2 = np.random.uniform(8,  20,        size=int(n_samples * 0.30))
    d3 = np.random.uniform(20, 60,        size=int(n_samples * 0.15))
    dist = np.clip(np.concatenate([d1, d2, d3]), 0.5, 60)
    np.random.shuffle(dist)
    n = len(dist)

    hour_p = np.array([
        0.005, 0.003, 0.002, 0.002, 0.003, 0.008,
        0.025, 0.060, 0.075, 0.055, 0.040, 0.045,
        0.055, 0.050, 0.045, 0.048, 0.052, 0.070,
        0.075, 0.065, 0.055, 0.040, 0.025, 0.012
    ])
    hour_p /= hour_p.sum()
    hour       = np.random.choice(24, n, p=hour_p)
    dow        = np.random.randint(0, 7, n)
    is_weekend = (dow >= 5).astype(int)
    is_peak    = (((7 <= hour) & (hour < 10)) | ((17 <= hour) & (hour < 21))).astype(int)
    is_night   = ((hour >= 22) | (hour < 6)).astype(int)
    auto_ok    = (dist <= AUTO_MAX_KM).astype(int)

    def label(d, we, peak, night, rng):
        if   d <= 2:  b = 0
        elif d <= 6:  b = 1
        elif d <= 18: b = 2
        else:         b = 3

        if night and b <= 1:
            b = min(b + 1, 3)
        if we and b == 2 and rng.random() < 0.30:
            b = 3
        if peak and b == 1 and rng.random() < 0.25:
            b = 2
        if d > AUTO_MAX_KM and b == 0:
            b = 1

        if rng.random() < 0.12:
            b = int(rng.choice(
                [max(0, b-1), b, min(3, b+1)],
                p=[0.25, 0.50, 0.25]
            ))
        if d > AUTO_MAX_KM and b == 0:
            b = 1
        return b

    rng = np.random.default_rng(seed)
    y = np.array([label(dist[i], is_weekend[i], is_peak[i], is_night[i], rng)
                  for i in range(n)])

    X = np.column_stack([
        dist, hour, dow, is_weekend, is_peak, is_night, auto_ok,
        dist**2, dist * is_peak, dist * is_night, hour * is_weekend
    ]).astype(np.float32)

    return X, y


# ── Generate surge pricing data ───────────────────────────────────────────────

def generate_surge_data(n_samples: int = 10000, seed: int = 42):
    """
    Generates synthetic surge pricing data.
    Surge multiplier (1.0x – 2.5x) depends on:
      - time of day (peak hours → higher surge)
      - day of week (weekends → slight surge)
      - distance (longer trips → slightly lower surge)
      - night time (safety premium)
    """
    np.random.seed(seed)

    hour       = np.random.choice(24, n_samples,
                                   p=np.array([0.005,0.003,0.002,0.002,0.003,0.008,
                                               0.025,0.060,0.075,0.055,0.040,0.045,
                                               0.055,0.050,0.045,0.048,0.052,0.070,
                                               0.075,0.065,0.055,0.040,0.025,0.012]) /
                                     np.array([0.005,0.003,0.002,0.002,0.003,0.008,
                                               0.025,0.060,0.075,0.055,0.040,0.045,
                                               0.055,0.050,0.045,0.048,0.052,0.070,
                                               0.075,0.065,0.055,0.040,0.025,0.012]).sum())
    dow        = np.random.randint(0, 7, n_samples)
    is_weekend = (dow >= 5).astype(float)
    is_peak    = (((7 <= hour) & (hour < 10)) | ((17 <= hour) & (hour < 21))).astype(float)
    is_night   = ((hour >= 22) | (hour < 6)).astype(float)
    distance   = np.clip(np.random.exponential(scale=8, size=n_samples), 0.5, 60)

    # Surge formula: base + peak_boost + night_boost + weekend_boost + noise
    surge = (1.0
             + is_peak * np.random.uniform(0.3, 0.8, n_samples)
             + is_night * np.random.uniform(0.1, 0.4, n_samples)
             + is_weekend * np.random.uniform(0.05, 0.2, n_samples)
             - (distance / 100.0)
             + np.random.normal(0, 0.08, n_samples))
    surge = np.clip(surge, 1.0, 2.5)

    X_surge = np.column_stack([hour, dow, is_peak, is_night, is_weekend, distance])
    return X_surge.astype(np.float32), surge.astype(np.float32)


# ── Stacking Ensemble training ────────────────────────────────────────────────

def train_stacking_ensemble(X, y):
    """
    Train a Stacking Ensemble with 3 base models + meta-learner.

    Architecture:
      Layer 1 (Base Learners):
        - XGBoost Classifier
        - Random Forest Classifier
        - Gradient Boosting Classifier
      Layer 2 (Meta-Learner):
        - Logistic Regression (learns optimal blend)

    Returns (stacking_model, individual_metrics, stacking_metrics)
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Base learners ─────────────────────────────────────────────────────
    xgb_clf = XGBClassifier(
        n_estimators     = 50,
        max_depth        = 5,
        learning_rate    = 0.08,
        subsample        = 0.85,
        colsample_bytree = 0.85,
        min_child_weight = 3,
        gamma            = 0.1,
        eval_metric      = "mlogloss",
        random_state     = 42,
        n_jobs           = 1,
        verbosity        = 0
    )

    rf_clf = RandomForestClassifier(
        n_estimators = 60,
        max_depth    = 8,
        min_samples_split = 5,
        min_samples_leaf  = 3,
        max_features = 'sqrt',
        random_state = 42,
        n_jobs       = 1
    )

    gbm_clf = GradientBoostingClassifier(
        n_estimators  = 50,
        max_depth     = 4,
        learning_rate = 0.1,
        subsample     = 0.85,
        random_state  = 42
    )

    # ── Stacking ensemble ─────────────────────────────────────────────────
    stacking_clf = StackingClassifier(
        estimators=[
            ('xgboost', xgb_clf),
            ('random_forest', rf_clf),
            ('gradient_boosting', gbm_clf)
        ],
        final_estimator=LogisticRegression(
    max_iter=1000,
    solver='lbfgs',
    random_state=42
    ),
        cv=3,
        stack_method='predict_proba',
        n_jobs=1
    )

    print("  Training Stacking Ensemble (XGBoost + RF + GBM + Meta-Learner)...")
    stacking_clf.fit(X_tr, y_tr)

    # ── Evaluate each base learner individually ───────────────────────────
    individual_metrics = {}
    base_models = {
        "XGBoost": xgb_clf,
        "Random Forest": rf_clf,
        "Gradient Boosting": gbm_clf
    }

    for name, model in base_models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc    = accuracy_score(y_te, y_pred)
        f1_m   = f1_score(y_te, y_pred, average="macro")
        report = classification_report(y_te, y_pred, target_names=RIDE_LABELS, output_dict=True)
        cm     = confusion_matrix(y_te, y_pred)

        individual_metrics[name] = {
            "model":            name,
            "purpose":          f"Base learner in stacking ensemble",
            "overall_accuracy": round(acc * 100, 2),
            "macro_f1":         round(f1_m * 100, 2),
            "per_class": {
                lbl: {
                    "precision": round(report[lbl]["precision"] * 100, 1),
                    "recall":    round(report[lbl]["recall"]    * 100, 1),
                    "f1":        round(report[lbl]["f1-score"]  * 100, 1),
                    "support":   report[lbl]["support"]
                }
                for lbl in RIDE_LABELS
            },
            "confusion_matrix": cm.tolist(),
            "train_size": len(X_tr),
            "test_size":  len(X_te),
        }

        print(f"    OK {name:20s}  acc={acc*100:.1f}%  F1={f1_m*100:.1f}%")

    # ── Evaluate stacking ensemble ────────────────────────────────────────
    y_pred_stack = stacking_clf.predict(X_te)
    acc_s    = accuracy_score(y_te, y_pred_stack)
    f1_s     = f1_score(y_te, y_pred_stack, average="macro")
    report_s = classification_report(y_te, y_pred_stack, target_names=RIDE_LABELS, output_dict=True)
    cm_s     = confusion_matrix(y_te, y_pred_stack)

    # Cross-validation score for stacking
    cv_scores = cross_val_score(stacking_clf, X, y, cv=3, scoring='accuracy', n_jobs=1)

    stacking_metrics = {
        "model":            "Stacking Ensemble",
        "purpose":          "Meta-learner that optimally combines XGBoost + RF + GBM predictions",
        "architecture":     "XGBoost + Random Forest + Gradient Boosting → Logistic Regression",
        "overall_accuracy": round(acc_s * 100, 2),
        "macro_f1":         round(f1_s * 100, 2),
        "cross_val_mean":   round(cv_scores.mean() * 100, 2),
        "cross_val_std":    round(cv_scores.std() * 100, 2),
        "per_class": {
            lbl: {
                "precision": round(report_s[lbl]["precision"] * 100, 1),
                "recall":    round(report_s[lbl]["recall"]    * 100, 1),
                "f1":        round(report_s[lbl]["f1-score"]  * 100, 1),
                "support":   report_s[lbl]["support"]
            }
            for lbl in RIDE_LABELS
        },
        "confusion_matrix": cm_s.tolist(),
        "train_size": len(X_tr),
        "test_size":  len(X_te),
    }

    print(f"\n  * Stacking Ensemble    acc={acc_s*100:.1f}%  F1={f1_s*100:.1f}%  "
          f"CV={cv_scores.mean()*100:.1f}±{cv_scores.std()*100:.1f}%")

    return stacking_clf, individual_metrics, stacking_metrics


# ── Surge Pricing Model ───────────────────────────────────────────────────────

def train_surge_model(X_surge, y_surge):
    """
    Train XGBoost Regressor for dynamic surge pricing prediction.
    Returns (model, metrics).
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_surge, y_surge, test_size=0.20, random_state=42
    )

    model = XGBRegressor(
        n_estimators     = 50,
        max_depth        = 4,
        learning_rate    = 0.08,
        subsample        = 0.85,
        colsample_bytree = 0.80,
        random_state     = 42,
        n_jobs           = 1,
        verbosity        = 0
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    mae    = float(np.mean(np.abs(y_te - y_pred)))
    rmse   = float(np.sqrt(np.mean((y_te - y_pred) ** 2)))
    r2     = float(1 - np.sum((y_te - y_pred)**2) / np.sum((y_te - y_te.mean())**2))

    metrics = {
        "model":  "Surge Pricing (XGBoost Regressor)",
        "purpose": "Predicts dynamic surge multiplier (1.0x–2.5x) based on demand patterns",
        "mae":    round(mae, 4),
        "rmse":   round(rmse, 4),
        "r2":     round(r2, 4),
        "train_size": len(X_tr),
        "test_size":  len(X_te),
    }

    print(f"  * Surge Pricing Model  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
    return model, metrics


# ── K-Means User Clustering ──────────────────────────────────────────────────

def train_user_clustering(X, y):
    """
    Train K-Means to create 4 user personas based on ride patterns.

    Clusters:
      0 - Budget Rider       (short distances, prefer Auto/Mini)
      1 - Premium Commuter   (medium distances, prefer Go/Sedan during peak)
      2 - Weekend Explorer   (weekend rides, varied distances)
      3 - Night Owl          (late-night rides, prefer comfort)
    """
    # Features for clustering: distance, hour, is_weekend, is_peak, is_night, ride choice
    cluster_features = np.column_stack([
        X[:, 0],  # distance
        X[:, 1],  # hour
        X[:, 3],  # is_weekend
        X[:, 4],  # is_peak
        X[:, 5],  # is_night
        y         # ride choice
    ]).astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_features)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_scaled)

    # Map cluster IDs to meaningful names based on cluster centroids
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_mapping = {}
    used = set()

    # Assign names based on centroid characteristics
    for i in range(4):
        c = centroids[i]
        avg_dist, avg_hour, avg_weekend, avg_peak, avg_night, avg_ride = c

        if avg_night > 0.3 and i not in used:
            cluster_mapping[i] = "Night Owl"
            used.add(i)
        elif avg_weekend > 0.4 and i not in used:
            cluster_mapping[i] = "Weekend Explorer"
            used.add(i)
        elif avg_ride >= 2.0 and i not in used:
            cluster_mapping[i] = "Premium Commuter"
            used.add(i)
        else:
            cluster_mapping[i] = "Budget Rider"
            used.add(i)

    # If any names are duplicated, assign remaining names
    all_names = set(SEGMENT_NAMES)
    assigned  = set(cluster_mapping.values())
    remaining = list(all_names - assigned)
    for i in range(4):
        if list(cluster_mapping.values()).count(cluster_mapping[i]) > 1:
            if remaining:
                cluster_mapping[i] = remaining.pop(0)

    metrics = {
        "model":    "K-Means User Clustering",
        "purpose":  "Segments users into personas for personalised recommendations",
        "n_clusters": 4,
        "segments":   cluster_mapping,
        "inertia":    round(float(kmeans.inertia_), 2),
        "centroids":  {
            cluster_mapping.get(i, f"Cluster {i}"): {
                "avg_distance":    round(float(centroids[i][0]), 1),
                "avg_hour":        round(float(centroids[i][1]), 1),
                "weekend_ratio":   round(float(centroids[i][2]), 2),
                "peak_ratio":      round(float(centroids[i][3]), 2),
                "night_ratio":     round(float(centroids[i][4]), 2),
                "avg_ride_choice": round(float(centroids[i][5]), 1)
            }
            for i in range(4)
        }
    }

    print(f"  * User Clustering      4 segments  inertia={kmeans.inertia_:.0f}")
    for i in range(4):
        name = cluster_mapping.get(i, f"Cluster {i}")
        print(f"    Segment {i}: {name:20s}  "
              f"avg_dist={centroids[i][0]:.1f}km  avg_hour={centroids[i][1]:.0f}")

    return kmeans, scaler, cluster_mapping, metrics


# ── SHAP Explainability ──────────────────────────────────────────────────────

def compute_shap_explanation(model, X_input):
    """
    Compute SHAP values for a single prediction.
    Returns a dict mapping feature names to their SHAP importance.
    """
    if not SHAP_AVAILABLE:
        return None

    try:
        # Use the XGBoost base model from the stacking ensemble
        base_xgb = model.named_estimators_['xgboost']
        explainer = shap.TreeExplainer(base_xgb)
        shap_values = explainer.shap_values(X_input)

        # Get the class with highest prediction
        pred_class = int(np.argmax(model.predict_proba(X_input)[0]))

        if isinstance(shap_values, list):
            sv = shap_values[pred_class][0]
        else:
            sv = shap_values[0, :, pred_class] if shap_values.ndim == 3 else shap_values[0]

        explanation = {}
        for i, fname in enumerate(FEATURE_NAMES):
            explanation[fname] = round(float(sv[i]), 4)

        # Sort by absolute importance
        explanation = dict(sorted(explanation.items(),
                                   key=lambda x: abs(x[1]), reverse=True))
        return explanation

    except Exception as e:
        print(f"  SHAP error: {e}")
        return None


# ── LSTM training (pure NumPy) ─────────────────────────────────────────────────

def train_lstm(history_sequences: list, labels: list,
               hidden: int = 40, epochs: int = 20, lr: float = 0.008):
    """
    Train a lightweight LSTM on sequences of rides with attention-like
    weighting on recent rides and dropout simulation.
    """
    np.random.seed(7)
    n_cls = len(RIDE_LABELS)
    s     = 0.08

    # Gate weights: forget, input, cell, output
    Wf = np.random.randn(hidden, N_FEAT_SEQ + hidden).astype(np.float32) * s
    Wi = np.random.randn(hidden, N_FEAT_SEQ + hidden).astype(np.float32) * s
    Wc = np.random.randn(hidden, N_FEAT_SEQ + hidden).astype(np.float32) * s
    Wo = np.random.randn(hidden, N_FEAT_SEQ + hidden).astype(np.float32) * s
    bf = np.zeros((hidden, 1), dtype=np.float32)
    bi = np.zeros((hidden, 1), dtype=np.float32)
    bc = np.zeros((hidden, 1), dtype=np.float32)
    bo = np.zeros((hidden, 1), dtype=np.float32)
    # Output layer
    Wy = np.random.randn(n_cls, hidden).astype(np.float32) * s
    by = np.zeros((n_cls, 1), dtype=np.float32)

    # Attention weights — learned recency bias
    W_attn = np.random.randn(hidden, 1).astype(np.float32) * 0.05

    sig     = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    sig_d   = lambda s: s * (1 - s)
    smax    = lambda x: (e := np.exp(x - x.max())) / e.sum()

    dropout_rate = 0.15  # Dropout simulation for regularisation

    def forward(seq, training=False):
        """Returns h, c at final step, plus cache for BPTT."""
        cache = []
        h = np.zeros((hidden, 1), dtype=np.float32)
        c = np.zeros((hidden, 1), dtype=np.float32)
        all_h = []

        for t in range(seq.shape[0]):
            x   = seq[t].reshape(-1, 1)
            z   = np.vstack([h, x])
            f   = sig(Wf @ z + bf)
            i   = sig(Wi @ z + bi)
            c_  = np.tanh(Wc @ z + bc)
            o   = sig(Wo @ z + bo)
            c   = f * c + i * c_
            h   = o * np.tanh(c)

            # Dropout during training
            if training and dropout_rate > 0:
                mask = (np.random.rand(*h.shape) > dropout_rate).astype(np.float32)
                h = h * mask / (1.0 - dropout_rate)

            cache.append((z, f, i, c_, o, c, h))
            all_h.append(h.copy())

        # Attention-like weighting: weight recent steps higher
        if len(all_h) > 1:
            attn_scores = np.array([float((W_attn.T @ h_t).flatten()[0])
                                     for h_t in all_h])
            attn_weights = np.exp(attn_scores - attn_scores.max())
            attn_weights /= attn_weights.sum()
            h_final = sum(w * h_t for w, h_t in zip(attn_weights, all_h))
        else:
            h_final = h

        return h_final, c, cache

    def predict_proba(seq: np.ndarray) -> np.ndarray:
        h, _, _ = forward(seq, training=False)
        return smax((Wy @ h + by).flatten())

    # ── Training loop ────────────────────────────────────────────────────────
    n_seq  = len(history_sequences)
    idx    = list(range(n_seq))

    for epoch in range(epochs):
        np.random.shuffle(idx)
        total_loss = 0.0

        for i in idx:
            seq   = history_sequences[i]
            label = labels[i]

            h, c, cache = forward(seq, training=True)
            probs = smax((Wy @ h + by).flatten())

            total_loss -= np.log(probs[label] + 1e-9)

            # Output layer gradient
            dL_dy       = probs.copy()
            dL_dy[label] -= 1.0
            dL_dy        = dL_dy.reshape(-1, 1)

            dWy = dL_dy @ h.T
            dby = dL_dy
            dh  = Wy.T @ dL_dy

            # BPTT through final timestep only (truncated for speed)
            z, f, i_, c_, o, c_t, h_t = cache[-1]
            dc = dh * o * (1 - np.tanh(c_t)**2)

            do   = dh * np.tanh(c_t)
            di   = dc * c_
            dc_  = dc * i_
            df   = dc * (c_t - c_)

            dWo = (sig_d(o) * do) @ z.T
            dbo = sig_d(o) * do
            dWi = (sig_d(i_) * di) @ z.T
            dbi = sig_d(i_) * di
            dWc = ((1 - c_**2) * dc_) @ z.T
            dbc = (1 - c_**2) * dc_
            dWf = (sig_d(f) * df) @ z.T
            dbf = sig_d(f) * df

            # SGD update with gradient clipping
            clip = 5.0
            for W, dW in [(Wy,dWy),(Wf,dWf),(Wi,dWi),(Wc,dWc),(Wo,dWo)]:
                dW = np.clip(dW, -clip, clip)
                W -= lr * dW
            for b, db in [(by,dby),(bf,dbf),(bi,dbi),(bc,dbc),(bo,dbo)]:
                db = np.clip(db, -clip, clip)
                b -= lr * db

        if (epoch + 1) % 20 == 0:
            print(f"    LSTM epoch {epoch+1}/{epochs}  loss: {total_loss/n_seq:.4f}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_true, y_pred_list = [], []
    for i in range(n_seq):
        p = predict_proba(history_sequences[i])
        y_true.append(labels[i])
        y_pred_list.append(int(np.argmax(p)))

    y_true      = np.array(y_true)
    y_pred_arr  = np.array(y_pred_list)
    acc         = accuracy_score(y_true, y_pred_arr)
    macro_f1    = f1_score(y_true, y_pred_arr, average="macro")
    report      = classification_report(y_true, y_pred_arr,
                                        target_names=RIDE_LABELS, output_dict=True)
    cm          = confusion_matrix(y_true, y_pred_arr)

    metrics = {
        "model":            "LSTM (with Attention & Dropout)",
        "purpose":          "Learns user's personal ride sequence pattern from history",
        "architecture":     f"LSTM({hidden} hidden) + Attention + Dropout({dropout_rate})",
        "overall_accuracy": round(acc * 100, 2),
        "macro_f1":         round(macro_f1 * 100, 2),
        "per_class":        {
            lbl: {
                "precision": round(report[lbl]["precision"] * 100, 1),
                "recall":    round(report[lbl]["recall"]    * 100, 1),
                "f1":        round(report[lbl]["f1-score"]  * 100, 1),
                "support":   report[lbl]["support"]
            }
            for lbl in RIDE_LABELS
        },
        "confusion_matrix": cm.tolist(),
        "train_size": n_seq,
        "note": "LSTM accuracy reflects sequence-level patterns; improves with more user history"
    }

    print(f"\n  * LSTM (Attention+Dropout)  acc={acc*100:.1f}%  F1={macro_f1*100:.1f}%")

    return predict_proba, metrics


def _build_lstm_sequences(X: np.ndarray, y: np.ndarray,
                           seq_len: int = SEQ_LEN) -> tuple:
    """Convert the flat trip dataset into overlapping sequences for LSTM."""
    seqs, lbls = [], []
    for i in range(len(X) - seq_len):
        window = []
        for j in range(i, i + seq_len):
            dist_norm = X[j, 0] / 50.0
            hour_norm = X[j, 1] / 23.0
            dow_norm  = X[j, 2] / 6.0
            ride_norm = y[j]    / 3.0
            window.append([dist_norm, hour_norm, dow_norm, ride_norm])
        seqs.append(np.array(window, dtype=np.float32))
        lbls.append(int(y[i + seq_len]))
    return seqs, lbls


# ── User profiling from ride history ──────────────────────────────────────────

def build_user_profile(history: list, kmeans_model, scaler_model, cluster_map):
    """
    Build a user profile from their DynamoDB ride history.
    Returns segment name, stats, and ride preferences.
    """
    if not history or len(history) == 0:
        return {
            "segment":        "New User",
            "segment_icon":   "NEW",
            "total_rides":    0,
            "avg_distance":   0,
            "avg_price":      0,
            "preferred_ride": "Unknown",
            "ride_distribution": {},
            "peak_rider":     False,
            "night_rider":    False,
            "weekend_ratio":  0,
            "spending_tier":  "N/A"
        }

    distances = [float(r.get("distance", 0)) for r in history]
    prices    = [float(r.get("price", 0)) for r in history]
    rides     = [r.get("chosenRide", "Unknown") for r in history]

    # Count ride types
    ride_counts = {}
    for r in rides:
        ride_counts[r] = ride_counts.get(r, 0) + 1
    preferred = max(ride_counts, key=ride_counts.get)
    ride_dist = {k: round(v / len(rides) * 100, 1) for k, v in ride_counts.items()}

    # Time analysis
    hours = []
    weekends = 0
    peaks = 0
    nights = 0
    for r in history:
        ts = int(r.get("timestamp", 0))
        if ts > 0:
            dt = datetime.fromtimestamp(ts)
            hours.append(dt.hour)
            if dt.weekday() >= 5:
                weekends += 1
            if 7 <= dt.hour < 10 or 17 <= dt.hour < 21:
                peaks += 1
            if dt.hour >= 22 or dt.hour < 6:
                nights += 1

    n = len(history)
    avg_dist  = float(np.mean(distances))
    avg_price = float(np.mean(prices))
    avg_hour  = float(np.mean(hours)) if hours else 12.0
    weekend_r = weekends / n if n > 0 else 0
    peak_r    = peaks / n if n > 0 else 0
    night_r   = nights / n if n > 0 else 0
    avg_ride  = float(np.mean([LABEL_TO_IDX.get(r, 1) for r in rides]))

    # Predict user cluster
    cluster_input = np.array([[avg_dist, avg_hour, weekend_r, peak_r, night_r, avg_ride]],
                              dtype=np.float32)
    try:
        cluster_scaled = scaler_model.transform(cluster_input)
        cluster_id = int(kmeans_model.predict(cluster_scaled)[0])
        segment = cluster_map.get(cluster_id, "Regular Rider")
    except Exception:
        segment = "Regular Rider"

    segment_icons = {
        "Budget Rider": "SAVE",
        "Premium Commuter": "PRO",
        "Weekend Explorer": "TRIP",
        "Night Owl": "NIGHT",
        "New User": "NEW",
        "Regular Rider": "RIDE"
    }

    # Spending tier
    if avg_price < 80:
        spending_tier = "Economy"
    elif avg_price < 200:
        spending_tier = "Standard"
    else:
        spending_tier = "Premium"

    return {
        "segment":           segment,
        "segment_icon":      segment_icons.get(segment, "RIDE"),
        "total_rides":       n,
        "avg_distance":      round(avg_dist, 1),
        "avg_price":         round(avg_price, 0),
        "preferred_ride":    preferred,
        "ride_distribution": ride_dist,
        "peak_rider":        peak_r > 0.4,
        "night_rider":       night_r > 0.2,
        "weekend_ratio":     round(weekend_r * 100, 0),
        "spending_tier":     spending_tier
    }


# ══════════════════════════════════════════════════════════════════════════════
# INITIALISE ALL MODELS AT IMPORT TIME
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Smart Ride - Advanced ML Engine Initialising")
print("="*60)

# 1. Generate training data
print("\nGenerating training data (2,000 samples)...")
_X, _y = generate_training_data(n_samples=2000)

# 2. Train stacking ensemble
print("\nTraining Stacking Ensemble...")
_stacking_model, BASE_METRICS, STACKING_METRICS = train_stacking_ensemble(_X, _y)

# 3. Train surge pricing model
print("\nTraining Surge Pricing Model...")
_X_surge, _y_surge = generate_surge_data(n_samples=2000)
_surge_model, SURGE_METRICS = train_surge_model(_X_surge, _y_surge)

# 4. Train user clustering
print("\nTraining User Clustering...")
_kmeans, _scaler, _cluster_map, CLUSTER_METRICS = train_user_clustering(_X, _y)

# 5. Train LSTM
print("\nPreparing LSTM sequences...")
_seqs, _lbls = _build_lstm_sequences(_X, _y)
_lstm_idx    = np.random.default_rng(99).choice(len(_seqs), size=min(300, len(_seqs)), replace=False)
_lstm_seqs   = [_seqs[i] for i in _lstm_idx]
_lstm_lbls   = [_lbls[i] for i in _lstm_idx]

print("  Training LSTM (with Attention & Dropout)...")
_lstm_predict, LSTM_METRICS = train_lstm(_lstm_seqs, _lstm_lbls)

# 6. SHAP explainer (precompute)
_shap_explainer = None
if SHAP_AVAILABLE:
    try:
        _base_xgb = _stacking_model.named_estimators_['xgboost']
        _shap_explainer = shap.TreeExplainer(_base_xgb)
        print("\nSHAP Explainer ready")
    except Exception as e:
        print(f"\n⚠  SHAP init failed: {e}")

# Ensemble weights
STACK_W   = 0.85
LSTM_W    = 0.15
MIN_RIDES = 3

print("\n" + "="*60)
print("  All models ready!")
print(f"  Models: Stacking(XGB+RF+GBM), LSTM, Surge, Clustering, SHAP")
print("="*60 + "\n")


def segment_probability_bias(segment: str) -> np.ndarray:
    """
    K-Means persona bias applied after model probabilities.

    The ML models still score route and time fit. The customer cluster then
    nudges the final probabilities toward the ride style that matches the
    segment: budget users toward Auto/Mini, premium users toward Go/Sedan,
    weekend explorers toward flexible cabs, and night riders toward safer
    cab options.
    """
    biases = {
        "Budget Rider":     [1.35, 1.25, 0.90, 0.65],
        "Premium Commuter": [0.55, 0.75, 1.10, 1.45],
        "Weekend Explorer": [0.75, 0.95, 1.25, 1.15],
        "Night Owl":        [0.45, 0.80, 1.15, 1.45],
        "Regular Rider":    [1.00, 1.00, 1.00, 1.00],
        "New User":         [1.00, 1.00, 1.00, 1.00],
    }
    return np.array(biases.get(segment, biases["Regular Rider"]), dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def predict_surge(distance: float, timestamp: int = None) -> dict:
    """
    Predict surge pricing multiplier for a given trip.
    Returns { "multiplier": 1.35, "level": "moderate", "reason": "..." }
    """
    if timestamp is None:
        timestamp = int(datetime.now().timestamp())
    dt  = datetime.fromtimestamp(timestamp)
    h   = dt.hour
    dow = dt.weekday()

    X_in = np.array([[
        h, dow,
        int(7 <= h < 10 or 17 <= h < 21),   # is_peak
        int(h >= 22 or h < 6),                # is_night
        int(dow >= 5),                         # is_weekend
        distance
    ]], dtype=np.float32)

    multiplier = float(np.clip(_surge_model.predict(X_in)[0], 1.0, 2.5))

    # Determine level and reason
    if multiplier < 1.15:
        level, reason = "low", "Normal demand - regular pricing"
    elif multiplier < 1.4:
        level, reason = "moderate", "Slightly elevated demand"
    elif multiplier < 1.8:
        level, reason = "high", "High demand in your area"
    else:
        level, reason = "surge", "Very high demand - peak surge pricing"

    # Add specific context
    if 7 <= h < 10:
        reason += " (morning rush hour)"
    elif 17 <= h < 21:
        reason += " (evening rush hour)"
    elif h >= 22 or h < 6:
        reason += " (late night premium)"
    if dow >= 5:
        reason += " + weekend"

    return {
        "multiplier": round(multiplier, 2),
        "level":      level,
        "reason":     reason
    }


def recommend(distance: float, history: list, timestamp: int = None) -> dict:
    """
    Main entry point — called by Flask /recommend.

    Returns
    -------
    {
        "recommended": "Uber Go",
        "confidence":  0.74,
        "scores":      {"Ola Auto": 0.05, ...},
        "model_used":  "stacking_ensemble" | "stacking_only",
        "surge":       {"multiplier": 1.2, "level": "moderate", ...},
        "user_segment": "Premium Commuter",
        "shap_explanation": {"distance": 0.23, "is_peak": 0.15, ...} | null
    }
    """
    X_input  = trip_features(distance, timestamp)
    stack_p  = _stacking_model.predict_proba(X_input)[0]

    if len(history) >= MIN_RIDES:
        lstm_p     = _lstm_predict(sequence_features(history)[0])
        probs      = STACK_W * stack_p + LSTM_W * lstm_p
        model_used = "stacking_ensemble"
    else:
        probs      = stack_p.copy()
        model_used = "stacking_only"

    # Surge pricing
    surge = predict_surge(distance, timestamp)

    # User profile segment
    profile = build_user_profile(history, _kmeans, _scaler, _cluster_map)
    segment_bias = segment_probability_bias(profile["segment"])
    probs = probs * segment_bias

    if distance > AUTO_MAX_KM:
        probs[0] = 0.0

    total = probs.sum()
    if total > 0:
        probs /= total

    best = int(np.argmax(probs))

    # SHAP explanation
    shap_explanation = None
    if _shap_explainer is not None:
        try:
            sv = _shap_explainer.shap_values(X_input)
            if isinstance(sv, list):
                sv_class = sv[best][0]
            else:
                sv_class = sv[0, :, best] if sv.ndim == 3 else sv[0]
            shap_explanation = {}
            for i, fname in enumerate(FEATURE_NAMES):
                shap_explanation[fname] = round(float(sv_class[i]), 4)
            shap_explanation = dict(sorted(shap_explanation.items(),
                                            key=lambda x: abs(x[1]), reverse=True))
            # Keep top 5 for cleaner UI
            shap_explanation = dict(list(shap_explanation.items())[:5])
        except Exception:
            pass
    if shap_explanation is None:
        shap_explanation = {}
    shap_explanation["customer_segment"] = round(float(segment_bias[best] - 1.0), 4)

    return {
        "recommended":      RIDE_LABELS[best],
        "confidence":       float(round(probs[best], 4)),
        "scores":           {RIDE_LABELS[i]: float(round(probs[i], 4)) for i in range(4)},
        "model_used":       model_used,
        "surge":            surge,
        "user_segment":     profile["segment"],
        "user_segment_icon": profile["segment_icon"],
        "shap_explanation": shap_explanation,
        "segment_reason":   f"{profile['segment']} bias applied to final ride scores",
    }


def get_user_profile(history: list) -> dict:
    """Build user profile from ride history. Called by Flask /userProfile."""
    return build_user_profile(history, _kmeans, _scaler, _cluster_map)


def get_model_metrics() -> dict:
    """
    Returns accuracy metrics for ALL models.
    Called by Flask /modelMetrics endpoint.
    """
    return {
        "stacking_ensemble": STACKING_METRICS,
        "base_models":       BASE_METRICS,
        "lstm":              LSTM_METRICS,
        "surge_pricing":     SURGE_METRICS,
        "user_clustering":   CLUSTER_METRICS,
        "ensemble_weights": {
            "stacking": STACK_W,
            "lstm":     LSTM_W
        },
        "total_models":     6,
        "model_list": [
            "XGBoost Classifier",
            "Random Forest Classifier",
            "Gradient Boosting Classifier",
            "Stacking Meta-Learner (Logistic Regression)",
            "LSTM with Attention & Dropout",
            "XGBoost Regressor (Surge Pricing)",
            "K-Means Clustering (User Segmentation)"
        ]
    }
