"""
Smart Ride — Conference Paper Chart Generator
==============================================
Run:   python generate_charts.py
Output: ./charts/  folder — 12 publication-ready PNG figures @ 300 DPI

Charts:
  01_dataset_distribution.png      Class + distance + hour distribution
  02_feature_correlation.png       Feature correlation heatmap
  03_feature_distributions.png     Per-feature box plots by ride class
  04_xgb_feature_importance.png    XGBoost feature importance (gain + weight)
  05_model_comparison.png          Accuracy / F1 / Precision / Recall bar chart
  06_xgb_confusion_matrix.png      XGBoost confusion matrix (% + count)
  07_lstm_confusion_matrix.png     LSTM confusion matrix
  08_ensemble_confusion_matrix.png Ensemble confusion matrix
  09_roc_curves.png                One-vs-Rest ROC + AUC for all 3 models
  10_learning_curve.png            XGBoost learning curve + LSTM loss curve
  11_precision_recall_per_class.png Per-class P / R / F1 grouped bars
  12_distance_vs_prediction.png    Predicted class across distance range

Requirements:  pip install xgboost scikit-learn matplotlib seaborn numpy
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.makedirs("charts", exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
RIDE_LBLS  = ["Ola Auto", "Ola Mini", "Uber Go", "Uber Sedan"]
PALETTE    = ["#4C9BE8", "#F0A500", "#3DBE7A", "#E85D75"]
FEAT_NAMES = [
    "Distance (km)", "Hour of day",   "Day of week",
    "Is weekend",    "Is peak hour",  "Is night",
    "Auto available","Distance²",     "Dist × peak",
    "Dist × night",  "Hour × weekend"
]
SEQ_LEN  = 5
N_FEAT_S = 4
HIDDEN   = 48
N_CLS    = 4
EPOCHS   = 60
LR       = 0.008

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "grid.alpha":        0.3,
    "legend.framealpha": 0.9,
})

# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Generate training data
# ══════════════════════════════════════════════════════════════════════
print("=" * 58)
print("  Smart Ride — Conference Chart Generator")
print("=" * 58)
print("\n[1/4] Generating training data ...")

np.random.seed(42)
N = 8000
d1   = np.random.exponential(4,    int(N * 0.55))
d2   = np.random.uniform(8,  20,   int(N * 0.30))
d3   = np.random.uniform(20, 60,   int(N * 0.15))
dist = np.clip(np.concatenate([d1, d2, d3]), 0.5, 60)
np.random.shuffle(dist)
n    = len(dist)

hp  = np.array([
    0.005, 0.003, 0.002, 0.002, 0.003, 0.008,
    0.025, 0.060, 0.075, 0.055, 0.040, 0.045,
    0.055, 0.050, 0.045, 0.048, 0.052, 0.070,
    0.075, 0.065, 0.055, 0.040, 0.025, 0.012,
])
hp /= hp.sum()
hour    = np.random.choice(24, n, p=hp)
dow     = np.random.randint(0, 7, n)
is_we   = (dow >= 5).astype(int)
is_pk   = (((7 <= hour) & (hour < 10)) | ((17 <= hour) & (hour < 21))).astype(int)
is_nt   = ((hour >= 22) | (hour < 6)).astype(int)
auto_ok = (dist <= 15).astype(int)

rng = np.random.default_rng(42)

def assign_label(d, we, pk, nt, r):
    b = 0 if d <= 2 else (1 if d <= 6 else (2 if d <= 18 else 3))
    if nt and b <= 1:                          b = min(b + 1, 3)
    if we and b == 2 and r.random() < 0.30:    b = 3
    if pk and b == 1 and r.random() < 0.25:    b = 2
    if d > 15 and b == 0:                      b = 1
    if r.random() < 0.12:
        b = int(r.choice([max(0, b-1), b, min(3, b+1)], p=[.25, .5, .25]))
    if d > 15 and b == 0:                      b = 1
    return b

y = np.array([assign_label(dist[i], is_we[i], is_pk[i], is_nt[i], rng)
              for i in range(n)])

X = np.column_stack([
    dist, hour, dow, is_we, is_pk, is_nt, auto_ok,
    dist**2, dist * is_pk, dist * is_nt, hour * is_we
])

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"   Total samples : {n}  |  Train: {len(X_tr)}  |  Test: {len(X_te)}")
print(f"   Class counts  : {np.bincount(y)}")


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Train models
# ══════════════════════════════════════════════════════════════════════
print("\n[2/4] Training models ...")

# ── XGBoost ──────────────────────────────────────────────────────────
xgb_model = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.08,
    subsample=0.85,   colsample_bytree=0.85,
    min_child_weight=3, gamma=0.1,
    eval_metric="mlogloss", random_state=42, verbosity=0,
)
xgb_model.fit(X_tr, y_tr)
y_pred_xgb  = xgb_model.predict(X_te)
y_prob_xgb  = xgb_model.predict_proba(X_te)
print(f"   XGBoost accuracy : {accuracy_score(y_te, y_pred_xgb)*100:.2f}%")

# ── LSTM (pure NumPy) ─────────────────────────────────────────────────
def build_sequences(X_arr, y_arr, seq_len=SEQ_LEN):
    seqs, lbls = [], []
    for i in range(len(X_arr) - seq_len):
        w = np.array([[X_arr[j,0]/50., X_arr[j,1]/23.,
                       X_arr[j,2]/6.,  y_arr[j]/3.]
                      for j in range(i, i + seq_len)], dtype=np.float32)
        seqs.append(w)
        lbls.append(int(y_arr[i + seq_len]))
    return seqs, lbls

# Build sequences from training data; use 1500-sample subset
all_seqs, all_lbls = build_sequences(X_tr, y_tr)
sel = np.random.default_rng(99).choice(
    len(all_seqs), size=min(1500, len(all_seqs)), replace=False)
tr_seqs = [all_seqs[i] for i in sel]
tr_lbls = [all_lbls[i] for i in sel]

# Initialise weights
np.random.seed(7)
s  = 0.08
Wf = np.random.randn(HIDDEN, N_FEAT_S + HIDDEN).astype(np.float32) * s
Wi = np.random.randn(HIDDEN, N_FEAT_S + HIDDEN).astype(np.float32) * s
Wc = np.random.randn(HIDDEN, N_FEAT_S + HIDDEN).astype(np.float32) * s
Wo = np.random.randn(HIDDEN, N_FEAT_S + HIDDEN).astype(np.float32) * s
bf = np.zeros((HIDDEN, 1), dtype=np.float32)
bi = np.zeros((HIDDEN, 1), dtype=np.float32)
bc = np.zeros((HIDDEN, 1), dtype=np.float32)
bo = np.zeros((HIDDEN, 1), dtype=np.float32)
Wy = np.random.randn(N_CLS, HIDDEN).astype(np.float32) * s
by = np.zeros((N_CLS, 1), dtype=np.float32)

_sig  = lambda x: 1. / (1. + np.exp(-np.clip(x, -30, 30)))
_smax = lambda x: (e := np.exp(x - x.max())) / e.sum()
_sd   = lambda sv: sv * (1 - sv)

def lstm_forward(seq):
    h = np.zeros((HIDDEN, 1), dtype=np.float32)
    c = np.zeros((HIDDEN, 1), dtype=np.float32)
    cache = []
    for t in range(seq.shape[0]):
        x = seq[t].reshape(-1, 1)
        z = np.vstack([h, x])
        f = _sig(Wf @ z + bf);  i_ = _sig(Wi @ z + bi)
        c_ = np.tanh(Wc @ z + bc);  o = _sig(Wo @ z + bo)
        c  = f * c + i_ * c_;       h  = o * np.tanh(c)
        cache.append((z, f, i_, c_, o, c, h))
    return h, c, cache

def lstm_predict(seq):
    h, _, _ = lstm_forward(seq)
    return _smax((Wy @ h + by).flatten())

# Training loop
lstm_losses = []
idx_list    = list(range(len(tr_seqs)))
for ep in range(EPOCHS):
    np.random.shuffle(idx_list)
    ep_loss = 0.0
    for i in idx_list:
        seq = tr_seqs[i];  lbl = tr_lbls[i]
        h, c, cache = lstm_forward(seq)
        probs = _smax((Wy @ h + by).flatten())
        ep_loss -= float(np.log(probs[lbl] + 1e-9))

        dL = probs.copy();  dL[lbl] -= 1.;  dL = dL.reshape(-1, 1)
        dWy = dL @ h.T;     dby = dL;       dh  = Wy.T @ dL
        z, f, i_, c_, o, ct, ht = cache[-1]
        dc  = dh * o * (1 - np.tanh(ct)**2)
        do  = dh * np.tanh(ct)
        di  = dc * c_;  dc_ = dc * i_;  df = dc * (ct - c_)

        for W, dW in [(Wy, dWy), (Wf, _sd(f)*df@z.T), (Wi, _sd(i_)*di@z.T),
                      (Wc, (1-c_**2)*dc_@z.T), (Wo, _sd(o)*do@z.T)]:
            W -= LR * np.clip(dW, -5, 5)
        for b, db in [(by, dby), (bf, _sd(f)*df), (bi, _sd(i_)*di),
                      (bc, (1-c_**2)*dc_), (bo, _sd(o)*do)]:
            b -= LR * np.clip(db, -5, 5)

    lstm_losses.append(ep_loss / len(tr_seqs))
    if (ep + 1) % 20 == 0:
        print(f"   LSTM epoch {ep+1:>3}/{EPOCHS}  loss: {lstm_losses[-1]:.4f}")

# LSTM test predictions
te_seqs, te_lbls = build_sequences(X_te, y_te)
y_true_lstm = np.array(te_lbls)
y_prob_lstm = np.array([lstm_predict(s) for s in te_seqs])
y_pred_lstm = np.argmax(y_prob_lstm, axis=1)
print(f"   LSTM accuracy    : {accuracy_score(y_true_lstm, y_pred_lstm)*100:.2f}%")

# Ensemble (aligned indices)
n_ens          = min(len(y_pred_lstm), len(X_te) - SEQ_LEN)
y_te_ens       = y_te[SEQ_LEN: SEQ_LEN + n_ens]
y_prob_xgb_ens = xgb_model.predict_proba(X_te[SEQ_LEN: SEQ_LEN + n_ens])
y_prob_ens     = 0.60 * y_prob_xgb_ens + 0.40 * y_prob_lstm[:n_ens]
y_pred_ens     = np.argmax(y_prob_ens, axis=1)
print(f"   Ensemble accuracy: {accuracy_score(y_te_ens, y_pred_ens)*100:.2f}%")


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — Compute metrics
# ══════════════════════════════════════════════════════════════════════
print("\n[3/4] Computing metrics ...")

def compute_metrics(name, y_true, y_pred, y_prob):
    return {
        "name":      name,
        "accuracy":  accuracy_score(y_true, y_pred) * 100,
        "macro_f1":  f1_score(y_true, y_pred, average="macro",  zero_division=0) * 100,
        "macro_pre": precision_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "macro_rec": recall_score(y_true, y_pred, average="macro",  zero_division=0) * 100,
        "report":    classification_report(y_true, y_pred, target_names=RIDE_LBLS, output_dict=True),
        "cm":        confusion_matrix(y_true, y_pred),
        "y_true":    y_true,
        "y_pred":    y_pred,
        "y_prob":    y_prob,
    }

m_xgb  = compute_metrics("XGBoost",  y_te,        y_pred_xgb,  y_prob_xgb)
m_lstm = compute_metrics("LSTM",      y_true_lstm, y_pred_lstm, y_prob_lstm)
m_ens  = compute_metrics("Ensemble",  y_te_ens,    y_pred_ens,  y_prob_ens)
ALL    = [m_xgb, m_lstm, m_ens]

print(f"\n  {'Model':<12} {'Accuracy':>10} {'Macro F1':>10} {'Precision':>11} {'Recall':>9}")
print("  " + "-" * 56)
for m in ALL:
    print(f"  {m['name']:<12} {m['accuracy']:>9.2f}%  {m['macro_f1']:>8.2f}%  "
          f"{m['macro_pre']:>9.2f}%  {m['macro_rec']:>8.2f}%")


# ══════════════════════════════════════════════════════════════════════
# STEP 4 — Generate all charts
# ══════════════════════════════════════════════════════════════════════
print("\n[4/4] Generating charts ...")


# ── Chart 01: Dataset Distribution ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Figure 1: Dataset Distribution Analysis",
             fontweight="bold", y=1.02)

counts = np.bincount(y)
bars   = axes[0].bar(RIDE_LBLS, counts, color=PALETTE, edgecolor="white", linewidth=0.8)
for bar, c in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f"{c}\n({c/n*100:.1f}%)", ha="center", va="bottom", fontsize=9)
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Samples")
axes[0].set_ylim(0, max(counts) * 1.28)

for ci, (lb, col) in enumerate(zip(RIDE_LBLS, PALETTE)):
    axes[1].hist(dist[y == ci], bins=30, alpha=0.65, color=col,
                 label=lb, edgecolor="white")
axes[1].set_title("Distance Distribution by Class")
axes[1].set_xlabel("Distance (km)")
axes[1].set_ylabel("Frequency")
axes[1].legend(fontsize=8)

axes[2].hist(hour, bins=24, range=(0, 24), color="#4C9BE8",
             edgecolor="white", alpha=0.8)
axes[2].axvspan(7,  10, alpha=0.15, color="orange", label="Morning peak")
axes[2].axvspan(17, 21, alpha=0.15, color="red",    label="Evening peak")
axes[2].set_title("Trip Hour Distribution")
axes[2].set_xlabel("Hour of Day")
axes[2].set_ylabel("Frequency")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("charts/01_dataset_distribution.png")
plt.close()
print("   ✓ 01_dataset_distribution.png")


# ── Chart 02: Feature Correlation Heatmap ────────────────────────────
fig, ax = plt.subplots(figsize=(11, 9))
corr = np.corrcoef(X.T)
mask = ~np.tril(np.ones_like(corr, dtype=bool))   # show lower triangle only
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            xticklabels=FEAT_NAMES, yticklabels=FEAT_NAMES,
            ax=ax, linewidths=0.4, square=True,
            annot_kws={"size": 7}, mask=mask)
ax.set_title("Figure 2: Feature Correlation Matrix", fontweight="bold", pad=14)
plt.xticks(rotation=35, ha="right", fontsize=8)
plt.yticks(rotation=0,  fontsize=8)
plt.tight_layout()
plt.savefig("charts/02_feature_correlation.png")
plt.close()
print("   ✓ 02_feature_correlation.png")


# ── Chart 03: Feature Distributions by Class ─────────────────────────
SHOW_IDX   = [0, 1, 2, 4, 5, 6]
SHOW_NAMES = [FEAT_NAMES[i] for i in SHOW_IDX]

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
fig.suptitle("Figure 3: Feature Distributions Across Ride Classes",
             fontweight="bold")

for ai, (fi, fn) in enumerate(zip(SHOW_IDX, SHOW_NAMES)):
    ax  = axes.flatten()[ai]
    bp  = ax.boxplot([X[y == c, fi] for c in range(4)], patch_artist=True,
                     medianprops=dict(color="white", linewidth=2))
    for patch, col in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax.set_title(fn)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(["Auto", "Mini", "Go", "Sedan"], fontsize=8)
    ax.set_ylabel("Value")

plt.tight_layout()
plt.savefig("charts/03_feature_distributions.png")
plt.close()
print("   ✓ 03_feature_distributions.png")


# ── Chart 04: XGBoost Feature Importance ─────────────────────────────
gain_raw  = xgb_model.get_booster().get_score(importance_type="gain")
gain_vals = np.array([gain_raw.get(f"f{i}", 0) for i in range(11)])
gain_pct  = gain_vals / gain_vals.sum() * 100
g_order   = np.argsort(gain_pct)

wt_raw   = xgb_model.get_booster().get_score(importance_type="weight")
wt_vals  = np.array([wt_raw.get(f"f{i}", 0) for i in range(11)])
wt_pct   = wt_vals / wt_vals.sum() * 100
w_order  = np.argsort(wt_pct)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figure 4: XGBoost Feature Importance Analysis",
             fontweight="bold")

g_colors = ["#4C9BE8" if v >= np.percentile(gain_pct, 60) else "#B0C8E8"
            for v in gain_pct[g_order]]
axes[0].barh([FEAT_NAMES[i] for i in g_order], gain_pct[g_order],
             color=g_colors, edgecolor="white")
axes[0].set_title("Gain Importance (%)")
axes[0].set_xlabel("Relative Importance (%)")
for i, v in enumerate(gain_pct[g_order]):
    if v > 0.5:
        axes[0].text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=8)

w_colors = ["#F0A500" if v >= np.percentile(wt_pct, 60) else "#F5D080"
            for v in wt_pct[w_order]]
axes[1].barh([FEAT_NAMES[i] for i in w_order], wt_pct[w_order],
             color=w_colors, edgecolor="white")
axes[1].set_title("Weight (Split Frequency) Importance (%)")
axes[1].set_xlabel("Relative Importance (%)")
for i, v in enumerate(wt_pct[w_order]):
    if v > 0.5:
        axes[1].text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=8)

plt.tight_layout()
plt.savefig("charts/04_xgb_feature_importance.png")
plt.close()
print("   ✓ 04_xgb_feature_importance.png")


# ── Chart 05: Model Comparison Bar Chart ─────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle("Figure 5: Model Performance Comparison", fontweight="bold")

model_names  = [m["name"] for m in ALL]
metric_groups = {
    "Accuracy (%)":  [m["accuracy"]  for m in ALL],
    "Macro F1 (%)":  [m["macro_f1"]  for m in ALL],
    "Precision (%)": [m["macro_pre"] for m in ALL],
    "Recall (%)":    [m["macro_rec"] for m in ALL],
}
metric_cols = ["#4C9BE8", "#3DBE7A", "#F0A500", "#E85D75"]
x       = np.arange(len(model_names))
width   = 0.18
offsets = np.linspace(-1.5, 1.5, 4) * width

for i, (metric_name, values) in enumerate(metric_groups.items()):
    bars = ax.bar(x + offsets[i], values, width,
                  label=metric_name, color=metric_cols[i],
                  edgecolor="white", alpha=0.88)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 108)
ax.legend(loc="lower right", fontsize=9)
ax.axhline(y=80, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
ax.text(len(model_names) - 0.5, 80.6, "80% reference line", fontsize=8, color="gray")

plt.tight_layout()
plt.savefig("charts/05_model_comparison.png")
plt.close()
print("   ✓ 05_model_comparison.png")


# ── Charts 06–08: Confusion Matrices ─────────────────────────────────
def save_confusion_matrix(m, fig_num, filename):
    cm       = m["cm"]
    cm_norm  = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=False, cmap="Blues",
                xticklabels=RIDE_LBLS, yticklabels=RIDE_LBLS,
                ax=ax, linewidths=0.5, linecolor="white",
                vmin=0, vmax=100)

    for i in range(len(RIDE_LBLS)):
        for j in range(len(RIDE_LBLS)):
            pct = cm_norm[i, j]
            cnt = cm[i, j]
            tc  = "white" if pct > 55 else "black"
            ax.text(j + 0.5, i + 0.42, f"{pct:.1f}%",
                    ha="center", va="center", fontsize=10, color=tc, fontweight="bold")
            ax.text(j + 0.5, i + 0.65, f"(n={cnt})",
                    ha="center", va="center", fontsize=7, color=tc)

    ax.set_title(f"Figure {fig_num}: {m['name']} Confusion Matrix\n"
                 f"Accuracy: {m['accuracy']:.2f}%", fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label",      fontsize=10)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"charts/{filename}")
    plt.close()
    print(f"   ✓ {filename}")

save_confusion_matrix(m_xgb,  6, "06_xgb_confusion_matrix.png")
save_confusion_matrix(m_lstm,  7, "07_lstm_confusion_matrix.png")
save_confusion_matrix(m_ens,   8, "08_ensemble_confusion_matrix.png")


# ── Chart 09: ROC Curves ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Figure 9: ROC Curves — One-vs-Rest per Class",
             fontweight="bold")

for ax, m in zip(axes, ALL):
    y_bin = label_binarize(m["y_true"], classes=[0, 1, 2, 3])
    for ci, (lb, col) in enumerate(zip(RIDE_LBLS, PALETTE)):
        fpr, tpr, _ = roc_curve(y_bin[:, ci], m["y_prob"][:, ci])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{lb} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_title(m["name"])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

plt.tight_layout()
plt.savefig("charts/09_roc_curves.png")
plt.close()
print("   ✓ 09_roc_curves.png")


# ── Chart 10: Learning Curves ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Figure 10: Model Learning Curves", fontweight="bold")

# 10a — XGBoost: train vs validation accuracy
train_sizes, tr_scores, val_scores = learning_curve(
    XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.08,
                  eval_metric="mlogloss", random_state=42, verbosity=0),
    X, y, cv=5, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 8),
    n_jobs=-1,
)
tr_m  = tr_scores.mean(axis=1)  * 100
tr_s  = tr_scores.std(axis=1)   * 100
val_m = val_scores.mean(axis=1) * 100
val_s = val_scores.std(axis=1)  * 100

axes[0].plot(train_sizes, tr_m,  "o-", color="#4C9BE8", label="Training accuracy")
axes[0].plot(train_sizes, val_m, "s-", color="#F0A500", label="CV accuracy")
axes[0].fill_between(train_sizes, tr_m-tr_s,   tr_m+tr_s,   alpha=0.15, color="#4C9BE8")
axes[0].fill_between(train_sizes, val_m-val_s, val_m+val_s, alpha=0.15, color="#F0A500")
axes[0].set_title("XGBoost — Training vs CV Accuracy")
axes[0].set_xlabel("Training Set Size")
axes[0].set_ylabel("Accuracy (%)")
axes[0].legend()

# 10b — LSTM: loss per epoch
axes[1].plot(range(1, EPOCHS + 1), lstm_losses, color="#3DBE7A", lw=2)
axes[1].fill_between(range(1, EPOCHS + 1), lstm_losses, alpha=0.15, color="#3DBE7A")
axes[1].set_title("LSTM — Training Loss per Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Cross-Entropy Loss")
axes[1].axhline(y=min(lstm_losses), color="gray", linestyle="--", alpha=0.5)
axes[1].text(EPOCHS * 0.55, min(lstm_losses) + 0.015,
             f"Min loss: {min(lstm_losses):.4f}", fontsize=8, color="gray")

plt.tight_layout()
plt.savefig("charts/10_learning_curve.png")
plt.close()
print("   ✓ 10_learning_curve.png")


# ── Chart 11: Per-Class Precision / Recall / F1 ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Figure 11: Per-Class Precision, Recall & F1-Score",
             fontweight="bold")

for ax, m in zip(axes, ALL):
    rep    = m["report"]
    p_vals = [rep[l]["precision"] * 100 for l in RIDE_LBLS]
    r_vals = [rep[l]["recall"]    * 100 for l in RIDE_LBLS]
    f_vals = [rep[l]["f1-score"]  * 100 for l in RIDE_LBLS]

    xpos = np.arange(len(RIDE_LBLS))
    w    = 0.25
    b1   = ax.bar(xpos - w, p_vals, w, label="Precision", color="#4C9BE8", edgecolor="white")
    b2   = ax.bar(xpos,     r_vals, w, label="Recall",    color="#F0A500", edgecolor="white")
    b3   = ax.bar(xpos + w, f_vals, w, label="F1-Score",  color="#3DBE7A", edgecolor="white")

    for bars in [b1, b2, b3]:
        for bar in bars:
            v = bar.get_height()
            if v > 5:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_title(m["name"])
    ax.set_xticks(xpos)
    ax.set_xticklabels(["Auto", "Mini", "Go", "Sedan"], fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=8)
    ax.axhline(y=80, color="gray", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("charts/11_precision_recall_per_class.png")
plt.close()
print("   ✓ 11_precision_recall_per_class.png")


# ── Chart 12: Predicted Ride vs Distance ─────────────────────────────
NOON_HOUR = 12
NOON_DOW  = 2

def single_trip_X(d):
    is_we_ = 0;  is_pk_ = 0;  is_nt_ = 0
    auto_  = int(d <= 15)
    return np.array([[d, NOON_HOUR, NOON_DOW, is_we_, is_pk_, is_nt_,
                      auto_, d**2, d*is_pk_, d*is_nt_, NOON_HOUR*is_we_]])

def single_lstm_seq(d):
    seq = np.zeros((SEQ_LEN, N_FEAT_S), dtype=np.float32)
    seq[-1] = [d / 50., NOON_HOUR / 23., NOON_DOW / 6., 0.5]
    return seq

dist_range  = np.linspace(0.5, 40, 300)
xgb_preds   = [int(np.argmax(xgb_model.predict_proba(single_trip_X(d))[0]))
               for d in dist_range]
lstm_preds  = [int(np.argmax(lstm_predict(single_lstm_seq(d))))
               for d in dist_range]
ens_preds   = [int(np.argmax(
                    0.6 * xgb_model.predict_proba(single_trip_X(d))[0]
                    + 0.4 * lstm_predict(single_lstm_seq(d))))
               for d in dist_range]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Figure 12: Predicted Ride Class Across Distance Range",
             fontweight="bold")

for ax, (preds, name) in zip(axes,
        [(xgb_preds, "XGBoost"),
         (lstm_preds, "LSTM"),
         (ens_preds,  "Ensemble")]):

    preds_arr = np.array(preds)
    for ci, (lb, col) in enumerate(zip(RIDE_LBLS, PALETTE)):
        mask = preds_arr == ci
        if mask.any():
            ax.scatter(dist_range[mask], np.full(mask.sum(), ci),
                       c=col, s=14, alpha=0.75, label=lb)

    ax.set_title(name)
    ax.set_xlabel("Distance (km)")
    ax.set_yticks(range(4))
    ax.set_yticklabels(RIDE_LBLS, fontsize=9)
    ax.set_ylabel("Predicted Ride")
    ax.axvline(x=15, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.text(15.3, 3.6, "Auto limit\n(15 km)", fontsize=7, color="gray")
    ax.set_xlim(0, 41)
    ax.legend(fontsize=7, loc="upper left")

plt.tight_layout()
plt.savefig("charts/12_distance_vs_prediction.png")
plt.close()
print("   ✓ 12_distance_vs_prediction.png")


# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 58)
print("  COMPLETE METRICS SUMMARY")
print("=" * 58)
print(f"\n  {'Model':<12} {'Accuracy':>10} {'Macro F1':>10} {'Precision':>11} {'Recall':>9}")
print("  " + "-" * 56)
for m in ALL:
    print(f"  {m['name']:<12} {m['accuracy']:>9.2f}%  {m['macro_f1']:>8.2f}%  "
          f"{m['macro_pre']:>9.2f}%  {m['macro_rec']:>8.2f}%")

print(f"\n  XGBoost — Per-class breakdown:")
print(f"  {'Class':<14} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
print("  " + "-" * 47)
for lb in RIDE_LBLS:
    r = m_xgb["report"][lb]
    print(f"  {lb:<14} {r['precision']*100:>9.1f}%  {r['recall']*100:>6.1f}%  "
          f"{r['f1-score']*100:>6.1f}%  {r['support']:>8}")

print(f"\n  Top features by XGBoost gain:")
top3 = np.argsort(gain_pct)[::-1][:3]
for rank, fi in enumerate(top3, 1):
    print(f"    {rank}. {FEAT_NAMES[fi]:<22}  {gain_pct[fi]:.1f}%")

print("\n" + "=" * 58)
print(f"  12 charts saved to: ./charts/")
print("=" * 58)