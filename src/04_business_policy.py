"""
04_business_policy.py
Translates the tree's risk_score (probability of NOT meeting the target) into an
operational traffic light per team and estimates the annual impact for the
IMMEX plant manager.

Run it with  python src/04_business_policy.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
pred = pd.read_csv(ROOT / "data" / "processed" / "test_predictions.csv")
FIG = ROOT / "reports" / "figures"
y, p = pred["missed_target"], pred["risk_score"]

# ---- 1. Threshold sweep -----------------------------------------------------
rows = []
for t in np.arange(0.10, 0.91, 0.05):
    yhat = (p >= t).astype(int)
    rows.append({"umbral": round(t, 2),
                 "alertas_%": yhat.mean() * 100,
                 "recall": recall_score(y, yhat, zero_division=0),
                 "precision": precision_score(y, yhat, zero_division=0)})
sweep = pd.DataFrame(rows)
print(sweep.round(2).to_string(index=False))
sweep.to_csv(ROOT / "reports" / "threshold_sweep.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(sweep["umbral"], sweep["recall"], "o-", label="Recall (non-compliance detected)")
ax.plot(sweep["umbral"], sweep["precision"], "s-", label="Precision (correct alerts)")
ax.plot(sweep["umbral"], sweep["alertas_%"] / 100, "^--", color="gray",
        label="% of shifts with an alert")
for t, c in [(0.30, "#e6a700"), (0.60, "#c0504d")]:
    ax.axvline(t, color=c, ls=":", lw=2)
ax.set_xlabel("Risk score threshold")
ax.set_ylabel("Proportion")
ax.set_title("Threshold trade-off for the traffic light")
ax.legend(loc="center right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "policy_threshold_tradeoff.png", dpi=200)
plt.close(fig)

# ---- 2. Traffic light -------------------------------------------------------
YELLOW, RED = 0.30, 0.60
pred["traffic_light"] = pd.cut(p, [-0.01, YELLOW, RED, 1.01],
                               labels=["Green", "Yellow", "Red"])
light = pred.groupby("traffic_light", observed=False).agg(
    shifts=("missed_target", "size"),
    misses=("missed_target", "sum"),
)
light["actual_miss_rate"] = light["misses"] / light["shifts"]
light["%_of_all_misses"] = light["misses"] / y.sum()
print("\nTRAFFIC LIGHT ON TEST SET\n", light.round(3).to_string())
light.to_csv(ROOT / "reports" / "traffic_light_test.csv")

# ---- 3. Annual impact (formula from the proposal) ---------------------------
ANNUAL_EVENTS = 904                      # 12 teams x 280 days x 26.9%
caught = light.loc[["Yellow", "Red"], "misses"].sum() / y.sum()
TARGET_REDUCTION = 0.175                 # target from the proposal (15 to 20%)
needed_effectiveness = TARGET_REDUCTION / caught
print(f"\nTraffic light coverage (yellow + red) is {caught:.1%} of the non-compliance events")
print(f"To achieve a 17.5% reduction in events, the interventions must fix "
      f"{needed_effectiveness:.1%} of the alerted cases")
print(f"Events avoided per year (target) {ANNUAL_EVENTS * TARGET_REDUCTION:.0f}")

# Sensitivity to the cost per event (ASSUMPTION, the team must validate this value)
for cost in [5_000, 10_000, 20_000]:
    print(f"  If each event costs ${cost:,} MXN, the annual savings are "
          f"${ANNUAL_EVENTS * TARGET_REDUCTION * cost:,.0f} MXN")

# Impact summary for slide 2
impact = pd.DataFrame({"cost_per_event_mxn": [5_000, 10_000, 20_000]})
impact["annual_savings_mxn"] = impact["cost_per_event_mxn"] * ANNUAL_EVENTS * TARGET_REDUCTION
impact.to_csv(ROOT / "reports" / "impact_summary.csv", index=False)
print("\nSaved to reports/impact_summary.csv")