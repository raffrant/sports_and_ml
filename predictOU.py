
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

N_FORM = 5  # rolling window


# ══════════════════════════════════════════════════════════════════════
# ROLLING STATS — fixed: threshold 1 (was 2), added BTTS + clean sheet
# ══════════════════════════════════════════════════════════════════════
def rolling_team_stats(df, team, before_date, n=N_FORM):
    past = df[
        ((df["home"] == team) | (df["away"] == team)) &
        (df["date"] < before_date)
    ].tail(n)

    if len(past) < 1:
        return None

    scored, conceded, totals, points = [], [], [], []
    btts, clean_sheets, failed_score  = [], [], []

    for _, row in past.iterrows():
        is_home = row["home"] == team
        gs = row["home_score"] if is_home else row["away_score"]
        gc = row["away_score"] if is_home else row["home_score"]
        scored.append(gs);   conceded.append(gc)
        totals.append(row["total_goals"])
        points.append(3 if gs > gc else (1 if gs == gc else 0))
        btts.append(1 if gs > 0 and gc > 0 else 0)          # both teams scored
        clean_sheets.append(1 if gc == 0 else 0)             # kept clean sheet
        failed_score.append(1 if gs == 0 else 0)             # failed to score

    return {
        "avg_scored":     np.mean(scored),
        "avg_conceded":   np.mean(conceded),
        "avg_total":      np.mean(totals),
        "over25_rate":    np.mean([1 if t > 2.5 else 0 for t in totals]),
        "over15_rate":    np.mean([1 if t > 1.5 else 0 for t in totals]),
        "avg_points":     np.mean(points),
        "btts_rate":      np.mean(btts),       # ✅ new
        "clean_sheet_rate": np.mean(clean_sheets),  # ✅ new
        "failed_score_rate": np.mean(failed_score), # ✅ new
        "scoring_consistency": np.std(scored),      # ✅ new — lower = more consistent
    }


# ══════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — fixed column names, added 10 new features
# ══════════════════════════════════════════════════════════════════════
def build_features(df, n=N_FORM):
    rows = []
    for _, match in df.iterrows():
        date, home, away = match["date"], match["home"], match["away"]

        h = rolling_team_stats(df, home, date, n)
        a = rolling_team_stats(df, away, date, n)
        if not h or not a:
            continue

        # H2H
        h2h = df[
            (((df["home"] == home) & (df["away"] == away)) |
             ((df["home"] == away) & (df["away"] == home))) &
            (df["date"] < date)
        ].tail(6)
        h2h_avg    = h2h["total_goals"].mean() if len(h2h) > 0 else 2.5
        h2h_over25 = (h2h["total_goals"] > 2.5).mean() if len(h2h) > 0 else 0.5
        h2h_btts   = ((h2h["home_score"] > 0) & (h2h["away_score"] > 0)).mean() \
                      if len(h2h) > 0 else 0.5                                 # ✅ new

        rows.append({
            # ── Home rolling stats ──────────────────────────────────
            "h_avg_scored":       h["avg_scored"],
            "h_avg_conceded":     h["avg_conceded"],
            "h_avg_total":        h["avg_total"],
            "h_over25_rate":      h["over25_rate"],
            "h_over15_rate":      h["over15_rate"],
            "h_avg_points":       h["avg_points"],
            "h_btts_rate":        h["btts_rate"],
            "h_clean_sheet_rate": h["clean_sheet_rate"],
            "h_failed_score_rate":h["failed_score_rate"],
            "h_consistency":      h["scoring_consistency"],

            # ── Away rolling stats ──────────────────────────────────
            "a_avg_scored":       a["avg_scored"],
            "a_avg_conceded":     a["avg_conceded"],
            "a_avg_total":        a["avg_total"],
            "a_over25_rate":      a["over25_rate"],
            "a_over15_rate":      a["over15_rate"],
            "a_avg_points":       a["avg_points"],
            "a_btts_rate":        a["btts_rate"],
            "a_clean_sheet_rate": a["clean_sheet_rate"],
            "a_failed_score_rate":a["failed_score_rate"],
            "a_consistency":      a["scoring_consistency"],

            # ── Combined signals ────────────────────────────────────
            "combined_avg_scored":   h["avg_scored"]   + a["avg_scored"],
            "combined_avg_conceded": h["avg_conceded"]  + a["avg_conceded"],
            "goal_expectation":      h["avg_scored"]   + a["avg_scored"]
                                   + h["avg_conceded"] + a["avg_conceded"],
            "combined_btts_rate":    h["btts_rate"]    + a["btts_rate"],  # ✅ new
            "defensive_weakness":    h["avg_conceded"] + a["avg_conceded"],  # ✅ new
            "attacking_strength":    h["avg_scored"]   + a["avg_scored"],    # ✅ new
            "combined_over15_rate":  h["over15_rate"]  + a["over15_rate"],   # ✅ new
            "both_clean_sheets":     h["clean_sheet_rate"] * a["clean_sheet_rate"],  # ✅ new
            "both_fail_to_score":    h["failed_score_rate"] * a["failed_score_rate"],  # ✅ new

            # ── H2H ─────────────────────────────────────────────────
            "h2h_avg_total":   h2h_avg,
            "h2h_over25_rate": h2h_over25,
            "h2h_btts_rate":   h2h_btts,

            # ── Context ─────────────────────────────────────────────
            "matchday":        match["matchday"],

            # ── Targets ─────────────────────────────────────────────
            "over25":          match["over25"],
            "winner":          match.get("winner", None),

            # ── Metadata ────────────────────────────────────────────
            "date":            date,
            "home":            home,
            "away":            away,
            "total_goals":     match["total_goals"],
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# TRAIN — fixed: auto-detect features, ROC-AUC primary, class balancing
# ══════════════════════════════════════════════════════════════════════
META_COLS = {"over25", "winner", "date", "home", "away", "total_goals"}

def train_models(feat_df):
    feat_df  = feat_df.dropna(subset=["over25"])
    FEATURES = [c for c in feat_df.columns if c not in META_COLS]  # ✅ auto-detect
    X, y     = feat_df[FEATURES], feat_df["over25"]
    imb      = (y == 0).sum() / max((y == 1).sum(), 1)

    print(f"\n📊 Dataset  : {len(feat_df)} matches | {len(FEATURES)} features")
    print(f"   Over 2.5  : {y.mean():.1%}  |  Under 2.5: {(1-y.mean()):.1%}\n")

    models = {
        "Logistic Regression": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, C=0.1,
                class_weight="balanced"   # ✅ fixes Under 2.5 recall
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(
                n_estimators=300, max_depth=3,
                learning_rate=0.03, subsample=0.8, random_state=42
            ))
        ]),
        "XGBoost": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=imb,     # ✅ handles class imbalance
                eval_metric="logloss", random_state=42, verbosity=0,
            ))
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model, best_auc = None, 0

    for name, model in models.items():
        acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
        auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()  # ✅ primary
        f1  = cross_val_score(model, X, y, cv=cv, scoring="f1").mean()
        print(f"  {name:22s} → Acc: {acc:.3f} | AUC: {auc:.3f} | F1: {f1:.3f}")
        if auc > best_auc:
            best_auc, best_model = auc, (name, model)

    print(f"\n  ✅ Best: {best_model[0]} (AUC = {best_auc:.3f})")
    best_model[1].fit(X, y)

    print("\nClassification Report:")
    print(classification_report(
        y, best_model[1].predict(X),
        target_names=["Under 2.5", "Over 2.5"]
    ))

    # Feature importance
    try:
        clf = best_model[1].named_steps["clf"]
        imp = pd.Series(clf.feature_importances_, index=FEATURES)
        print("Top 10 features:")
        print(imp.sort_values(ascending=False).head(10).to_string())
    except Exception:
        pass

    return best_model[1], FEATURES


# ══════════════════════════════════════════════════════════════════════
# OUTCOME MODEL — new: predicts Home Win / Draw / Away Win
# ══════════════════════════════════════════════════════════════════════
def train_outcome_model(feat_df):
    feat_df = feat_df.dropna(subset=["winner"]).copy()
    feat_df["result_code"] = feat_df["winner"].map({
        "HOME_TEAM": 2, "DRAW": 1, "AWAY_TEAM": 0
    })
    feat_df = feat_df.dropna(subset=["result_code"])
    FEATURES = [c for c in feat_df.columns if c not in META_COLS | {"result_code"}]
    X, y = feat_df[FEATURES], feat_df["result_code"].astype(int)

    model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", random_state=42, verbosity=0,
        ))
    ])

    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
    print(f"\n  Outcome model (3-class) → Acc: {acc:.3f}")
    model.fit(X, y)
    print(classification_report(y, model.predict(X),
                                  target_names=["Away Win","Draw","Home Win"]))
    return model, FEATURES


# ══════════════════════════════════════════════════════════════════════
# PREDICT — fixed bug in past filter, added outcome, full card output
# ══════════════════════════════════════════════════════════════════════
def predict_next_match(
    ou_model, ou_features,
    outcome_model, outcome_features,
    df, home_team, away_team, matchday=35,
):
    today = pd.Timestamp.today()

    h = rolling_team_stats(df, home_team, today, n=N_FORM)
    a = rolling_team_stats(df, away_team, today, n=N_FORM)

    # ✅ Fixed: was checking (home OR away) instead of each separately
    if h is None:
        print(f"❌ No history for '{home_team}'.")
        print("Teams:", sorted(df["home"].unique().tolist()))
        return
    if a is None:
        print(f"❌ No history for '{away_team}'.")
        print("Teams:", sorted(df["home"].unique().tolist()))
        return

    h2h = df[
        (((df["home"] == home_team) & (df["away"] == away_team)) |
         ((df["home"] == away_team) & (df["away"] == home_team)))
    ].tail(6)

    row = {
        "h_avg_scored":       h["avg_scored"],
        "h_avg_conceded":     h["avg_conceded"],
        "h_avg_total":        h["avg_total"],
        "h_over25_rate":      h["over25_rate"],
        "h_over15_rate":      h["over15_rate"],
        "h_avg_points":       h["avg_points"],
        "h_btts_rate":        h["btts_rate"],
        "h_clean_sheet_rate": h["clean_sheet_rate"],
        "h_failed_score_rate":h["failed_score_rate"],
        "h_consistency":      h["scoring_consistency"],
        "a_avg_scored":       a["avg_scored"],
        "a_avg_conceded":     a["avg_conceded"],
        "a_avg_total":        a["avg_total"],
        "a_over25_rate":      a["over25_rate"],
        "a_over15_rate":      a["over15_rate"],
        "a_avg_points":       a["avg_points"],
        "a_btts_rate":        a["btts_rate"],
        "a_clean_sheet_rate": a["clean_sheet_rate"],
        "a_failed_score_rate":a["failed_score_rate"],
        "a_consistency":      a["scoring_consistency"],
        "combined_avg_scored":   h["avg_scored"]   + a["avg_scored"],
        "combined_avg_conceded": h["avg_conceded"]  + a["avg_conceded"],
        "goal_expectation":      h["avg_scored"]   + a["avg_scored"]
                               + h["avg_conceded"] + a["avg_conceded"],
        "combined_btts_rate":    h["btts_rate"]    + a["btts_rate"],
        "defensive_weakness":    h["avg_conceded"] + a["avg_conceded"],
        "attacking_strength":    h["avg_scored"]   + a["avg_scored"],
        "combined_over15_rate":  h["over15_rate"]  + a["over15_rate"],
        "both_clean_sheets":     h["clean_sheet_rate"] * a["clean_sheet_rate"],
        "both_fail_to_score":    h["failed_score_rate"] * a["failed_score_rate"],
        "h2h_avg_total":         h2h["total_goals"].mean() if len(h2h) > 0 else 2.5,
        "h2h_over25_rate":       (h2h["total_goals"] > 2.5).mean() if len(h2h) > 0 else 0.5,
        "h2h_btts_rate":         ((h2h["home_score"]>0) & (h2h["away_score"]>0)).mean()
                                  if len(h2h) > 0 else 0.5,
        "matchday":              matchday,
    }

    # Over/Under
    X_ou  = pd.DataFrame([row])[ou_features]
    p_ou  = ou_model.predict_proba(X_ou)[0]

    # Outcome
    X_res = pd.DataFrame([row])[outcome_features]
    p_res = outcome_model.predict_proba(X_res)[0]  # [Away, Draw, Home]

    # Print prediction card
    ou_label = "⚽ OVER 2.5" if p_ou[1] >= 0.5 else "🔒 UNDER 2.5"
    outcomes = [
        (f"🏠 {home_team[:20]} Win", p_res[2]),
        ("🤝 Draw",                  p_res[1]),
        (f"✈️  {away_team[:20]} Win", p_res[0]),
    ]

    print(f"\n{'═'*52}")
    print(f"  {home_team}")
    print(f"  vs {away_team}  (MD {matchday})")
    print(f"  {'─'*48}")

    print(f"\n  RESULT")
    for label, prob in sorted(outcomes, key=lambda x: -x[1]):
        bar = "█" * int(prob * 24)
        print(f"  {label:<28} {prob:.1%}  {bar}")

    print(f"\n  GOALS")
    print(f"  {ou_label}")
    print(f"  Over 2.5  {p_ou[1]:.1%}  {'█' * int(p_ou[1]*24)}")
    print(f"  Under 2.5 {p_ou[0]:.1%}  {'█' * int(p_ou[0]*24)}")

    print(f"\n  FORM (last {N_FORM})")
    print(f"  {home_team[:22]:22s}  scored {h['avg_scored']:.2f}  conceded {h['avg_conceded']:.2f}  BTTS {h['btts_rate']:.0%}")
    print(f"  {away_team[:22]:22s}  scored {a['avg_scored']:.2f}  conceded {a['avg_conceded']:.2f}  BTTS {a['btts_rate']:.0%}")
    print(f"  Goal expectation: {row['goal_expectation']:.2f}")

    if len(h2h) > 0:
        h_wins = ((h2h["home"] == home_team) & (h2h["winner"] == "HOME_TEAM")).sum() + \
                 ((h2h["away"] == home_team) & (h2h["winner"] == "AWAY_TEAM")).sum()
        a_wins = len(h2h) - h_wins - (h2h["winner"] == "DRAW").sum()
        print(f"\n  H2H (last {len(h2h)})  {home_team[:14]} {h_wins}W – "
              f"{(h2h['winner']=='DRAW').sum()}D – {a_wins}W {away_team[:14]}")
        print(f"  Avg goals: {h2h['total_goals'].mean():.2f}  BTTS: {row['h2h_btts_rate']:.0%}")

    best = max(outcomes, key=lambda x: x[1])
    print(f"\n  💰 KELLY HINT")
    print(f"  Bet '{best[0].strip()}' only if bookmaker implied prob < {best[1]:.1%}")
    print(f"{'═'*52}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = pd.read_csv("pl_matches.csv")
    df = df[df["status"] == "FINISHED"].copy()
    df["date"]        = pd.to_datetime(df["date"])
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["over25"]      = (df["total_goals"] > 2.5).astype(int)
    df = df.sort_values("date").reset_index(drop=True)

    feat_df = build_features(df, n=N_FORM)

    print("\n── Over/Under 2.5 ─────────────────────────────────────")
    ou_model, ou_features = train_models(feat_df)

    print("\n── Match Outcome ───────────────────────────────────────")
    outcome_model, outcome_features = train_outcome_model(feat_df)

    # ✏️ Change to your next match exact team names
    print("\nAvailable teams:")
    print(sorted(df["home"].unique().tolist()))

    predict_next_match(
        ou_model, ou_features,
        outcome_model, outcome_features,
        df,
        home_team = "Arsenal FC",
        away_team = "Fulham FC",
        matchday  = 35,
    )
