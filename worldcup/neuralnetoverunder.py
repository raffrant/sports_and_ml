import requests, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE    = "https://api.football-data.org/v4"
API_KEY = "YOUR_TOKEN"
HEADERS = {"X-Auth-Token": "mytoken"}
PL_CODE = "PL"

def get(endpoint: str) -> dict:
    r = requests.get(f"{BASE}{endpoint}", headers=HEADERS, timeout=10)
    r.raise_for_status()
    time.sleep(7)
    return r.json()

def get_matches(season: int = 2024, pl_code: str = PL_CODE) -> pd.DataFrame:
    data = get(f"/competitions/{pl_code}/matches?season={season}")
    rows = []
    for m in data["matches"]:
        if m["status"] != "FINISHED":
            continue
        rows.append({
            "match_id":   m["id"],
            "date":       m["utcDate"][:10],
            "matchday":   m["matchday"],
            "home":       m["homeTeam"]["name"],
            "away":       m["awayTeam"]["name"],
            "home_score": m["score"]["fullTime"]["home"],
            "away_score": m["score"]["fullTime"]["away"],
            "winner":     m["score"]["winner"],
            "season":     season,
        })
    return pd.DataFrame(rows)

def build_dataset(seasons: list[int] = [2023, 2024], pl_code: str = PL_CODE) -> pd.DataFrame:
    frames = []
    for s in seasons:
        print(f"Fetching season {s}...")
        frames.append(get_matches(s, pl_code))
    df = pd.concat(frames, ignore_index=True)
    df["date"]        = pd.to_datetime(df["date"])
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["over25"]      = (df["total_goals"] > 2.5).astype(int)
    return df.sort_values("date").reset_index(drop=True)

# ─── 2. Rolling Features ─────────────────────────────────────────────────────

def rolling_team_features(df: pd.DataFrame, team: str, before_date, n: int = 5) -> dict:
    mask = (
        ((df["home"] == team) | (df["away"] == team)) &
        (df["date"] < before_date) &
        (df["home_score"].notna())
    )
    recent = df[mask].sort_values("date").tail(n)

    if len(recent) == 0:
        return {
            "avg_gf": np.nan, "avg_ga": np.nan,
            "avg_total": np.nan, "over25_rate": np.nan,
            "win_rate": np.nan,
        }

    gf, ga, wins = [], [], 0
    for _, row in recent.iterrows():
        if row["home"] == team:
            gf.append(row["home_score"])
            ga.append(row["away_score"])
            wins += int(row["winner"] == "HOME_TEAM")
        else:
            gf.append(row["away_score"])
            ga.append(row["home_score"])
            wins += int(row["winner"] == "AWAY_TEAM")

    totals = [g + a for g, a in zip(gf, ga)]
    return {
        "avg_gf":      np.mean(gf),
        "avg_ga":      np.mean(ga),
        "avg_total":   np.mean(totals),
        "over25_rate": np.mean([t > 2.5 for t in totals]),
        "win_rate":    wins / len(recent),
    }

def build_features(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    rows = []
    for _, match in df.iterrows():
        h = rolling_team_features(df, match["home"], match["date"], n)
        a = rolling_team_features(df, match["away"], match["date"], n)

        rows.append({
            "match_id":             match["match_id"],
            "date":                 match["date"],
            "home":                 match["home"],
            "away":                 match["away"],
            "season":               match["season"],
            "over25":               match["over25"],
            "total_goals":          match["total_goals"],
            "home_score":           match["home_score"],   # ✅ flat column
            "away_score":           match["away_score"],   # ✅ flat column
            # home rolling
            "h_avg_gf":             h["avg_gf"],
            "h_avg_ga":             h["avg_ga"],
            "h_avg_total":          h["avg_total"],
            "h_over25_rate":        h["over25_rate"],
            "h_win_rate":           h["win_rate"],
            # away rolling
            "a_avg_gf":             a["avg_gf"],
            "a_avg_ga":             a["avg_ga"],
            "a_avg_total":          a["avg_total"],
            "a_over25_rate":        a["over25_rate"],
            "a_win_rate":           a["win_rate"],
            # combined
            "combined_avg_total":   (h["avg_total"]   + a["avg_total"])   / 2,
            "combined_over25_rate": (h["over25_rate"] + a["over25_rate"]) / 2,
            "winner":               match["winner"],
        })

    return pd.DataFrame(rows).dropna()

# ─── 3. Model ────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "h_avg_gf", "h_avg_ga", "h_avg_total", "h_over25_rate", "h_win_rate",
    "a_avg_gf", "a_avg_ga", "a_avg_total", "a_over25_rate", "a_win_rate",
    "combined_avg_total", "combined_over25_rate",
]

class OverUnderNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

def train_model(feat_df: pd.DataFrame, epochs: int = 80, lr: float = 1e-3):
    X = feat_df[FEATURE_COLS].values.astype(np.float32)
    y = feat_df["over25"].values.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    train_dl = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=64, shuffle=True
    )

    model     = OverUnderNet(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_val)).numpy()
                acc   = ((preds > 0.5) == y_val).mean()
            print(f"Epoch {epoch+1:3d} | Val Acc: {acc:.3f}")

    return model, scaler

# ─── 4. Predict ──────────────────────────────────────────────────────────────

def predict_match(
    df: pd.DataFrame,
    model,
    scaler,
    home: str,
    away: str,
    match_date: str,
    n: int = 5,
    edge_threshold: float = 0.60,
):
    date = pd.Timestamp(match_date)
    h = rolling_team_features(df, home, date, n)
    a = rolling_team_features(df, away, date, n)

    features = np.array([[
        h["avg_gf"],  h["avg_ga"],  h["avg_total"],  h["over25_rate"],  h["win_rate"],
        a["avg_gf"],  a["avg_ga"],  a["avg_total"],  a["over25_rate"],  a["win_rate"],
        (h["avg_total"]   + a["avg_total"])   / 2,
        (h["over25_rate"] + a["over25_rate"]) / 2,
    ]], dtype=np.float32)

    features_scaled = scaler.transform(features)
    model.eval()
    with torch.no_grad():
        prob_over  = model(torch.tensor(features_scaled)).item()
    prob_under = 1 - prob_over

    print(f"\n{home} vs {away} — {match_date}")
    print(f"  P(over 2.5)  = {prob_over:.2%}")
    print(f"  P(under 2.5) = {prob_under:.2%}")

    if prob_over >= edge_threshold:
        print(f"  ✅ BET: OVER 2.5  (edge = {prob_over:.2%})")
    elif prob_under >= edge_threshold:
        print(f"  ✅ BET: UNDER 2.5 (edge = {prob_under:.2%})")
    else:
        print(f"  ⚠️  NO EDGE — skip this match")

    return prob_over


if __name__ == "__main__":
    pass
