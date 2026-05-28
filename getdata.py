import requests
import pandas as pd
import time

API_KEY  = "YOUR_FREE_API_KEY"   # from football-data.org
BASE     = "https://api.football-data.org/v4"
HEADERS  = {"X-Auth-Token": #toten}
PL_CODE  = "PL"   # Premier League competition code

# ─── Helper ────────────────────────────────────────────────────────────────────
def get(endpoint: str) -> dict:
    r = requests.get(f"{BASE}{endpoint}", headers=HEADERS, timeout=10)
    r.raise_for_status()
    time.sleep(7)  # Free tier: 10 calls/min max
    return r.json()


# ─── 1. Standings ───────────────────────────────────────────────────────────────
def get_standings(season: int = 2024) -> pd.DataFrame:
    data = get(f"/competitions/{PL_CODE}/standings?season={season}")
    rows = []
    for entry in data["standings"][0]["table"]:
        rows.append({
            "pos":        entry["position"],
            "team":       entry["team"]["name"],
            "played":     entry["playedGames"],
            "won":        entry["won"],
            "draw":       entry["draw"],
            "lost":       entry["lost"],
            "gf":         entry["goalsFor"],
            "ga":         entry["goalsAgainst"],
            "gd":         entry["goalDifference"],
            "points":     entry["points"],
            "form":       entry.get("form", ""),
        })
    return pd.DataFrame(rows)


# ─── 2. All Matches (full season) ───────────────────────────────────────────────
def get_matches(season: int = 2024) -> pd.DataFrame:
    data = get(f"/competitions/{PL_CODE}/matches?season={season}")
    rows = []
    for m in data["matches"]:
        rows.append({
            "date":        m["utcDate"][:10],
            "matchday":    m["matchday"],
            "home":        m["homeTeam"]["name"],
            "away":        m["awayTeam"]["name"],
            "status":      m["status"],
            "home_score":  m["score"]["fullTime"]["home"],
            "away_score":  m["score"]["fullTime"]["away"],
            "winner":      m["score"]["winner"],
        })
    return pd.DataFrame(rows)


# ─── 3. Top Scorers ──────────────────────────────────────────────────────────────
def get_scorers(season: int = 2024, limit: int = 20) -> pd.DataFrame:
    data = get(f"/competitions/{PL_CODE}/scorers?season={season}&limit={limit}")
    rows = []
    for entry in data["scorers"]:
        p = entry["player"]
        rows.append({
            "player":      p["name"],
            "nationality": p["nationality"],
            "team":        entry["team"]["name"],
            "goals":       entry["goals"],
            "assists":     entry.get("assists", 0),
            "penalties":   entry.get("penalties", 0),
        })
    return pd.DataFrame(rows)


# ─── 4. Head-to-Head ─────────────────────────────────────────────────────────────
def get_h2h(match_id: int, limit: int = 10) -> pd.DataFrame:
    data = get(f"/matches/{match_id}/head2head?limit={limit}")
    rows = []
    for m in data["matches"]:
        rows.append({
            "date":       m["utcDate"][:10],
            "home":       m["homeTeam"]["name"],
            "away":       m["awayTeam"]["name"],
            "home_score": m["score"]["fullTime"]["home"],
            "away_score": m["score"]["fullTime"]["away"],
            "winner":     m["score"]["winner"],
        })
    return pd.DataFrame(rows)


# ─── 5. Multi-season loop ────────────────────────────────────────────────────────
def build_dataset(seasons: list[int]) -> pd.DataFrame:
    all_matches = []
    for season in seasons:
        print(f"Fetching season {season}...")
        df = get_matches(season)
        df["season"] = season
        all_matches.append(df)
    return pd.concat(all_matches, ignore_index=True)


# ─── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Standings
    standings = get_standings(2025)
    standings.to_csv("pl_standings.csv", index=False)
    print(standings.to_string(index=False))

    # Full match dataset across 4 seasons
    matches = build_dataset([2025])
    matches.to_csv("pl_matches.csv", index=False)
    print(f"\n✅ {len(matches)} matches saved → pl_matches.csv")
