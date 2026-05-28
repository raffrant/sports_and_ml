# 🏀 NBA Over/Under ML Predictor

Predict whether an NBA game will go **Over or Under** the sportsbook total using real historical game data and a machine learning pipeline with Kelly Criterion bet sizing.

---

## 📁 Files

| File | Description |
|------|-------------|
| `nba_overunder_ml.py` | Core ML pipeline: data generation, model training, prediction, Kelly Criterion |
| `specificgame.ipynb` | Notebook to predict a specific real matchup tonight |

---

## ⚙️ How It Works

### 1. Training Data — Real NBA Games
The model trains on **real historical box scores** from the 2022-23 and 2023-24 NBA regular seasons, pulled via `nba_api`.

Each row is one game and contains:

| Feature | Description |
|---------|-------------|
| `home_pace` / `away_pace` | Approx. possessions per game (FGA − OREB + TOV + 0.44×FTA) |
| `home_efg_pct` / `away_efg_pct` | Effective FG% — accounts for 3-point value |
| `home_reb_pct` / `away_reb_pct` | Share of total rebounds |
| `sportsbook_line` | Simulated Vegas total (based on real point totals ± noise) |

### 2. Models
Two Random Forest models are trained:
- **Regressor** → predicts total points scored
- **Classifier** → predicts probability the game goes Over

### 3. Prediction on a Specific Matchup
You fetch current season stats for two teams via `LeagueDashTeamStats`, build a matchup feature row using the same feature schema, and call `predict_specific_matchup()`.

### 4. Kelly Criterion
The classifier's probability is used to calculate the optimal bet size:

f* = (b × p − q) / b

Where `b = decimal odds − 1`, `p = P(Over)`, `q = 1 − p`. A quarter-Kelly fraction is applied to manage variance.

---

## 🚀 Quickstart

### Install dependencies
```bash
pip install nba_api scikit-learn pandas numpy
```

### Run the full pipeline
```bash
python nba_overunder_ml.py
```

### Predict a specific game (Notebook)
Open `specificgame.ipynb` and run all cells. Edit the team names and sportsbook line in the last cell:

```python
predict_specific_matchup(
    team_a_stats=newceltics_dict,
    team_b_stats=newnuggets_dict,
    situational_context=context,
    sportsbook_line=215.5,
    sportsbook_odds_decimal=1.909,
    reg_model=reg_model,
    clf_model=clf_model
)
```

### Output example
