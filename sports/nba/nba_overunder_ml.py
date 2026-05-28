"""
NBA Over/Under Prediction & Kelly Criterion ML Pipeline.

Goal:
----
Generate a synthetic dataset representing NBA matchups with advanced stats,
use machine learning to predict total points and whether the game goes Over or Under,
and calculate the recommended bet size using the Kelly Criterion.

Main outputs:
------------
1. An NBA-level dataset (one row per game matchup).
2. Regression model for predicted Total Points.
3. Classification model for Over/Under prediction.
4. Kelly Criterion stake sizing based on the classifier's predicted probabilities.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------
# 1. DATA GENERATION (Simulating NBA Advanced Stats)
# ---------------------------------------------------------------------

import pandas as pd
import numpy as np
import time
from nba_api.stats.endpoints import leaguegamelog

def generate_nba_dataset(seasons=['2023-24', '2022-23']) -> pd.DataFrame:
    """
    Generate NBA matchup data using REAL historical box scores from nba_api.
    The output columns are aligned with get_team_stats().
    """
    dfs = []

    for season in seasons:
        log = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star='Regular Season'
        ).get_data_frames()[0]
        dfs.append(log)
        time.sleep(0.6)

    all_logs = pd.concat(dfs, ignore_index=True)

    home_games = all_logs[all_logs['MATCHUP'].str.contains(' vs. ')].copy()
    away_games = all_logs[all_logs['MATCHUP'].str.contains(' @ ')].copy()

    games = pd.merge(
        home_games,
        away_games,
        on=['GAME_ID', 'GAME_DATE'],
        suffixes=('_home', '_away')
    )

    df = pd.DataFrame()
    df['game_id'] = games['GAME_ID']

    df['home_fga'] = games['FGA_home']
    df['away_fga'] = games['FGA_away']
    df['home_3pa'] = games['FG3A_home']
    df['away_3pa'] = games['FG3A_away']

    df['home_2p_pct'] = np.where(
        (games['FGA_home'] - games['FG3A_home']) > 0,
        (games['FGM_home'] - games['FG3M_home']) / (games['FGA_home'] - games['FG3A_home']),
        np.nan
    )
    df['away_2p_pct'] = np.where(
        (games['FGA_away'] - games['FG3A_away']) > 0,
        (games['FGM_away'] - games['FG3M_away']) / (games['FGA_away'] - games['FG3A_away']),
        np.nan
    )

    df['home_3p_pct'] = games['FG3_PCT_home']
    df['away_3p_pct'] = games['FG3_PCT_away']
    df['home_oreb'] = games['OREB_home']
    df['away_oreb'] = games['OREB_away']

    df['home_tov'] = games['TOV_home']
    df['away_tov'] = games['TOV_away']
    df['home_fta'] = games['FTA_home']
    df['away_fta'] = games['FTA_away']

    df['home_fg2m'] = (games['FGA_home'] * games['FG_PCT_home']) - (games['FG3A_home'] * games['FG3_PCT_home'])
    df['away_fg2m'] = (games['FGA_away'] * games['FG_PCT_away']) - (games['FG3A_away'] * games['FG3_PCT_away'])
    df['home_fg2a'] = games['FGA_home'] - games['FG3A_home']
    df['away_fg2a'] = games['FGA_away'] - games['FG3A_away']
    df['home_fg2_pct'] = np.where(df['home_fg2a'] > 0, df['home_fg2m'] / df['home_fg2a'], np.nan)
    df['away_fg2_pct'] = np.where(df['away_fg2a'] > 0, df['away_fg2m'] / df['away_fg2a'], np.nan)

    df['home_pace'] = games['FGA_home'] - games['OREB_home'] + games['TOV_home'] + 0.44 * games['FTA_home']
    df['away_pace'] = games['FGA_away'] - games['OREB_away'] + games['TOV_away'] + 0.44 * games['FTA_away']

    df['actual_total'] = games['PTS_home'] + games['PTS_away']

    rng = np.random.default_rng(42)
    df['sportsbook_line'] = np.round((df['actual_total'] + rng.normal(0, 5, len(df))) * 2) / 2
    df['went_over'] = (df['actual_total'] > df['sportsbook_line']).astype(int)
    df['odds_over'] = 1.909
    
    fg3m = games['FG3A_home'] * games['FG3_PCT_home']#t["FG3M"]
    fga = games['FG3A_home'] +df['home_fg2a']#t["FGA"]
    fgm = games['FGA_home'] * games['FG_PCT_home']#t["FG2M"]+ t["FG3M"]
    fg3maway = games['FG3A_away'] * games['FG3_PCT_away']#t["FG3M"]
    fgaaway = games['FG3A_away'] +df['away_fg2a']#t["FGA"]
    fgmaway = games['FGA_away'] * games['FG_PCT_away']#t["FG2M"]+ t["FG3M"]
    off_reb = games['OREB_home']#t["OREB"]
    opp_def_reb = games['DREB_away']#o["DREB"]
    off_reb_home = games["OREB_home"]
    def_reb_home = games["DREB_home"]
    off_reb_away = games["OREB_away"]
    def_reb_away = games["DREB_away"]
    print( games["FGA_home"] - games["OREB_home"] + games["TOV_home"] + 0.44 * games["FTA_home"])
    return pd.DataFrame({    
        "home_pace": games["FGA_home"] - games["OREB_home"] + games["TOV_home"] + 0.44 * games["FTA_home"],
        "away_pace": games["FGA_away"] - games["OREB_away"] + games["TOV_away"] + 0.44 * games["FTA_away"],
        "home_efg_pct": ((fgm + 0.5 * fg3m) / fga),
        "away_efg_pct": ((fgmaway + 0.5 * fg3maway) / fgaaway),
        "home_reb_pct": ((off_reb_home+def_reb_home) / (off_reb_home + def_reb_home+off_reb_away + def_reb_away)),
        "away_reb_pct": ((off_reb_away+def_reb_away) / (off_reb_home + def_reb_home+off_reb_away + def_reb_away)),
        "sportsbook_line": df['sportsbook_line'],
        "actual_total": df['actual_total'],
        "went_over": df['went_over'],
        "odds_over": df['odds_over']
    })
# "home_off_reb_pct": (off_reb / (off_reb + opp_def_reb)),
      #  "away_def_reb_pct": (games["DREB_away"] / (games['DREB_away'] + games['OREB_home'])),
      #  "actual_total": df['actual_total'],
      #  "went_over": df['went_over'],
    #return df.dropna().reset_index(drop=True)

# ---------------------------------------------------------------------
# 2. MACHINE LEARNING MODELS
# ---------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """Define the features the model is allowed to see before the game."""
    exclude = ['game_id', 'actual_total', 'went_over', 'odds_over']
    feature_cols = [c for c in df.columns if c not in exclude]
    return df[feature_cols].copy(), feature_cols

def train_nba_models(df: pd.DataFrame) -> Dict:
    """Train Regression (Total Points) and Classification (Over/Under) Models."""
    X, feature_cols = prepare_features(df)
    
    y_reg = df['actual_total']
    y_clf = df['went_over']
    
    X_tr, X_te, y_reg_tr, y_reg_te, y_clf_tr, y_clf_te = train_test_split(
        X, y_reg, y_clf, test_size=0.25, random_state=42
    )
    
    # Regression Model (Predicting exact points, akin to predicting Cluster Fidelity)
    reg_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    reg_model.fit(X_tr, y_reg_tr)
    pred_total = reg_model.predict(X_te)
    
    # Classification Model (Predicting Over/Under, akin to identifying "Good" states)
    clf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    clf_model.fit(X_tr, y_clf_tr)
    pred_over = clf_model.predict(X_te)
    pred_over_proba = clf_model.predict_proba(X_te)[:, 1] # Probability of going OVER
    
    # Metrics
    reg_metrics = {
        'MAE': float(mean_absolute_error(y_reg_te, pred_total)),
        'R2': float(r2_score(y_reg_te, pred_total))
    }
    
    clf_metrics = {
        'Accuracy': float(accuracy_score(y_clf_te, pred_over)),
        'ROC_AUC': float(roc_auc_score(y_clf_te, pred_over_proba))
    }
    
    return {
        'reg_model': reg_model,
        'clf_model': clf_model,
        'X_te': X_te,
        'y_reg_te': y_reg_te,
        'y_clf_te': y_clf_te,   
        'pred_total': pred_total,
        'pred_over_proba': pred_over_proba,
        'reg_metrics': reg_metrics,
        'clf_metrics': clf_metrics
    }

# ---------------------------------------------------------------------
# 3. KELLY CRITERION IMPLEMENTATION
# ---------------------------------------------------------------------

def calculate_kelly_criterion(prob_win: float, decimal_odds: float, fraction: float = 0.5) -> float:
    """
    Calculate the Kelly Criterion to determine the optimal bet sizing.
    Formula: f* = (p * (b + 1) - 1) / b
    Where 'b' is the net fractional odds (decimal_odds - 1)
    'p' is the probability of winning.
    
    Parameters:
    - fraction: Use "Fractional Kelly" (e.g., 0.5 for Half-Kelly) to reduce volatility.
    
    Returns % of bankroll to wager. If negative, do not bet.
    """
    b = decimal_odds - 1.0
    p = prob_win
    q = 1.0 - p
    
    f_star = (b * p - q) / b
    
    # If edge is negative, return 0 (no bet)
    if f_star <= 0:
        return 0.0
    
    return f_star * fraction

def apply_betting_strategy(models_dict: Dict, original_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the Kelly Criterion to our test set predictions."""
    X_te = models_dict['X_te']
    results_df = X_te.copy()
    
    results_df['actual_total'] = models_dict['y_reg_te']
    results_df['actual_over'] = models_dict['y_clf_te']
    results_df['sportsbook_line'] = original_df.loc[X_te.index, 'sportsbook_line']
    results_df['odds_over'] = original_df.loc[X_te.index, 'odds_over']
    
    results_df['predicted_total'] = models_dict['pred_total']
    results_df['prob_over'] = models_dict['pred_over_proba']
    
    # Calculate Kelly Criterion for the OVER bet
    # If the model predicts a high probability of going over, we calculate the bet size.
    results_df['kelly_stake_pct'] = results_df.apply(
        lambda row: calculate_kelly_criterion(row['prob_over'], row['odds_over'], fraction=0.25), axis=1
    )
    
    return results_df

def predict_specific_matchup(
    team_a_stats: dict, 
    team_b_stats: dict, 
    situational_context: dict,
    sportsbook_line: float,
    sportsbook_odds_decimal: float,
    reg_model, 
    clf_model):
    """
    Predicts the exact Over/Under edge for a specific matchup tonight.
    """
    
    # 1. Calculate Interaction Features
    #expected_pace = (team_a_stats['home_pace'] + team_b_stats['away_pace']) / 2
    
   # team_a_efg_edge = team_a_stats['off_efg_pct'] - team_b_stats['def_efg_pct']
   # team_b_efg_edge = team_b_stats['off_efg_pct'] - team_a_stats['def_efg_pct']
    
   # team_a_reb_edge = team_a_stats['off_reb_pct'] - team_b_stats['def_reb_pct']
   # team_b_reb_edge = team_b_stats['off_reb_pct'] - team_a_stats['def_reb_pct']
    
    # 2. Build the Feature Vector for the ML Model
    matchup_features = pd.DataFrame([{
        'home_pace': team_a_stats['home_pace'],
        'away_pace': team_b_stats['away_pace'],
        'home_efg_pct': team_a_stats['home_efg_pct'],
        'away_efg_pct': team_b_stats['away_efg_pct'],
        'home_reb_pct': team_a_stats['home_reb_pct'],
        'away_reb_pct': team_b_stats['away_reb_pct'],
        'sportsbook_line': sportsbook_line}])
    #pd.DataFrame([{
    #    'expected_pace': expected_pace,
    #    'team_a_efg_edge': team_a_efg_edge,
    #    'team_b_efg_edge': team_b_efg_edge,
    #    'team_a_reb_edge': team_a_reb_edge,
    #    'team_b_reb_edge': team_b_reb_edge,
    #    'team_a_rest_days': situational_context['team_a_rest'],
    #    'team_b_rest_days': situational_context['team_b_rest'],
    #    'is_b2b_team_a': situational_context['is_b2b_a'],
    #    'is_b2b_team_b': situational_context['is_b2b_b'],
    #    'sportsbook_line': sportsbook_line
    #}])
    print(matchup_features)
    # 3. Model Inference
    # Predict the exact total points (Regression)
    predicted_points = reg_model.predict(matchup_features)[0]
    
    # Predict the probability of the OVER hitting (Classification)
    prob_over = clf_model.predict_proba(matchup_features)[0][1]
    
    # 4. Apply Kelly Criterion
    # f* = (bp - q) / b 
    # b = decimal odds - 1
    b = sportsbook_odds_decimal - 1
    q = 1 - prob_over
    kelly_fraction = (b * prob_over - q) / b
    
    # Use Quarter-Kelly to manage bankroll variance
    recommended_bet_size = max(0, kelly_fraction * 0.25)
    
    print(f"--- MATCHUP PREDICTION ---")
    print(f"Sportsbook Line: {sportsbook_line}")
    print(f"Model Projected Points: {predicted_points:.1f}")
    print(f"Probability of OVER: {prob_over * 100:.1f}%")
    
    if recommended_bet_size > 0:
        print(f"✅ EDGE FOUND: Bet {recommended_bet_size * 100:.2f}% of your bankroll on the OVER.")
    else:
        # Check if there is an edge on the UNDER
        prob_under = 1 - prob_over
        kelly_under = (b * prob_under - prob_over) / b
        recommended_under_bet = max(0, kelly_under * 0.25)
        
        if recommended_under_bet > 0:
            print(f"✅ EDGE FOUND: Bet {recommended_under_bet * 100:.2f}% of your bankroll on the UNDER.")
        else:
            print(f"❌ NO EDGE: The sportsbook line is too sharp. Pass on this game.")
            
    return predicted_points, prob_over, recommended_bet_size
# ---------------------------------------------------------------------
# 4. EXECUTION
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("--- Step 1: Generating NBA Dataset ---")
    nba_df = generate_nba_dataset(seasons=['2023-24', '2022-23'])
    print(f"Dataset Shape: {nba_df.shape}")
    
    print("\n--- Step 2: Training Models ---")
    M = train_nba_models(nba_df)
    
    print("\nRegression Metrics (Predicted Points vs Actual Points):")
    print(M['reg_metrics'])
    
    print("\nClassification Metrics (Predicted Over/Under vs Actual):")
    print(M['clf_metrics'])
    
    print("\n--- Step 3: Applying Kelly Criterion Betting Strategy ---")
    betting_results = apply_betting_strategy(M, nba_df)
    
    # Show the top 5 games where the model suggests the largest bet
    top_bets = betting_results.sort_values(by='kelly_stake_pct', ascending=False).head(5)
    print("\nTop 5 Recommended Bets (OVER):")
    cols_to_show = ['sportsbook_line', 'predicted_total', 'actual_total', 'prob_over', 'actual_over', 'kelly_stake_pct']
    print(top_bets[cols_to_show])
    
    nba_df.to_csv("nba_dataset.csv", index=False)
    betting_results.to_csv("nba_betting_results.csv", index=False)
    print("\nData saved to CSVs for Jupyter Notebook analysis.")
