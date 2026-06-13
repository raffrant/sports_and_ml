Machine Learning in Sports

A collection of sports analytics and prediction projects built with machine learning, statistical modeling, and historical match data. The repository focuses on three practical directions: basketball analysis in the EuroLeague, NBA over/under prediction, and football over/under 2.5 goals modeling.
Overview

This section of the repository explores how machine learning can be used to understand performance, identify patterns, and generate match-level predictions in sports.

The projects are organized around three use cases:

    EuroLeague basketball analysis — player and match analysis from the most recent EuroLeague season. (basketball folder)

    NBA totals prediction — over/under modeling using current-season and historical NBA data. (nba folder)

    Football goals prediction — over/under 2.5 goals forecasting using team history and current-season context. (premierleague and worldcup folder)

Project Structure

text
sports/
├── basketball/
├── nba/
├── worldcup/
└── premierleague/

Projects
1. EuroLeague Basketball Analysis

Folder: basketball/

This project analyzes basketball players and matches from the most recent EuroLeague season. The focus is on extracting useful patterns from team and player performance, comparing metrics across games, and understanding the factors that shape match outcomes. Specifically, we focused on Kendrick Nunn (Panathinaiko's player) to find each true shooting.

Typical tasks in this folder may include:

    player-level statistical analysis,

    team comparisons,

    match trend exploration,

    performance breakdowns by game or season segment.

2. NBA Over/Under Prediction

Folder: nba/

This project analyzes and predicts whether an NBA game will finish over or under the posted total. The models use current-season information together with previous seasons to build features that reflect team style, pace, shooting efficiency, rebounding, and scoring environment.

The NBA workflow typically includes:

    collecting historical game and team data,

    engineering pre-game features,

    training regression and classification models,

    estimating the probability of an over/under outcome,

    give a prediction O/U along with a percentage of your bankroll you can bet. 
    
3. Football Over/Under 2.5 Goals Prediction

Folder: premierleague/ and worldcup/

This project predicts whether a football match will finish over or under 2.5 goals. The approach combines historical team behavior with current-season form, allowing the model to capture both long-term identity and short-term momentum.

The football workflow typically includes:

    analyzing team scoring and conceding history (multiple seasons analysis using football api)

    incorporating recent form and season context,

    modeling total-goal patterns,

    predicting over/under 2.5 outcomes for upcoming matches.

Methods

Across the sports projects, the general pipeline includes:

    Data collection and cleaning.

    Exploratory analysis and feature engineering.

    Model training and validation.

    Match-level prediction.

    Performance evaluation and iteration.

Depending on the project, models may include regression, classification, probabilistic scoring, and other predictive ML techniques.
Why This Repository

This repository is built around a practical idea: sports data becomes more useful when it moves beyond descriptive statistics and into decision-oriented modeling. Instead of only summarizing what happened, these projects aim to estimate what is likely to happen next.

That makes the repository useful for:

    sports analytics practice,

    predictive modeling experiments,

    betting-oriented research,

    feature engineering on real match data,

    comparing different ML approaches across sports.

Getting Started

Clone the repository:

bash
git clone https://github.com/raffrant/sports_and_ml.git
cd sports_and_ml

Open the folder you want to work on:

bash
cd basketball

or

bash
cd nba

or

bash
cd premierleague

or

bash
cd worldcup

Common Tools

These projects commonly use:

    Python

    pandas

    numpy

    scikit-learn

    matplotlib / seaborn

    Jupyter Notebook

Install dependencies with:

bash
pip install -r requirements.txt

Future Directions

Planned improvements may include:

    richer feature sets,

    more seasons of historical data,

    model comparison dashboards,

    better calibration of probabilities,

    deployment as notebooks, scripts, or interactive apps.

Visualization
For Jupyter notebooks, you can use https://nbviewer.org/ to display any results if there is any problem with Github sharing the content.

Notes

Each folder can have its own data analysis, logic, and modeling assumptions. For that reason, the best way to use this repository is to treat each project as its own mini-pipeline inside the broader sports ML collection.
License

MIT license
