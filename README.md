⚽ Premier League — Over/Under 2.5 Goals Classifier

    A machine learning pipeline that predicts whether a Premier League match will produce
    Over or Under 2.5 goals, and also predicts the match outcome (Win / Draw / Loss).
    Built entirely on free, legal APIs — no scraping, no Cloudflare, no IP bans.

📋 Table of Contents

    Overview

    Project Structure

    Data Sources

    Installation

    Quick Start

    Pipeline

    Features

    Models

    Results

    Predict a Match

    Extending to Champions League

    Important Caveats

    License

Overview

This project builds a complete end-to-end classification system for football betting analysis.
Given two teams and a matchday, it outputs:

    P(Over 2.5 goals) and P(Under 2.5 goals)

    P(Home Win), P(Draw), P(Away Win)

    A confidence score and Kelly Criterion hint

Accuracy ceiling: ~64–67% (the theoretical maximum for O/U 2.5 in football).
Any model claiming higher is overfitting.
