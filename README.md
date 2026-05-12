# NBA Player Performance & Trade Value Predictor
**DATA 300 · Spring 2026 · Dickinson College**  
**Team: Anh Vu & Drew Norton**

---

## Overview

NBA front offices spend hundreds of millions on player contracts, yet salaries frequently don't reflect actual on-court production. This project builds a two-phase machine learning pipeline to answer one core question:

> **Which NBA players are undervalued or overvalued relative to their contracts?**

We scrape 10 years of NBA data from Basketball Reference (2015–2024), engineer predictive features, train regression models to forecast next-season performance, cluster players into archetypes, and combine everything into a composite **Trade Value Score** that ranks every player by salary efficiency.

---

## How to Run

**Install dependencies:**
```bash
pip install requests beautifulsoup4 pandas numpy scikit-learn matplotlib seaborn tqdm
```

**Run notebooks in order — each one saves outputs the next one loads:**

```
01_data_collection.ipynb      →  data/processed/players_combined.csv
02_feature_engineering.ipynb  →  data/processed/players_features.csv
                                  data/processed/feature_config.json
03_regression.ipynb           →  data/processed/player_predictions.csv
                                  data/processed/cv_results.csv
                                  models/best_*.pkl
04_clustering.ipynb           →  data/processed/players_clustered.csv
05_trade_value.ipynb          →  data/processed/trade_value_leaderboard.csv
```

> ⚠️ Notebook 01 takes ~45 minutes due to Basketball Reference rate limiting (4 second delay between requests). After the first run, all raw CSVs are cached and re-runs are instant.

**Salary data:** BBRef salary scraping may fail. If it does, download `NBA Player Salaries` from Kaggle and place at `data/raw/salaries/kaggle_salaries.csv`. The notebook will load it automatically.

---

## Notebook Descriptions

### 01 — Data Collection
Scrapes Basketball Reference for seasons 2016–2024:
- **Per-game stats:** PTS, REB, AST, STL, BLK, TOV, FG%, 3P%, FT%
- **Advanced metrics:** PER, TS%, BPM, VORP, Win Shares, usage rates
- **Salary data:** Contract values normalized as % of salary cap

Cleaning decisions:
- Players with fewer than 20 games excluded
- Traded players: `TOT` rows kept, individual team stints dropped
- Missing salaries imputed with league minimum ($1M)
- Salary cap values hardcoded by season (2016: $70M → 2024: $136M)

Output: `data/processed/players_combined.csv` (~5,000 rows, 35+ columns)

---

### 02 — Feature Engineering
Builds all predictive signals from the combined dataset:

| Feature Group | Examples |
|---|---|
| Lag features | `PER_lag1`, `BPM_lag1` — previous season values |
| 3-year rolling mean | `PER_roll3_mean` — smoothed trend |
| 3-year rolling std | `PER_roll3_std` — consistency signal |
| Year-over-year delta | `PER_delta1` — improvement or decline |
| Age curve | `age_sq`, `years_from_peak` (relative to age 27) |
| Binary flag | `past_prime` (age > 30) |
| Salary efficiency | `salary_pct_cap`, `dollars_per_PER` |

**Train/test split:** Temporal — train on 2016–2022, test on 2023–2024. No random splitting to prevent data leakage.

**Targets:** `PER_next`, `PTS_next`, `WS_next` — next season values created via `shift(-1)` per player.

Output: `data/processed/players_features.csv`, `data/processed/feature_config.json`

---

### 03 — Regression Modeling
Trains and compares 5 models per target using 5-fold cross-validation:

| Model | Role |
|---|---|
| KNN Regressor | Baseline |
| Linear Regression | Interpretable baseline |
| Ridge Regression | Regularized baseline |
| Random Forest | Ensemble, handles non-linearity |
| Gradient Boosting | Primary — best expected RMSE |

First compares lag1-only vs 3-year rolling feature sets. Best feature set selected by Gradient Boosting RMSE, then used for all models.

Best model per target selected by CV RMSE, retrained on full training set, evaluated on held-out 2023–2024 test set.

Figures produced: `model_comparison_rmse.png`, `model_comparison_r2.png`, `predicted_vs_actual.png`, `feature_importance.png`, `learning_curve.png`

Output: `data/processed/player_predictions.csv`, `models/best_*.pkl`

---

### 04 — Player Archetype Clustering
Clusters players into archetypes using 9 efficiency/rate statistics:

```
PER, TS%, BPM, VORP, ast%, reb%, blk%, stl%, tov%
```

Rate stats used instead of volume stats to avoid separating starters from bench players by minutes alone.

**k selection:** Elbow method (WCSS) + silhouette scores tested for k=2–12. Default k=6 — adjust `K_FINAL` if your plots suggest otherwise.

**Validation:** Hierarchical clustering (Ward linkage) run in parallel. Adjusted Rand Index measures agreement — ARI > 0.6 = strong structural agreement.

**Archetypes:**

| Archetype | Example Players | Key Stats |
|---|---|---|
| Elite Scorer | Luka, Trae, SGA | High PER, BPM, VORP |
| 3-and-D Wing | Mikal Bridges, OG | High TS%, stl% |
| Defensive Anchor | Gobert | High blk%, reb% |
| Playmaking Guard | CP3, Tyus Jones | High ast% |
| Stretch Big | KAT, Porzingis | High TS%, reb% |
| Utility Bench | Role players | Below average across all |

Figures produced: `clustering_k_selection.png`, `silhouette_analysis.png`, `pca_cluster_biplot.png`, `pca_loadings.png`, `archetype_radar.png`

Output: `data/processed/players_clustered.csv`

---

### 05 — Trade Value Score & Leaderboard
Combines predictions, salary, age, and archetype into a single score:

```
Trade Value = 0.45 × Projected Output Score
            + 0.35 × Salary Efficiency Score
            + 0.15 × Age Curve Factor
            + 0.05 × Archetype Bonus
```

Rescaled to **[-100, 100]** — positive = undervalued, negative = overvalued.

**Age curve multipliers:**

| Age | Multiplier |
|---|---|
| ≤ 22 | 1.15× |
| 23–24 | 1.10× |
| 25–28 | 1.00× |
| 29–30 | 0.93× |
| 31–32 | 0.85× |
| 33–34 | 0.75× |
| 35+ | 0.60× |

**Archetype bonus:** Defensive Anchors (+0.10) and 3-and-D Wings (+0.08) receive bonuses because defensive contributions are systematically undervalued in salary negotiations. Elite Scorers receive no bonus (market correctly values stars). Utility Bench receives a small penalty (-0.05).

**Tiers:** Highly Undervalued / Undervalued / Fairly Valued / Overvalued / Highly Overvalued

**Validation checks:**
- Moderate positive correlation with VORP/BPM confirms score captures real production
- Negative correlation with salary confirms cheap contracts score higher
- Rookie/min contract players expected to dominate top 50
- Aging stars on max deals expected to dominate bottom 50

Figures produced: `trade_value_leaderboard.png`, `salary_vs_output.png`, `tv_by_archetype.png`, `age_vs_trade_value.png`, `tv_component_breakdown.png`, `tv_vs_vorp_validation.png`

Output: `data/processed/trade_value_leaderboard.csv`

---

## Repository Structure

```
├── 01_data_collection.ipynb
├── 02_feature_engineering.ipynb
├── 03_regression.ipynb
├── 04_clustering.ipynb
├── 05_trade_value.ipynb
├── data/
│   ├── raw/
│   │   ├── per_game/          ← one CSV per season from BBRef
│   │   ├── advanced/          ← one CSV per season from BBRef
│   │   └── salaries/          ← one CSV per season + kaggle backup
│   └── processed/             ← intermediate and final outputs
├── figures/                   ← all plots saved as PNG
├── models/                    ← fitted .pkl files (gitignored)
├── requirements.txt
└── .gitignore
```

---

## Dataset

**Source:** [Basketball Reference](https://www.basketball-reference.com) · Seasons 2016–2024

**Salary cap by season:**

| Season | Cap |
|---|---|
| 2016 | $70.0M |
| 2017 | $94.1M |
| 2018 | $99.1M |
| 2019 | $101.9M |
| 2020–2021 | $109.1M |
| 2022 | $112.4M |
| 2023 | $123.7M |
| 2024 | $136.0M |

---

## Course Topics Covered

- Supervised learning: KNN, Linear Regression, Ridge, Random Forest, Gradient Boosting
- Unsupervised learning: K-Means clustering, Hierarchical clustering
- Model evaluation: 5-fold CV, RMSE, MAE, R², silhouette score, Adjusted Rand Index
- Feature engineering: lag features, rolling averages, polynomial age terms
- Dimensionality reduction: PCA for cluster visualization
- Web scraping: BeautifulSoup, rate limiting, HTML table parsing

---

*Data: Basketball Reference 2016–2024 · DATA 300 Spring 2026 · Dickinson College*
