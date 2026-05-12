# NBA Player Performance & Trade Value Predictor
**DATA 300 · Spring 2026 · Dickinson College**  
**Team: Anh Vu & Drew Norton**

---

## Project Description

NBA front offices spend hundreds of millions of dollars on player contracts, yet player salaries frequently don't reflect actual on-court production. This project builds a two-phase machine learning pipeline to answer one core question:

> **Which NBA players are undervalued or overvalued relative to their contracts?**

We scrape 10 years of NBA data from Basketball Reference (2015–2024), engineer predictive features, train regression models to forecast next-season performance, cluster players into archetypes, and combine everything into a composite **Trade Value Score** that ranks every player by salary efficiency.

---

## Repository Structure

```
├── 01_data_collection.ipynb       # Scrape BBRef: per-game, advanced, salary data
├── 02_feature_engineering.ipynb   # Lag features, rolling averages, age curves
├── 03_regression.ipynb            # KNN, Linear, Random Forest, Gradient Boosting
├── 04_clustering.ipynb            # K-Means archetypes + PCA visualization
├── 05_trade_value.ipynb           # Trade Value Score + leaderboard
├── figures/                       # All output plots (PNG)
├── data/
│   ├── raw/
│   │   ├── per_game/              # One CSV per season
│   │   ├── advanced/              # One CSV per season
│   │   └── salaries/              # One CSV per season + Kaggle backup
│   └── processed/                 # Intermediate and final CSVs
├── models/                        # Saved .pkl model files (gitignored)
├── requirements.txt
└── .gitignore
```

---

## How to Run

Run notebooks **in order** — each one saves outputs that the next one loads.

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run notebooks in sequence
```
01_data_collection.ipynb      → produces data/processed/players_combined.csv
02_feature_engineering.ipynb  → produces data/processed/players_features.csv
03_regression.ipynb           → produces data/processed/player_predictions.csv
04_clustering.ipynb           → produces data/processed/players_clustered.csv
05_trade_value.ipynb          → produces data/processed/trade_value_leaderboard.csv
```

> **Note:** Notebook 01 takes ~45 minutes due to rate limiting on Basketball Reference (4 second delay between requests). After the first run, all raw CSVs are cached and subsequent runs are instant.


---

## Dataset

**Source:** [Basketball Reference](https://www.basketball-reference.com) · Seasons 2015–2024

| Table | Contents | Rows |
|---|---|---|
| Per-game stats | PTS, REB, AST, STL, BLK, FG%, 3P%, FT% | ~5,000 player-seasons |
| Advanced stats | PER, BPM, VORP, TS%, Win Shares, usage rates | ~5,000 player-seasons |
| Salary data | Contract value by season | ~4,500 player-seasons |
| **Merged output** | All of the above | **~5,000 rows, 35+ columns** |

**Cleaning decisions:**
- Players with fewer than 20 games excluded (noise reduction)
- Traded players: individual team stints kept, season-total rows dropped
- Missing salaries imputed with league minimum
- Salary normalized as % of that season's cap (accounts for cap growth 2015–2024)

**Salary cap backup:** If BBRef salary scraping fails, download `NBA Player Salaries (1990-2023)` from Kaggle and place at `data/raw/salaries/kaggle_salaries.csv`

---

## Methods

### Phase 1 — Feature Engineering (Notebook 02)

| Feature Group | Examples |
|---|---|
| Lag features | `PER_lag1`, `BPM_lag1`, `VORP_lag1` — previous season values |
| Rolling averages | `PER_roll3_mean` — 3-season rolling average |
| Year-over-year delta | `PER_delta1` — improvement or decline signal |
| Age curve | `age_sq`, `years_from_peak` — polynomial age modeling |
| Salary efficiency | `salary_pct_cap`, `dollars_per_PER` |

**Prediction targets:** `PER_next`, `PTS_next`, `WS_next` (next season values via `shift(-1)`)

**Train/test split:** Temporal — train on 2016–2022, test on 2023–2024 (no data leakage)

### Phase 2 — Regression Models (Notebook 03)

Four models trained and compared per target via 5-fold cross-validation:

| Model | Role |
|---|---|
| KNN Regressor | Baseline |
| Linear Regression / Ridge | Interpretable baseline |
| Random Forest | Ensemble, handles non-linearity |
| Gradient Boosting | Primary — best expected RMSE |

Best model per target selected by CV RMSE and evaluated on the held-out 2023–2024 test set.

### Phase 3 — Clustering (Notebook 04)

K-Means clustering on 9 efficiency/rate statistics to discover player archetypes:

| Archetype | Example Players | Defining Stats |
|---|---|---|
| Elite Scorer | Luka, Trae, SGA | High PER, BPM, VORP |
| 3-and-D Wing | Mikal Bridges, OG | High TS%, stl_pct |
| Defensive Anchor | Gobert | High blk_pct, reb_pct |
| Playmaking Guard | CP3, Tyus Jones | High ast_pct |
| Stretch Big | KAT, Porzingis | High TS%, reb_pct |
| Utility Bench | Role players | Below-average across all |

Optimal k selected via elbow method + silhouette scores. Validated with Hierarchical Clustering (Adjusted Rand Index).

### Phase 4 — Trade Value Score (Notebook 05)

```
Trade Value = 0.45 × Projected Output Score
            + 0.35 × Salary Efficiency
            + 0.15 × Age Curve Factor
            + 0.05 × Archetype Bonus
```

Rescaled to **[-100, 100]** where positive = undervalued, negative = overvalued.

**Age curve factor:** Players aged ≤22 get a 1.15× premium (upside + cheap contract); players 34+ get a 0.60× discount (decline risk).

**Archetype bonus:** Defensive Anchors (+0.10) and 3-and-D Wings (+0.08) receive bonuses because defensive contributions are systematically undervalued in salary negotiations.

---

## Evaluation

| Dimension | Method |
|---|---|
| Regression accuracy | 5-fold CV RMSE & MAE per target; final test set evaluation on 2023–2024 |
| Clustering quality | Silhouette score, WCSS elbow method, Hierarchical ARI agreement |
| Trade value validity | Correlation with VORP/BPM; rookie contract players in top 50 check; aging stars in bottom 50 check |

---

## Figures

All plots saved to `figures/` during notebook execution:

| Figure | Notebook | Description |
|---|---|---|
| `age_curve.png` | 02 | Average PER by age — confirms non-linear peak |
| `correlation_heatmap.png` | 02 | Feature–target correlation matrix |
| `model_comparison_rmse.png` | 03 | CV RMSE across all models and targets |
| `predicted_vs_actual.png` | 03 | Scatter of predictions vs holdout truth |
| `feature_importance.png` | 03 | Top 15 features per target |
| `learning_curve.png` | 03 | Bias-variance diagnosis |
| `clustering_k_selection.png` | 04 | Elbow + silhouette plots |
| `pca_cluster_biplot.png` | 04 | 2D archetype visualization |
| `archetype_radar.png` | 04 | Radar chart of archetype stat profiles |
| `trade_value_leaderboard.png` | 05 | Top 15 undervalued / bottom 15 overvalued |
| `salary_vs_output.png` | 05 | Core insight: salary vs projected PER |
| `age_vs_trade_value.png` | 05 | Where age creates biggest salary mismatches |
| `tv_vs_vorp_validation.png` | 05 | Sanity check vs established metric |

---

## Expected Findings

**Undervalued players:**
- Young players on rookie contracts performing above their salary tier
- 3-and-D specialists whose defensive metrics aren't reflected in box scores
- Players whose advanced metrics (BPM, VORP) outperform their visibility

**Overvalued players:**
- Aging stars on max contracts past their statistical peak
- High-scorer, low-efficiency players paid for counting stats alone
- Injury-prone veterans on long-term guaranteed deals

---

## Requirements

```
requests
beautifulsoup4
pandas
numpy
scikit-learn
matplotlib
seaborn
tqdm
```

Install with: `pip install -r requirements.txt`

---

## Course Topics Covered

- Supervised learning: regression (Linear, Ridge, Random Forest, Gradient Boosting)
- Unsupervised learning: K-Means and Hierarchical clustering
- Model evaluation: cross-validation, RMSE, MAE, R², silhouette score
- Feature engineering: lag features, rolling averages, polynomial terms
- Dimensionality reduction: PCA for visualization
- Web scraping: BeautifulSoup, rate limiting, data cleaning

---

*Data source: Basketball Reference (2015–2024). Project completed for DATA 300, Spring 2026, Dickinson College.*
