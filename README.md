# Pace and Space -- NBA Team Strategy Analytics

A production-grade data engineering and data science portfolio project built on Databricks Free Edition using PySpark 4.1 and Delta Lake. Ingests 80 years of NBA game data and answers: **how do winning teams play differently, and has the definition of winning basketball changed over time?**

## Three Analytical Pillars

### 1. The Pace Revolution
Quantifies how NBA tempo and three-point shooting evolved from 1985 to 2022 using the Dean Oliver possession formula on box score data. Shows the Warriors 73-win season in 2016 as the visible inflection point after which the entire league shifted style.

### 2. Style Fingerprints
KMeans clustering on 10-dimension style vectors (pace, three-point rate, turnover rate, assist rate, paint scoring share, fast break rate, offensive rebounding, defensive aggression, free throw rate, second chance rate) per team per season. No era labels provided -- the model rediscovered the Warriors revolution: "Slow/Defensive/Grind" went from 17 teams in 2009 to zero by 2017 and never returned.

### 3. Championship DNA
XGBoost classifier predicting conference finals appearance from regular season metrics. Temporal train/test split (2009-2018 train, 2019-2022 test). ROC-AUC 0.791. SHAP attribution found that turnover rate and offensive rebounding are stronger predictors than three-point rate or pace -- ball security matters more than spacing in playoff series.

## Architecture

```
SQLite source (nba.sqlite, 14M rows)
        |
        v
Unity Catalog Volume
(workspace.bronze_nba.raw_files)
        |
  Python ingestion
  (sqlite -> PySpark -> Delta)
        |
        v
Bronze Layer (workspace.bronze_nba)
15 tables, raw strings, immutable
        |
  PySpark cleaning
  wide -> long unpivot
  type casting, filtering
        |
        v
Silver Layer (workspace.silver_nba)
game: 128k rows (one row per team per game)
other_stats: 56k rows
Z-ordered on season_year + team_id
        |
  PySpark aggregations
  derived metrics
        |
        v
Gold Layer (workspace.gold_nba)
pace_revolution: 1,519 team-seasons
style_fingerprints: 390 team-seasons + cluster labels
        |
    sklearn KMeans          XGBoost + SHAP
    10-dim clustering       temporal split
    silhouette scoring      scale_pos_weight=6.5
        |                       |
   Style labels            ROC-AUC 0.791
   per team/season         SHAP attribution
        |
   MLflow tracking
   (experiment: /Shared/championship_dna)
        |
   Plotly visualizations
```

## Key Design Decisions

See [docs/architecture.md](docs/architecture.md) for full reasoning behind each decision.

| Decision | Choice | Why |
|---|---|---|
| Storage format | Delta Lake on all layers | ACID atomicity for backfill recovery; schema enforcement; time travel |
| Processing | PySpark throughout | Cluster already running; pipeline consistency; 13.6M row play-by-play justifies architecture |
| Optimization | Z-order on season_year + team_id | Avoids small file problem; two-dimensional co-location |
| Play-by-play | Ingested to Bronze, not used for analysis | All three pillars answerable from aggregated box scores |
| Train/test split | Temporal (2009-2018 / 2019-2022) | Prevents future data leaking into past predictions |
| Class imbalance | scale_pos_weight = 6.5 | No synthetic data generation on 52-sample positive class |
| Cluster label | Conference finals appearance | 4 positives per season, meaningful basketball threshold |

## Data Source

Kaggle: NBA Database by wyattowalsh

- 65,000+ games since 1946-47
- 13.6M rows of play-by-play
- Box scores, team stats, player info

## Tech Stack

| Component | Technology |
|---|---|
| Platform | Databricks Free Edition (serverless) |
| Processing | PySpark 4.1 |
| Storage | Delta Lake on Unity Catalog |
| ML | XGBoost + SHAP + sklearn |
| Experiment tracking | MLflow (native Databricks) |
| Visualization | Plotly |
| Package management | uv |

## Notebooks (run in order)

All notebooks run on Databricks serverless. Import from the `notebooks/` folder into your Databricks workspace.

| Notebook | Description |
|---|---|
| 00_environment_test | Verify PySpark, Delta Lake, Unity Catalog access |
| 01_bronze_ingestion | Read SQLite, write 15 Delta tables to bronze_nba |
| 02_silver_game | Unpivot game table, cast types, filter season types |
| 03_gold_pace_revolution | Dean Oliver pace formula, aggregate to team-season |
| 04_pace_revolution_viz | Plotly timeline of pace and 3pt rate 1985-2022 |
| 05_gold_style_fingerprints | 10-dim style vectors, KMeans clustering, cluster labels |
| 06_championship_dna | XGBoost classifier, SHAP attribution, MLflow logging |

## Setup

1. Download nba.sqlite from Kaggle (NBA Database by wyattowalsh)
2. Upload to `workspace.bronze_nba.raw_files` Unity Catalog volume via Databricks UI (under 5GB, fits the file uploader)
3. Run notebooks in order on Databricks Free Edition serverless compute
4. MLflow experiment results visible at `/Shared/championship_dna` in your workspace

## Findings

**Pace Revolution:** Three-point attempt rate grew 50% from 2013 to 2022. The 2016 Warriors season is the visible inflection point. By 2017, slow defensive basketball had completely disappeared from the league.

**Style Fingerprints:** Unsupervised clustering without era labels detected the Warriors revolution. In 2018, 15 of 30 teams were classified as "Warriors Style / Ball Movement" -- peak copycat era. By 2020, the league consolidated into "Analytics Optimized" and "Fast 3pt" as two mature modern styles.

**Championship DNA:** Ball security (low turnover rate) and offensive rebounding are stronger predictors of conference finals appearance than three-point rate or pace. High regular season win percentage is necessary but not sufficient -- the model identifies teams that overperform their style profile in the regular season and get exposed in playoffs.

## Resume Bullets

**Data Engineering:**
Built Pace and Space, an NBA analytics pipeline on Databricks -- PySpark 4.1 processing 14M rows through Delta Lake medallion architecture (Bronze/Silver/Gold), Z-ordered by season and team for query optimization, with MLflow experiment tracking and Unity Catalog storage

**Data Science:**
Engineered Championship DNA model predicting NBA playoff success from regular season metrics -- XGBoost classifier with temporal train/test split and scale_pos_weight for class imbalance (ROC-AUC 0.791), SHAP attribution identifying turnover rate and offensive rebounding as dominant predictors over pace and three-point rate; KMeans clustering on 10-dimension style vectors detected the Warriors revolution without era labels
