# Pace and Space

### How winning basketball changed -- and when

In 2009, 17 of 30 NBA teams played slow, defensive, grind-it-out basketball. By 2017, that number was zero. Something happened. This project figures out exactly what, when, and why -- and whether the teams that adopted it first actually won more.

Built on Databricks with PySpark and Delta Lake. 80 years of NBA data. Three questions, three answers.

---

## The Questions

**Did pace and three-point shooting actually change the game, or does it just feel that way?**
The Pace Revolution -- quantify exactly when tempo and 3pt rate shifted, year by year, team by team, from 1985 to 2022.

**Can you fingerprint how a team plays?**
Style Fingerprints -- cluster every team-season into a playing style using 10 measurable dimensions. No era labels. See if the model rediscovers history on its own.

**What actually predicts a championship run?**
Championship DNA -- train a model on regular season stats to predict conference finals appearances. Use SHAP to find which metrics actually matter vs which ones just feel important.

---

## What We Found

### The Pace Revolution
Three-point attempt rate grew **50% in nine years** -- from 0.26 in 2013 to 0.40 in 2021. The inflection point is unmistakable: the 2016 Warriors 73-win season. Before 2016, the trend was gradual. After 2016, the entire league moved at once.

The 1994 spike (when the NBA shortened the 3pt line) and the 1997 drop (when they moved it back) are both visible in the data without any labels -- the pipeline just shows you what happened.

<img width="1287" height="617" alt="image" src="https://github.com/user-attachments/assets/395a915c-7a0c-4c57-87a1-f6ad71cf8302" />


### Style Fingerprints
KMeans clustering on 10 style dimensions with no era labels given. The model found six clusters and labeled them by their stat profiles:

| Cluster | Style | Avg Win % |
|---|---|---|
| Slow / Defensive / Grind | Low pace, paint dominant, isolation | 0.483 |
| Paint Dominant / Isolation | High FTA, low 3pt, physical | 0.450 |
| Balanced / Halfcourt | Middle of everything | 0.547 |
| Fast 3pt / Run and Gun | High pace + high 3pt rate | 0.453 |
| Warriors Style / Ball Movement | High pace + high assists + high 3pt | 0.543 |
| Analytics Optimized / Efficient | Selective 3pt, low turnovers, efficient | 0.552 |

What the model found without being told:

- 2009: 17 teams in "Slow/Defensive/Grind". Zero in Warriors Style or Analytics Optimized.
- 2017: "Slow/Defensive/Grind" hits zero and never comes back.
- 2018: 15 of 30 teams classified as "Warriors Style" -- peak copycat era.
- 2020 onward: league settles into two mature modern styles.

<img width="1297" height="602" alt="image" src="https://github.com/user-attachments/assets/1b8cf313-6c43-4bbc-88e3-b1a24aeba542" />


### Championship DNA
XGBoost predicting conference finals appearance from regular season stats. Trained on 2009-2018, tested on 2019-2022. ROC-AUC: **0.791**.

SHAP attribution on the test set:
<img width="962" height="712" alt="image" src="https://github.com/user-attachments/assets/c9a1cde6-8320-4b41-b6a5-59bd2567ea45" />

- **Turnover rate** -- the strongest style predictor. Low turnovers push strongly toward championship contention. Playoff defense is tighter; teams that turn it over in the regular season get exposed.
- **Offensive rebounding** -- second chance points matter more in playoff series where every possession counts.
- **Three-point rate** -- surprisingly weak signal. Shoots a lot of 3s does not mean wins a championship.
- **Pace** -- near zero effect. Tempo alone predicts almost nothing about playoff success.
- **Win percentage** -- necessary but not sufficient. The model finds teams that overperform their style profile in the regular season and get exposed later.

The conventional wisdom says "shoot more 3s to win." The data says "take care of the ball."

---

## Architecture

```mermaid
flowchart TD
    A[("nba.sqlite\n14M rows\n65K games")] --> B

    subgraph ingest ["Ingestion"]
        B["Python + sqlite3\nChunked reads for play-by-play\n500K rows per chunk"]
    end

    B --> C

    subgraph bronze ["Bronze Layer  --  workspace.bronze_nba"]
        C["15 Delta Tables\nRaw strings, immutable\n_ingested_at + _source metadata"]
    end

    C --> D

    subgraph silver ["Silver Layer  --  workspace.silver_nba"]
        D["game: 128K rows\nWide to long unpivot\nOne row per team per game"]
        E["other_stats: 56K rows\ntry_cast for NaN handling\nZ-ordered on team_id"]
    end

    D --> F
    E --> F

    subgraph gold ["Gold Layer  --  workspace.gold_nba"]
        F["pace_revolution\n1,519 team-seasons\nDean Oliver pace formula\nZ-ordered season_year + team_id"]
        G["style_fingerprints\n390 team-seasons\n10-dim style vectors\nKMeans cluster labels"]
    end

    F --> H
    G --> H
    G --> I

    subgraph ml ["ML + Viz"]
        H["Plotly\nPace + 3pt timeline\nStyle distribution chart"]
        I["XGBoost + SHAP\nTemporal split 2009-2018 / 2019-2022\nscale_pos_weight = 6.5\nROC-AUC 0.791"]
    end

    I --> J["MLflow\n/Shared/championship_dna\nParams + metrics + SHAP artifact"]
```

---

## Tech Stack

> **Platform**
> Databricks Free Edition -- serverless notebooks, Unity Catalog storage, native MLflow

> **Data Processing**
> PySpark 4.1 -- medallion architecture, wide-to-long transforms, window functions, Z-ordering

> **Storage**
> Delta Lake on Unity Catalog -- ACID transactions, schema enforcement, time travel

> **Machine Learning**
> XGBoost + SHAP + scikit-learn -- classifier, explainability, temporal validation

> **Clustering**
> sklearn KMeans -- 10-dimension style vectors, StandardScaler, silhouette scoring

> **Visualization**
> Plotly -- interactive dual-axis timeline, stacked bar evolution chart

> **Experiment Tracking**
> MLflow (native Databricks) -- params, metrics, SHAP artifact, model registry

> **Package Management**
> uv -- fast, reproducible Python environments

---

## Data Source

Kaggle -- [NBA Database by wyattowalsh](https://www.kaggle.com/datasets/wyattowalsh/basketball)

- 65,000+ games since the 1946-47 season
- 13.6M rows of play-by-play
- Complete box scores, team stats, player info
- Single SQLite file, 2.35GB

---

## How to Run

**Prerequisites:** Databricks Free Edition account, Kaggle account to download the dataset.

1. Download `nba.sqlite` from the Kaggle link above
2. In Databricks: Catalog -> `workspace` -> `bronze_nba` -> Create volume `raw_files` -> Upload file (fits the 5GB UI limit)
3. Import notebooks from the `notebooks/` folder into your Databricks workspace
4. Run in order -- each notebook reads from the previous layer's Delta tables

| Notebook | What it does |
|---|---|
| `00_environment_test` | Verifies PySpark, Delta Lake, Unity Catalog write access |
| `01_bronze_ingestion` | Reads SQLite, writes 15 Delta tables to `bronze_nba` |
| `02_silver_game` | Unpivots game table wide-to-long, casts types, filters season types |
| `03_gold` | Pace formula, style fingerprints, KMeans clustering, XGBoost, MLflow |
| `04_pace_viz` | Plotly dual-axis timeline, style distribution chart |

MLflow results visible at `/Shared/championship_dna` in your workspace after running `03_gold`.

---

