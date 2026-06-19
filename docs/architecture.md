# Architecture Decisions

## Storage: Delta Lake over plain Parquet

Delta Lake wraps Parquet with a transaction log. Every write is atomic -- it either commits fully or does not appear at all. For a 44-season historical backfill, this means a failed API call or cluster crash never produces a partially written file. Seasons that completed successfully remain intact and readable. Without Delta, a mid-write failure produces silent corruption -- wrong numbers with no error.

Additional reasons per layer:
- Bronze: atomicity during backfill
- Silver: schema enforcement rejects rows that do not match expected types
- Gold: time travel allows querying table state before schema changes

## Processing: PySpark throughout

The cluster is already running in Databricks serverless. Switching between Pandas and PySpark mid-pipeline adds conversion overhead and two mental models in the same codebase. PySpark is used throughout for consistency. The 13.6M row play-by-play table genuinely justifies the distributed compute architecture.

Exception: KMeans clustering uses sklearn because the dataset (390 rows) fits in memory and PySpark ML hits serverless cache limits iterating over multiple K values.

## Optimization: Z-ordering over partitioning

Partitioning on season_year creates 44+ folders. On Databricks serverless, many small partitions degrade I/O performance. Z-ordering co-locates rows with similar season_year and team_id values within files using a space-filling curve, so queries filtering on either dimension read fewer files without the small file problem.

Applied to: silver_nba.game, silver_nba.other_stats, gold_nba.pace_revolution.

## Play-by-play: ingested but not used for analysis

All three analytical pillars (pace revolution, style fingerprints, championship DNA) are answerable from aggregated box score stats. The nba_api pre-computes pace, 3pt rate, and net rating at the team-game level. Play-by-play (13.6M rows) is ingested into Bronze to demonstrate PySpark at scale and as a foundation for future extensions (possession-level analysis, play type frequency).

## ML: temporal train/test split

Training on 2009-2018, testing on 2019-2022. Stratified random split would leak future team style patterns into past predictions -- a team's 2022 style fingerprint would inform predictions about 2015 playoff outcomes. Temporal split matches real-world deployment: train on history, predict the current season.

## ML: scale_pos_weight over SMOTE

52 conference finalists out of 390 total team-seasons (13% positive rate). scale_pos_weight = 338/52 = 6.5 tells XGBoost to weight positive examples 6.5x in gradient updates. SMOTE would generate synthetic positive examples by interpolating between 52 real ones -- risky on a small sample where the interpolated points may not represent real team profiles.

## Label: conference finals appearance

"Won championship" gives 1 positive per 30 teams per season -- too sparse for a reliable classifier. "Made playoffs" gives 16 of 30 teams per season -- too easy, model learns nothing interesting. "Reached conference finals" gives 4 positives per season, is a meaningful basketball threshold (top 4 teams in the league), and yields 52 positive examples across 13 seasons -- learnable with scale_pos_weight.
