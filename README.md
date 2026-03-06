# Bundesliga Player Average Age Metrics

This project builds two raw Bundesliga datasets and several derived analysis
CSV files for the report pages in `docs/`.

## Python Files

- `main.py`
  - Runs the full workflow.
  - Loads cached raw outputs or rebuilds them.
  - Writes the raw CSV files, derived analysis CSV files, and terminal summary.
- `pipeline_config.py`
  - Parses command line arguments.
  - Builds the small `Config` object used by the pipeline.
  - Prepares the output directory and shared environment settings.
- `pipeline_utils.py`
  - Stores shared constants.
  - Provides small helper functions for parsing values, fixing names,
    formatting progress, rounding output, and loading `soccerdata`.
- `espn_data_download_pipeline.py`
  - Builds the ESPN dataset for RQ9.
  - Loads match sheets, player profile ages, and cached match summary files.
  - Returns one row per match, team, and player.
- `whoscored_data_download_pipeline.py`
  - Builds the WhoScored dataset for RQ4.
  - Loads the season schedule and cached event files.
  - Returns one row per match and player with rating information.
- `analysis_of_rq4_rq9_questions.py`
  - Creates the derived analysis tables used by the website.
  - Aggregates the RQ4 and RQ9 metrics.
  - Builds the short terminal answers for both research questions.

## Run

```powershell
python main.py
```

Optional refresh:

```powershell
python main.py --refresh
```

## Output Files

The pipeline writes exactly two raw source datasets to `data/outputs/`:

- `espn_player_match_data_for_rq9.csv`
  - one row per `match x team x player`
  - includes the raw fields needed to answer RQ9: `player`, `team`, `age`,
    `player_goals`, `player_shots`, `team_goals`, `team_shots`
- `whoscored_player_match_data_for_rq4.csv`
  - one row per `match x player`
  - includes the raw fields needed to answer RQ4: `player`, `team`,
    `home_away`, `overall_rating`, `is_starting_xi`,
    `is_man_of_the_match`

The analysis step also writes derived CSV files under `docs/data/` for the
website charts and text.

## Notes

- The raw CSVs are analysis inputs, not final written answers.
- Fresh source data still uses `soccerdata` cache files under
  `.soccerdata_cache/`.
