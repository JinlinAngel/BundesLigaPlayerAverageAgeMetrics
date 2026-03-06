"""Build the derived analysis tables for RQ4 and RQ9.

Input: raw ESPN and WhoScored output tables.
Output: derived CSV tables and short terminal answer strings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline_config import DEFAULT_OUTPUT_DIR
from pipeline_utils import (
    RQ4_MIN_MATCHES_FOR_LEADERBOARD,
    RQ4_MIN_MATCHES_PER_SIDE_FOR_DELTA,
    fit_correlation,
    to_bool,
)


RAW_RQ9_FILE = "espn_player_match_data_for_rq9.csv"
RAW_RQ4_FILE = "whoscored_player_match_data_for_rq4.csv"
DEFAULT_DOCS_DATA_DIR = Path(__file__).resolve().parent / "docs" / "data"
BEST_AGE_MIN_TOTAL_SHOTS = 80


def first_non_null(series: pd.Series) -> object:
    """Return the first non-null value from a Series.

    Input: pandas Series.
    Output: first non-null value or `nan`.
    """
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return valid.iloc[0]


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two Series and keep `nan` for zero denominators.

    Input: numerator Series and denominator Series.
    Output: float Series with safe division.
    """
    top = pd.to_numeric(numerator, errors="coerce")
    bottom = pd.to_numeric(denominator, errors="coerce")
    return top.divide(bottom.where(bottom != 0))


def normalize_rq9(rq9_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the raw ESPN table for downstream analysis.

    Input: raw RQ9 DataFrame.
    Output: copied DataFrame with cleaned types.
    """
    out = rq9_df.copy()
    out["season"] = out["season"].astype(str)
    out["game_id"] = out["game_id"].astype(str)
    out["player_id"] = out["player_id"].astype(str)
    for column in (
        "age",
        "player_goals",
        "player_shots",
        "team_goals",
        "team_shots",
    ):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def normalize_rq4(rq4_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the raw WhoScored table for downstream analysis.

    Input: raw RQ4 DataFrame.
    Output: copied DataFrame with cleaned types.
    """
    out = rq4_df.copy()
    out["season"] = out["season"].astype(str)
    out["game_id"] = out["game_id"].astype(str)
    out["player_id"] = out["player_id"].astype(str)
    out["overall_rating"] = pd.to_numeric(
        out["overall_rating"],
        errors="coerce",
    )
    out["is_starting_xi"] = out["is_starting_xi"].map(to_bool)
    out["is_man_of_the_match"] = out["is_man_of_the_match"].map(to_bool)
    return out


def build_season_age_summary(rq9_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one age summary row per season.

    Input: normalized RQ9 DataFrame.
    Output: DataFrame for `bundesliga_season_age_summary.csv`.
    """
    season_players = (
        rq9_df.groupby(["season", "season_label", "player_id"], dropna=False)
        .agg(age=("age", first_non_null))
        .reset_index()
    )
    return (
        season_players.groupby(["season", "season_label"], dropna=False)
        .agg(
            unique_players=("player_id", "nunique"),
            avg_age=("age", "mean"),
            min_age=("age", "min"),
            max_age=("age", "max"),
            missing_age=("age", lambda series: int(series.isna().sum())),
        )
        .reset_index()
        .sort_values(["season"], kind="stable")
    )


def build_team_age_summary(rq9_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one age summary row per team.

    Input: normalized RQ9 DataFrame.
    Output: DataFrame for `bundesliga_team_age_summary.csv`.
    """
    team_players = (
        rq9_df.groupby(
            ["season", "season_label", "team", "player_id"],
            dropna=False,
        )
        .agg(age=("age", first_non_null))
        .reset_index()
    )
    return (
        team_players.groupby(
            ["season", "season_label", "team"],
            dropna=False,
        )
        .agg(
            player_count=("player_id", "nunique"),
            avg_age=("age", "mean"),
            min_age=("age", "min"),
            max_age=("age", "max"),
            missing_age=("age", lambda series: int(series.isna().sum())),
        )
        .reset_index()
        .sort_values(
            ["season", "avg_age", "team"],
            ascending=[True, False, True],
            kind="stable",
        )
    )


def build_rq4_home_away_player_ratings(rq4_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player ratings by home and away matches.

    Input: normalized RQ4 DataFrame.
    Output: DataFrame for `rq4_home_away_player_ratings.csv`.
    """
    summary = (
        rq4_df.groupby(
            ["season", "season_label", "home_away", "player", "player_id"],
            dropna=False,
        )
        .agg(
            matches=("game_id", "nunique"),
            teams=("team", "nunique"),
            avg_overall_rating=("overall_rating", "mean"),
            median_overall_rating=("overall_rating", "median"),
            best_overall_rating=("overall_rating", "max"),
            worst_overall_rating=("overall_rating", "min"),
            starts=("is_starting_xi", "sum"),
            motm_awards=("is_man_of_the_match", "sum"),
        )
        .reset_index()
    )
    summary["eligible_for_leaderboard"] = (
        summary["matches"] >= RQ4_MIN_MATCHES_FOR_LEADERBOARD
    )
    return summary.sort_values(
        ["season", "home_away", "avg_overall_rating", "matches", "player"],
        ascending=[True, True, False, False, True],
        kind="stable",
    )


def build_rq4_player_home_away_delta(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Compare average player ratings between home and away matches.

    Input: aggregated RQ4 ratings DataFrame.
    Output: DataFrame for `rq4_player_home_away_delta.csv`.
    """
    join_keys = ["season", "season_label", "player", "player_id"]
    value_columns = [
        "matches",
        "teams",
        "avg_overall_rating",
        "median_overall_rating",
        "best_overall_rating",
        "worst_overall_rating",
        "starts",
        "motm_awards",
    ]

    home = ratings_df.loc[
        ratings_df["home_away"] == "home",
        join_keys + value_columns,
    ].copy()
    away = ratings_df.loc[
        ratings_df["home_away"] == "away",
        join_keys + value_columns,
    ].copy()

    home = home.rename(
        columns={
            "matches": "home_matches",
            "teams": "home_teams",
            "avg_overall_rating": "home_avg_overall_rating",
            "median_overall_rating": "home_median_overall_rating",
            "best_overall_rating": "home_best_overall_rating",
            "worst_overall_rating": "home_worst_overall_rating",
            "starts": "home_starts",
            "motm_awards": "home_motm_awards",
        }
    )
    away = away.rename(
        columns={
            "matches": "away_matches",
            "teams": "away_teams",
            "avg_overall_rating": "away_avg_overall_rating",
            "median_overall_rating": "away_median_overall_rating",
            "best_overall_rating": "away_best_overall_rating",
            "worst_overall_rating": "away_worst_overall_rating",
            "starts": "away_starts",
            "motm_awards": "away_motm_awards",
        }
    )

    delta = home.merge(away, how="inner", on=join_keys)
    delta["matches_total"] = delta["home_matches"] + delta["away_matches"]
    delta["avg_rating_delta_home_minus_away"] = (
        delta["home_avg_overall_rating"]
        - delta["away_avg_overall_rating"]
    )
    delta["abs_avg_rating_delta"] = (
        delta["avg_rating_delta_home_minus_away"].abs()
    )
    delta["eligible_both_sides"] = (
        delta["home_matches"] >= RQ4_MIN_MATCHES_PER_SIDE_FOR_DELTA
    ) & (
        delta["away_matches"] >= RQ4_MIN_MATCHES_PER_SIDE_FOR_DELTA
    )

    return delta[
        [
            "season",
            "season_label",
            "player",
            "player_id",
            "home_matches",
            "away_matches",
            "matches_total",
            "home_avg_overall_rating",
            "away_avg_overall_rating",
            "avg_rating_delta_home_minus_away",
            "abs_avg_rating_delta",
            "home_median_overall_rating",
            "away_median_overall_rating",
            "home_best_overall_rating",
            "away_best_overall_rating",
            "home_worst_overall_rating",
            "away_worst_overall_rating",
            "home_starts",
            "away_starts",
            "home_motm_awards",
            "away_motm_awards",
            "home_teams",
            "away_teams",
            "eligible_both_sides",
        ]
    ].sort_values(
        [
            "season",
            "abs_avg_rating_delta",
            "avg_rating_delta_home_minus_away",
            "player",
        ],
        ascending=[True, False, False, True],
        kind="stable",
    )


def build_rq9_team_match_efficiency(
    rq9_df: pd.DataFrame,
    team_age_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build one team-match row with goals, shots, and average age.

    Input: normalized RQ9 DataFrame and team age summary DataFrame.
    Output: DataFrame for `rq9_team_match_efficiency.csv`.
    """
    team_lookup = team_age_df[
        ["season", "season_label", "team", "avg_age"]
    ].drop_duplicates()
    match_df = (
        rq9_df.groupby(
            ["season", "season_label", "game_id", "team"],
            dropna=False,
        )
        .agg(
            goals=("team_goals", first_non_null),
            shots=("team_shots", first_non_null),
        )
        .reset_index()
        .merge(
            team_lookup,
            how="left",
            on=["season", "season_label", "team"],
        )
    )
    match_df["goals_per_shot"] = safe_ratio(
        match_df["goals"],
        match_df["shots"],
    )
    return match_df[
        [
            "season",
            "season_label",
            "game_id",
            "team",
            "avg_age",
            "goals",
            "shots",
            "goals_per_shot",
        ]
    ].sort_values(["season", "game_id", "team"], kind="stable")


def build_rq9_team_age_vs_efficiency(
    team_match_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate team efficiency against average age.

    Input: team-match DataFrame.
    Output: DataFrame for `rq9_team_age_vs_efficiency.csv`.
    """
    team_df = (
        team_match_df.groupby(
            ["season", "season_label", "team", "avg_age"],
            dropna=False,
        )
        .agg(
            matches=("game_id", "nunique"),
            total_goals=("goals", "sum"),
            total_shots=("shots", "sum"),
        )
        .reset_index()
    )
    team_df["goals_per_shot"] = safe_ratio(
        team_df["total_goals"],
        team_df["total_shots"],
    )
    return team_df[
        [
            "season",
            "season_label",
            "team",
            "avg_age",
            "matches",
            "total_goals",
            "total_shots",
            "goals_per_shot",
        ]
    ].sort_values(
        ["season", "goals_per_shot", "team"],
        ascending=[True, False, True],
        kind="stable",
    )


def build_rq9_player_age_profile(rq9_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player efficiency by integer age band.

    Input: normalized RQ9 DataFrame.
    Output: DataFrame for `rq9_player_age_profile.csv`.
    """
    working = rq9_df.dropna(subset=["age"]).copy()
    working["age_int"] = np.floor(working["age"]).astype(int)
    profile = (
        working.groupby(["season", "age_int"], dropna=False)
        .agg(
            players=("player_id", "nunique"),
            total_goals=("player_goals", "sum"),
            total_shots=("player_shots", "sum"),
        )
        .reset_index()
    )
    profile = profile.loc[profile["total_shots"] > 0].copy()
    profile["goals_per_shot"] = safe_ratio(
        profile["total_goals"],
        profile["total_shots"],
    )
    return profile.sort_values(
        ["season", "goals_per_shot", "total_shots", "age_int"],
        ascending=[True, False, False, True],
        kind="stable",
    )


def build_best_age_candidate(
    profile: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    """Pick the best eligible player age band for one season scope.

    Input: age profile DataFrame and season label.
    Output: one-row DataFrame or empty DataFrame.
    """
    eligible = profile.loc[
        profile["total_shots"] >= BEST_AGE_MIN_TOTAL_SHOTS
    ].copy()
    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "min_total_shots",
                "best_age_int",
                "goals_per_shot",
                "total_shots",
                "total_goals",
                "players",
            ]
        )

    best_row = (
        eligible.sort_values(
            ["goals_per_shot", "total_shots", "total_goals", "age_int"],
            ascending=[False, False, False, True],
            kind="stable",
        )
        .head(1)
        .copy()
    )
    if "season" in best_row.columns:
        best_row["season"] = season
    else:
        best_row.insert(0, "season", season)

    best_row.insert(1, "min_total_shots", BEST_AGE_MIN_TOTAL_SHOTS)
    best_row = best_row.rename(columns={"age_int": "best_age_int"})
    return best_row[
        [
            "season",
            "min_total_shots",
            "best_age_int",
            "goals_per_shot",
            "total_shots",
            "total_goals",
            "players",
        ]
    ]


def build_rq9_player_best_age(rq9_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the strongest player age band per season and overall.

    Input: normalized RQ9 DataFrame.
    Output: DataFrame for `rq9_player_best_age.csv`.
    """
    rows = []
    per_season_profile = build_rq9_player_age_profile(rq9_df)
    for season, group in per_season_profile.groupby("season", sort=True):
        rows.append(build_best_age_candidate(group.copy(), str(season)))

    overall = rq9_df.dropna(subset=["age"]).copy()
    overall["age_int"] = np.floor(overall["age"]).astype(int)
    overall["player_season_key"] = (
        overall["season"].astype(str)
        + "::"
        + overall["player_id"].astype(str)
    )
    overall_profile = (
        overall.groupby("age_int", dropna=False)
        .agg(
            players=("player_season_key", "nunique"),
            total_goals=("player_goals", "sum"),
            total_shots=("player_shots", "sum"),
        )
        .reset_index()
    )
    overall_profile = overall_profile.loc[
        overall_profile["total_shots"] > 0
    ].copy()
    overall_profile["goals_per_shot"] = safe_ratio(
        overall_profile["total_goals"],
        overall_profile["total_shots"],
    )
    rows.append(build_best_age_candidate(overall_profile, "all"))
    return pd.concat(rows, ignore_index=True)


def build_quadratic_model_row(
    scope: str,
    season: str,
    team_df: pd.DataFrame,
) -> dict[str, object]:
    """Fit one quadratic age-efficiency model summary row.

    Input: scope label, season label, and team summary DataFrame.
    Output: dictionary with model summary values.
    """
    valid = team_df.dropna(subset=["avg_age", "goals_per_shot"]).copy()
    if valid.empty:
        return {
            "scope": scope,
            "season": season,
            "n_teams": 0,
            "pearson_r_age_efficiency": np.nan,
            "estimated_peak_age": np.nan,
            "estimated_peak_goals_per_shot": np.nan,
            "model_note": "No valid team rows were available.",
        }

    pearson = fit_correlation(valid["avg_age"], valid["goals_per_shot"])
    min_age = float(valid["avg_age"].min())
    max_age = float(valid["avg_age"].max())

    if len(valid) < 3:
        return {
            "scope": scope,
            "season": season,
            "n_teams": int(len(valid)),
            "pearson_r_age_efficiency": pearson,
            "estimated_peak_age": np.nan,
            "estimated_peak_goals_per_shot": np.nan,
            "model_note": "Not enough rows for a quadratic model.",
        }

    quad_a, quad_b, quad_c = np.polyfit(
        valid["avg_age"],
        valid["goals_per_shot"],
        2,
    )
    if quad_a >= 0:
        return {
            "scope": scope,
            "season": season,
            "n_teams": int(len(valid)),
            "pearson_r_age_efficiency": pearson,
            "estimated_peak_age": np.nan,
            "estimated_peak_goals_per_shot": np.nan,
            "model_note": "Quadratic model has no concave maximum.",
        }

    peak_age = -quad_b / (2 * quad_a)
    peak_efficiency = quad_a * peak_age**2 + quad_b * peak_age + quad_c
    if peak_age < min_age or peak_age > max_age:
        note = (
            "The fitted quadratic peak lies outside the observed "
            "team-average age range "
            f"({min_age:.2f}-{max_age:.2f}), so the data does not support "
            "one exact optimal team-average age."
        )
        return {
            "scope": scope,
            "season": season,
            "n_teams": int(len(valid)),
            "pearson_r_age_efficiency": pearson,
            "estimated_peak_age": np.nan,
            "estimated_peak_goals_per_shot": np.nan,
            "model_note": note,
        }

    return {
        "scope": scope,
        "season": season,
        "n_teams": int(len(valid)),
        "pearson_r_age_efficiency": pearson,
        "estimated_peak_age": float(peak_age),
        "estimated_peak_goals_per_shot": float(peak_efficiency),
        "model_note": "",
    }


def build_rq9_optimal_age_summary(team_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize RQ9 age-efficiency model results.

    Input: team summary DataFrame.
    Output: DataFrame for `rq9_optimal_age_summary.csv`.
    """
    rows = [build_quadratic_model_row("all_seasons", "all", team_df)]
    for season, group in team_df.groupby("season", sort=True):
        rows.append(
            build_quadratic_model_row(
                "single_season",
                str(season),
                group.copy(),
            )
        )
    return pd.DataFrame.from_records(rows)


def build_analysis_tables(
    rq9_df: pd.DataFrame,
    rq4_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build all derived analysis tables used by the docs pages.

    Input: raw RQ9 and RQ4 DataFrames.
    Output: dictionary from relative CSV path to DataFrame.
    """
    rq9 = normalize_rq9(rq9_df)
    rq4 = normalize_rq4(rq4_df)

    season_age = build_season_age_summary(rq9)
    team_age = build_team_age_summary(rq9)
    rq4_ratings = build_rq4_home_away_player_ratings(rq4)
    rq4_delta = build_rq4_player_home_away_delta(rq4_ratings)
    rq9_match = build_rq9_team_match_efficiency(rq9, team_age)
    rq9_team = build_rq9_team_age_vs_efficiency(rq9_match)
    rq9_profile = build_rq9_player_age_profile(rq9)
    rq9_best_age = build_rq9_player_best_age(rq9)
    rq9_optimal = build_rq9_optimal_age_summary(rq9_team)

    return {
        "other/bundesliga_season_age_summary.csv": season_age,
        "other/bundesliga_team_age_summary.csv": team_age,
        "rq4/rq4_home_away_player_ratings.csv": rq4_ratings,
        "rq4/rq4_player_home_away_delta.csv": rq4_delta,
        "rq9/rq9_team_match_efficiency.csv": rq9_match,
        "rq9/rq9_team_age_vs_efficiency.csv": rq9_team,
        "rq9/rq9_player_age_profile.csv": rq9_profile,
        "rq9/rq9_player_best_age.csv": rq9_best_age,
        "rq9/rq9_optimal_age_summary.csv": rq9_optimal,
    }


def load_raw_outputs(
    input_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two raw pipeline CSV files from disk.

    Input: directory with raw output files.
    Output: tuple with `(rq9_df, rq4_df)`.
    """
    rq9_path = Path(input_dir) / RAW_RQ9_FILE
    rq4_path = Path(input_dir) / RAW_RQ4_FILE

    if not rq9_path.exists():
        raise FileNotFoundError(f"Missing raw RQ9 file: {rq9_path}")
    if not rq4_path.exists():
        raise FileNotFoundError(f"Missing raw RQ4 file: {rq4_path}")

    return pd.read_csv(rq9_path), pd.read_csv(rq4_path)


def write_analysis_outputs(
    tables: dict[str, pd.DataFrame],
    output_root: Path = DEFAULT_DOCS_DATA_DIR,
) -> list[Path]:
    """Write all derived analysis tables to disk.

    Input: table dictionary and output root path.
    Output: list of written file paths.
    """
    written_paths: list[Path] = []
    for relative_path, df in tables.items():
        path = Path(output_root) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        written_paths.append(path)
    return written_paths


def build_terminal_answers(
    tables: dict[str, pd.DataFrame],
) -> dict[str, str]:
    """Build short answer strings for the terminal output.

    Input: dictionary with derived analysis tables.
    Output: dictionary with short text answers for RQ4 and RQ9.
    """
    rq4_ratings = tables["rq4/rq4_home_away_player_ratings.csv"]
    rq9_team = tables["rq9/rq9_team_age_vs_efficiency.csv"]
    rq9_best_age = tables["rq9/rq9_player_best_age.csv"]
    rq9_optimal = tables["rq9/rq9_optimal_age_summary.csv"]

    rq4_home = rq4_ratings.loc[
        rq4_ratings["eligible_for_leaderboard"]
        & (rq4_ratings["home_away"] == "home"),
        "avg_overall_rating",
    ]
    rq4_away = rq4_ratings.loc[
        rq4_ratings["eligible_for_leaderboard"]
        & (rq4_ratings["home_away"] == "away"),
        "avg_overall_rating",
    ]
    mean_home = float(rq4_home.mean()) if not rq4_home.empty else np.nan
    mean_away = float(rq4_away.mean()) if not rq4_away.empty else np.nan
    mean_delta = mean_home - mean_away

    top_home_row = rq4_ratings.loc[
        rq4_ratings["eligible_for_leaderboard"]
        & (rq4_ratings["home_away"] == "home")
    ].head(1)
    top_away_row = rq4_ratings.loc[
        rq4_ratings["eligible_for_leaderboard"]
        & (rq4_ratings["home_away"] == "away")
    ].head(1)

    best_age_row = rq9_best_age.loc[rq9_best_age["season"] != "all"].head(1)
    if best_age_row.empty:
        best_age_row = rq9_best_age.head(1)

    optimal_row = rq9_optimal.loc[
        rq9_optimal["scope"] == "single_season"
    ].head(1)
    if optimal_row.empty:
        optimal_row = rq9_optimal.head(1)

    pearson = (
        float(optimal_row.iloc[0]["pearson_r_age_efficiency"])
        if not optimal_row.empty
        else np.nan
    )
    best_age_int = (
        int(best_age_row.iloc[0]["best_age_int"])
        if not best_age_row.empty
        else None
    )
    best_age_efficiency = (
        float(best_age_row.iloc[0]["goals_per_shot"])
        if not best_age_row.empty
        else np.nan
    )
    min_team_age = (
        float(rq9_team["avg_age"].min()) if not rq9_team.empty else np.nan
    )
    max_team_age = (
        float(rq9_team["avg_age"].max()) if not rq9_team.empty else np.nan
    )
    model_note = (
        str(optimal_row.iloc[0]["model_note"]).strip()
        if not optimal_row.empty
        else ""
    )

    top_home_player = (
        str(top_home_row.iloc[0]["player"])
        if not top_home_row.empty
        else "n/a"
    )
    top_away_player = (
        str(top_away_row.iloc[0]["player"])
        if not top_away_row.empty
        else "n/a"
    )

    rq4_answer = (
        "RQ4 | "
        f"top_home={top_home_player} | "
        f"top_away={top_away_player} | "
        f"mean_home={mean_home:.3f} | "
        f"mean_away={mean_away:.3f} | "
        f"delta={mean_delta:+.3f}"
    )

    rq9_answer = (
        "RQ9 | "
        f"pearson={pearson:.3f} | "
        f"team_age_range={min_team_age:.2f}-{max_team_age:.2f} | "
        f"best_player_age_band={best_age_int} | "
        f"band_goals_per_shot={best_age_efficiency:.3f}"
    )
    if model_note and model_note.lower() != "nan":
        rq9_answer = f"{rq9_answer} | quadratic_peak=outside_range"

    return {"rq4": rq4_answer, "rq9": rq9_answer}


def main(argv: Iterable[str] | None = None) -> int:
    """Generate the analysis CSV files from existing raw outputs.

    Input: optional iterable of CLI argument strings.
    Output: process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Generate derived docs/data CSVs for RQ4 and RQ9."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_DOCS_DATA_DIR,
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    rq9_df, rq4_df = load_raw_outputs(args.input_dir)
    tables = build_analysis_tables(rq9_df, rq4_df)
    written_paths = write_analysis_outputs(tables, args.output_root)

    print("[ok] Generated analysis CSVs:")
    for path in written_paths:
        print(f" - {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
