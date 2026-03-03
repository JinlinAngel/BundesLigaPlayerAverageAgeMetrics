from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable


STARTUP_PACKAGES = {
    "pandas": "pandas",
    "requests": "requests",
}

_DEPENDENCY_STATUS: dict[str, bool] = {}


def ensure_package(import_name: str, package_name: str) -> bool:
    if import_name in _DEPENDENCY_STATUS:
        return _DEPENDENCY_STATUS[import_name]

    try:
        importlib.import_module(import_name)
        _DEPENDENCY_STATUS[import_name] = True
        return True
    except ImportError:
        print(f"[setup] Paket '{package_name}' fehlt. Installation wird gestartet ...")

    if import_name == "soccerdata" and sys.version_info >= (3, 14):
        print(
            "[setup] soccerdata ist mit Python "
            f"{sys.version_info.major}.{sys.version_info.minor} derzeit nicht kompatibel "
            "(benoetigt typischerweise Python 3.12 oder 3.13 wegen lxml-Binary-Wheels)."
        )
        _DEPENDENCY_STATUS[import_name] = False
        return False

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    except subprocess.CalledProcessError:
        _DEPENDENCY_STATUS[import_name] = False
        return False

    try:
        importlib.import_module(import_name)
        _DEPENDENCY_STATUS[import_name] = True
        return True
    except ImportError:
        _DEPENDENCY_STATUS[import_name] = False
        return False


for _import_name, _package_name in STARTUP_PACKAGES.items():
    if not ensure_package(_import_name, _package_name):
        raise RuntimeError(
            f"Automatische Installation von '{_package_name}' fehlgeschlagen."
        )

import pandas as pd
import numpy as np


DEFAULT_SEASONS = ["2324", "2425"]  # 2023/24 und 2024/25
DEFAULT_LEAGUE = "GER-Bundesliga"
DEFAULT_SOURCE_PRIORITY = ["espn", "fbref"]
ESPN_MATCH_SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/{league_id}/summary?event={game_id}"
)
ESPN_ATHLETE_URL = "https://site.web.api.espn.com/apis/common/v3/sports/soccer/athletes/{athlete_id}"


@dataclass
class Config:
    seasons: list[str]
    league: str
    soccerdata_cache_dir: Path
    player_cache_dir: Path
    output_dir: Path
    refresh: bool
    no_cache: bool
    no_store: bool
    proxy: str | None
    source_priority: list[str]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Berechnet Durchschnittsalter von Bundesliga-Spielern je Team und "
            "Saison (Standard: 2023/24 + 2024/25)."
        )
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=DEFAULT_SEASONS,
        help="Saisoncodes fuer soccerdata, z. B. 2324 2425 (Default).",
    )
    parser.add_argument(
        "--league",
        default=DEFAULT_LEAGUE,
        help="League-Code fuer soccerdata (Default: GER-Bundesliga).",
    )
    parser.add_argument(
        "--soccerdata-cache-dir",
        default="./.soccerdata_cache",
        help="Cache-Ordner fuer soccerdata.",
    )
    parser.add_argument(
        "--player-cache-dir",
        default="./data/player_cache",
        help="Eigenes Zwischenspeicher-Verzeichnis fuer Spielerlisten je Saison.",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/outputs",
        help="Ausgabeverzeichnis fuer CSV-Dateien.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignoriert vorhandene saisonale Player-Caches und laedt neu.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Setzt SOCCERDATA_NOCACHE=true.",
    )
    parser.add_argument(
        "--no-store",
        action="store_true",
        help="Setzt SOCCERDATA_NOSTORE=true.",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="Optionaler Proxy fuer soccerdata, z. B. socks5://127.0.0.1:9050.",
    )
    parser.add_argument(
        "--source-priority",
        nargs="+",
        default=DEFAULT_SOURCE_PRIORITY,
        help=(
            "Abrufreihenfolge der Datenquellen, z. B. "
            "--source-priority espn fbref (Default) oder --source-priority fbref espn."
        ),
    )
    return parser.parse_args(argv)


def configure_env(config: Config) -> None:
    config.soccerdata_cache_dir.mkdir(parents=True, exist_ok=True)
    config.player_cache_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["SOCCERDATA_DIR"] = str(config.soccerdata_cache_dir.resolve())
    os.environ.setdefault("SOCCERDATA_LOGLEVEL", "INFO")
    if config.no_cache:
        os.environ["SOCCERDATA_NOCACHE"] = "true"
    if config.no_store:
        os.environ["SOCCERDATA_NOSTORE"] = "true"


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        cols: list[str] = []
        for col in out.columns:
            pieces = [str(p).strip() for p in col if p and str(p).strip()]
            cols.append("_".join(pieces) if pieces else "col")
        out.columns = cols
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def parse_age_to_float(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return float("nan")

    if "-" in text:
        # FBref kann Alter als "YY-DDD" (Jahre-Tage) liefern.
        try:
            years, days = text.split("-", 1)
            return int(years) + int(days) / 365.25
        except (TypeError, ValueError):
            return float("nan")

    try:
        return float(text)
    except ValueError:
        return float("nan")


def parse_espn_display_dob(value: object) -> date | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    if "T" in text:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def age_years_at_reference_date(birth_date: date, reference_date: date) -> float:
    return (reference_date - birth_date).days / 365.25


def parse_season_years(season_code: str) -> tuple[int, int] | None:
    code = str(season_code).strip()
    if len(code) != 4 or not code.isdigit():
        return None

    left = int(code[:2])
    right = int(code[2:])
    if right == (left + 1) % 100:
        start = 2000 + left
        end = 2000 + right
        if end <= start:
            end += 100
        return start, end

    full_year = int(code)
    if 1900 <= full_year <= 2100:
        return full_year, full_year + 1
    return None


def normalize_season_label(season_code: str) -> str:
    years = parse_season_years(season_code)
    if years is None:
        return str(season_code).strip()
    start, end = years
    return f"{start}/{end}"


def season_reference_date(season_code: str) -> date:
    years = parse_season_years(season_code)
    if years is not None:
        _, end = years
        return date(end, 6, 30)
    return date.today()


def cache_file_for_season(cache_dir: Path, league: str, season: str) -> Path:
    safe_league = league.replace("/", "_").replace(" ", "_")
    return cache_dir / f"{safe_league}_{season}_players.csv"


def build_requests_proxies(proxy: str | None) -> dict[str, str] | None:
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


def load_json_from_cache_or_url(
    url: str,
    cache_file: Path,
    refresh: bool,
    proxy: str | None = None,
    timeout: int = 30,
) -> dict:
    if cache_file.exists() and not refresh:
        with cache_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    import requests

    response = requests.get(
        url,
        timeout=timeout,
        proxies=build_requests_proxies(proxy),
    )
    response.raise_for_status()
    payload = response.json()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return payload


def should_log_progress(current: int, total: int, interval: int) -> bool:
    if total <= 0:
        return True
    return current == 1 or current == total or (current % max(1, interval) == 0)


def format_progress(label: str, current: int, total: int) -> str:
    if total <= 0:
        return f"[progress] {label}: 0/0 (100.0%)"
    percent = (current / total) * 100
    return f"[progress] {label}: {current}/{total} ({percent:.1f}%)"


def load_players_from_fbref(config: Config, season: str, sd_module) -> pd.DataFrame:
    fbref = sd_module.FBref(
        leagues=config.league,
        seasons=[season],
        no_cache=config.no_cache,
        no_store=config.no_store,
        proxy=config.proxy,
    )
    raw = fbref.read_player_season_stats(stat_type="standard")
    if raw is None or raw.empty:
        raise RuntimeError(f"Keine FBref-Daten fuer Saison {season} gefunden.")

    players = flatten_columns(raw).reset_index()

    needed = {"league", "season", "team", "player"}
    missing = [c for c in needed if c not in players.columns]
    if missing:
        raise RuntimeError(
            f"Erwartete Spalten fehlen fuer Saison {season}: {', '.join(missing)}"
        )
    if "age" not in players.columns:
        raise RuntimeError(
            f"Spalte 'age' fehlt in den Daten fuer Saison {season}. "
            "Alternative Datenquellen: ESPN."
        )

    players["season"] = players["season"].astype(str)
    players["season_label"] = players["season"].apply(normalize_season_label)
    players["age_ref"] = players["age"].apply(parse_age_to_float)
    players["data_source"] = "FBref"

    keep_cols = [
        c
        for c in players.columns
        if c
        in {
            "league",
            "season",
            "season_label",
            "team",
            "player",
            "age",
            "age_ref",
            "minutes",
            "playing_time_min",
            "data_source",
        }
    ]
    return players[keep_cols].copy()


def load_players_from_espn(config: Config, season: str, sd_module) -> pd.DataFrame:
    espn = sd_module.ESPN(
        leagues=config.league,
        seasons=[season],
        no_cache=config.no_cache,
        no_store=config.no_store,
        proxy=config.proxy,
    )
    schedule = espn.read_schedule().reset_index()
    if schedule.empty:
        raise RuntimeError(f"Keine ESPN-Spielplan-Daten fuer Saison {season} gefunden.")

    required_cols = {"game_id", "league_id"}
    missing_schedule_cols = [c for c in required_cols if c not in schedule.columns]
    if missing_schedule_cols:
        raise RuntimeError(
            f"ESPN-Spielplan enthaelt nicht alle erwarteten Spalten: "
            f"{', '.join(missing_schedule_cols)}"
        )

    summary_cache_dir = config.player_cache_dir / "_espn_match_summaries" / str(season)
    roster_rows: list[dict] = []
    total_matches = len(schedule)
    summary_cache_hits = 0
    summary_downloads = 0
    summary_interval = max(1, total_matches // 20)
    print(
        f"[info] ESPN Saison {season}: Verarbeite {total_matches} Spiele "
        "(Match-Summaries)."
    )
    for idx, (_, match) in enumerate(schedule.iterrows(), start=1):
        game_id = str(match["game_id"])
        league_id = str(match["league_id"])
        url = ESPN_MATCH_SUMMARY_URL.format(league_id=league_id, game_id=game_id)
        cache_file = summary_cache_dir / f"{game_id}.json"
        from_cache = cache_file.exists() and not config.refresh
        data = load_json_from_cache_or_url(
            url=url,
            cache_file=cache_file,
            refresh=config.refresh,
            proxy=config.proxy,
        )
        if from_cache:
            summary_cache_hits += 1
        else:
            summary_downloads += 1
        if should_log_progress(idx, total_matches, summary_interval):
            print(
                (
                    f"{format_progress('Match-Summaries', idx, total_matches)} | "
                    f"cache={summary_cache_hits}, download={summary_downloads}"
                ),
                flush=True,
            )

        for team_data in data.get("rosters", []):
            team_name = (
                team_data.get("team", {}).get("displayName")
                or team_data.get("team", {}).get("name")
            )
            if not team_name:
                continue
            for player_data in team_data.get("roster", []):
                athlete = player_data.get("athlete", {})
                player_id = str(athlete.get("id") or "").strip()
                player_name = (
                    athlete.get("displayName") or athlete.get("fullName") or ""
                ).strip()
                if not player_id or not player_name:
                    continue
                roster_rows.append(
                    {
                        "league": config.league,
                        "season": str(season),
                        "team": team_name,
                        "player": player_name,
                        "player_id": player_id,
                        "game_id": game_id,
                    }
                )

    if not roster_rows:
        raise RuntimeError(f"Keine ESPN-Rosterdaten fuer Saison {season} gefunden.")

    roster_df = pd.DataFrame.from_records(roster_rows)
    players = (
        roster_df.groupby(
            ["league", "season", "team", "player", "player_id"], dropna=False
        )
        .agg(appearances=("game_id", "nunique"))
        .reset_index()
    )

    athlete_cache_dir = config.player_cache_dir / "_espn_athletes"
    athlete_profiles: list[dict] = []
    unique_player_ids = sorted(players["player_id"].dropna().astype(str).unique())
    total_athletes = len(unique_player_ids)
    athlete_cache_hits = 0
    athlete_downloads = 0
    athlete_interval = max(1, total_athletes // 20)
    print(
        f"[info] ESPN Saison {season}: Verarbeite {total_athletes} Spielerprofile "
        "(Athlete API)."
    )
    for idx, player_id in enumerate(unique_player_ids, start=1):
        athlete_url = ESPN_ATHLETE_URL.format(athlete_id=player_id)
        athlete_cache_file = athlete_cache_dir / f"{player_id}.json"
        from_cache = athlete_cache_file.exists() and not config.refresh
        try:
            athlete_payload = load_json_from_cache_or_url(
                url=athlete_url,
                cache_file=athlete_cache_file,
                refresh=config.refresh,
                proxy=config.proxy,
            )
        except Exception:
            athlete_profiles.append(
                {"player_id": player_id, "display_dob": None, "age_current": None}
            )
        else:
            athlete_data = athlete_payload.get("athlete", {})
            athlete_profiles.append(
                {
                    "player_id": player_id,
                    "display_dob": athlete_data.get("displayDOB"),
                    "age_current": athlete_data.get("age"),
                }
            )
        if from_cache:
            athlete_cache_hits += 1
        else:
            athlete_downloads += 1
        if should_log_progress(idx, total_athletes, athlete_interval):
            print(
                (
                    f"{format_progress('Spielerprofile', idx, total_athletes)} | "
                    f"cache={athlete_cache_hits}, download={athlete_downloads}"
                ),
                flush=True,
            )

    athlete_df = pd.DataFrame.from_records(athlete_profiles)
    players = players.merge(athlete_df, how="left", on="player_id")

    reference_date = season_reference_date(season)
    birth_dates = players["display_dob"].apply(parse_espn_display_dob)
    players["age_ref"] = birth_dates.apply(
        lambda b: age_years_at_reference_date(b, reference_date)
        if isinstance(b, date)
        else float("nan")
    )
    players["age_ref"] = players["age_ref"].fillna(
        pd.to_numeric(players["age_current"], errors="coerce")
    )
    players["age"] = players["age_ref"].round(2)
    players["season_label"] = players["season"].apply(normalize_season_label)
    players["data_source"] = "ESPN"

    keep_cols = [
        "league",
        "season",
        "season_label",
        "team",
        "player",
        "player_id",
        "appearances",
        "age",
        "age_ref",
        "data_source",
    ]
    return players[keep_cols].copy()


def load_or_fetch_players_for_season(config: Config, season: str) -> pd.DataFrame:
    if not ensure_package("soccerdata", "soccerdata"):
        pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(
            "Paket 'soccerdata' fehlt oder konnte nicht automatisch installiert werden. "
            f"Aktuelle Python-Version: {pyver}. "
            "Bitte mit einer kompatiblen Python-Version "
            "(empfohlen 3.12/3.13) installieren: pip install soccerdata pandas"
        )
    import soccerdata as sd

    cache_file = cache_file_for_season(config.player_cache_dir, config.league, season)
    if cache_file.exists() and not config.refresh:
        cached_df = pd.read_csv(cache_file)
        if "age_ref" not in cached_df.columns and "age" in cached_df.columns:
            cached_df["age_ref"] = cached_df["age"].apply(parse_age_to_float)
        cached_age = pd.to_numeric(cached_df.get("age_ref"), errors="coerce")
        if cached_age.notna().any() and float(cached_age.median()) > 80:
            print(
                f"[warn] Verwerfe veralteten Cache fuer Saison {season} "
                "(unplausible Alterswerte erkannt)."
            )
        else:
            return cached_df

    source_loaders = {
        "fbref": load_players_from_fbref,
        "espn": load_players_from_espn,
    }
    source_errors: list[str] = []
    final_df: pd.DataFrame | None = None
    for source in config.source_priority:
        loader = source_loaders[source]
        print(f"[info] Versuche Datenquelle '{source.upper()}' fuer Saison {season} ...")
        try:
            final_df = loader(config, season, sd)
            print(f"[ok] Datenquelle '{source.upper()}' erfolgreich fuer Saison {season}.")
            break
        except Exception as exc:
            print(
                f"[warn] Datenquelle '{source.upper()}' fehlgeschlagen "
                f"({type(exc).__name__}): {exc}"
            )
            source_errors.append(f"{source.upper()}: {exc}")

    if final_df is None:
        raise RuntimeError(
            f"Alle Datenquellen fehlgeschlagen fuer Saison {season}: "
            + " | ".join(source_errors)
        )

    final_df.to_csv(cache_file, index=False)
    return final_df


def compute_team_player_table(players_df: pd.DataFrame) -> pd.DataFrame:
    table = players_df[["season", "season_label", "team", "player", "age_ref"]].copy()
    table = table.sort_values(["season", "team", "player"], kind="stable")
    return table


def compute_team_summary(players_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        players_df.groupby(["season", "season_label", "team"], dropna=False)
        .agg(
            player_count=("player", "nunique"),
            avg_age=("age_ref", "mean"),
            min_age=("age_ref", "min"),
            max_age=("age_ref", "max"),
            missing_age=("age_ref", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(
        ["season", "avg_age", "team"], ascending=[True, False, True], kind="stable"
    )
    return grouped


def deduplicate_for_league_view(players_df: pd.DataFrame) -> pd.DataFrame:
    dedup = players_df.copy()
    minute_col = None
    for candidate in ("playing_time_min", "minutes", "appearances"):
        if candidate in dedup.columns:
            minute_col = candidate
            break
    if minute_col:
        dedup[minute_col] = pd.to_numeric(dedup[minute_col], errors="coerce").fillna(-1)
        dedup = dedup.sort_values(
            ["season", "player", minute_col],
            ascending=[True, True, False],
            kind="stable",
        )
    else:
        dedup = dedup.sort_values(["season", "player"], kind="stable")
    dedup = dedup.drop_duplicates(subset=["season", "player"], keep="first")
    return dedup


def compute_season_and_global_summary(
    players_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dedup = deduplicate_for_league_view(players_df)
    season_summary = (
        dedup.groupby(["season", "season_label"], dropna=False)
        .agg(
            unique_players=("player", "nunique"),
            avg_age=("age_ref", "mean"),
            min_age=("age_ref", "min"),
            max_age=("age_ref", "max"),
            missing_age=("age_ref", lambda s: int(s.isna().sum())),
        )
        .reset_index()
        .sort_values(["season"], kind="stable")
    )

    global_summary = pd.DataFrame(
        [
            {
                "scope": "all_team_entries",
                "unique_players": int(players_df["player"].nunique()),
                "avg_age": players_df["age_ref"].mean(),
                "min_age": players_df["age_ref"].min(),
                "max_age": players_df["age_ref"].max(),
                "missing_age": int(players_df["age_ref"].isna().sum()),
            },
            {
                "scope": "dedup_per_player_and_season",
                "unique_players": int(dedup["player"].nunique()),
                "avg_age": dedup["age_ref"].mean(),
                "min_age": dedup["age_ref"].min(),
                "max_age": dedup["age_ref"].max(),
                "missing_age": int(dedup["age_ref"].isna().sum()),
            },
        ]
    )
    return season_summary, global_summary


def _parse_int(value: object) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
    except TypeError:
        if value is None:
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _fix_mojibake(text: str) -> str:
    if "Ã" not in text and "Â" not in text:
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def compute_rq9_age_vs_efficiency(config: Config, team_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for season in config.seasons:
        summary_dir = config.player_cache_dir / "_espn_match_summaries" / str(season)
        if not summary_dir.exists():
            print(
                f"[warn] RQ9: Kein Match-Summary-Cache fuer Saison {season} gefunden "
                f"({summary_dir})."
            )
            continue
        for match_file in sorted(summary_dir.glob("*.json")):
            with match_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            competitors = payload.get("header", {}).get("competitions", [])
            competitors = competitors[0].get("competitors", []) if competitors else []
            goals_by_side: dict[str, int] = {}
            for comp in competitors:
                side = str(comp.get("homeAway") or "").strip().lower()
                if not side:
                    continue
                goals = _parse_int(comp.get("score"))
                if goals is None:
                    continue
                goals_by_side[side] = goals

            for team_data in payload.get("boxscore", {}).get("teams", []):
                side = str(team_data.get("homeAway") or "").strip().lower()
                team_name = (
                    team_data.get("team", {}).get("displayName")
                    or team_data.get("team", {}).get("name")
                    or ""
                ).strip()
                if not side or not team_name:
                    continue
                team_name = _fix_mojibake(team_name)
                stats = team_data.get("statistics", [])
                shots_raw = next(
                    (s.get("displayValue") for s in stats if s.get("name") == "totalShots"),
                    None,
                )
                shots = _parse_int(shots_raw)
                goals = goals_by_side.get(side)
                if shots is None or goals is None:
                    continue
                rows.append(
                    {
                        "season": str(season),
                        "season_label": normalize_season_label(str(season)),
                        "game_id": match_file.stem,
                        "team": team_name,
                        "home_away": side,
                        "goals": goals,
                        "shots": shots,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "season",
                "season_label",
                "team",
                "avg_age",
                "matches",
                "home_matches",
                "away_matches",
                "total_goals",
                "total_shots",
                "goals_per_shot",
            ]
        )

    match_stats = pd.DataFrame.from_records(rows)
    aggregated = (
        match_stats.groupby(["season", "season_label", "team"], dropna=False)
        .agg(
            matches=("game_id", "nunique"),
            home_matches=("home_away", lambda s: int((s == "home").sum())),
            away_matches=("home_away", lambda s: int((s == "away").sum())),
            total_goals=("goals", "sum"),
            total_shots=("shots", "sum"),
        )
        .reset_index()
    )
    aggregated["goals_per_shot"] = (
        aggregated["total_goals"] / aggregated["total_shots"].replace({0: pd.NA})
    )
    aggregated["season"] = aggregated["season"].astype(str)
    team_age = team_summary[["season", "team", "avg_age"]].copy()
    team_age["season"] = team_age["season"].astype(str)
    rq9 = aggregated.merge(
        team_age,
        how="left",
        on=["season", "team"],
    )
    rq9 = rq9[
        [
            "season",
            "season_label",
            "team",
            "avg_age",
            "matches",
            "home_matches",
            "away_matches",
            "total_goals",
            "total_shots",
            "goals_per_shot",
        ]
    ]
    rq9 = rq9.sort_values(
        ["season", "goals_per_shot", "team"],
        ascending=[True, False, True],
        kind="stable",
    )
    return rq9


def _fit_rq9_peak(df: pd.DataFrame, scope: str, season: str | None = None) -> dict:
    base = {
        "scope": scope,
        "season": season or "all",
        "n_teams": int(len(df)),
        "pearson_r_age_efficiency": float("nan"),
        "estimated_peak_age": float("nan"),
        "estimated_peak_goals_per_shot": float("nan"),
        "model_note": "",
    }
    if df.empty or len(df) < 3:
        base["model_note"] = "Zu wenige Datenpunkte fuer Peak-Schaetzung."
        return base

    x = pd.to_numeric(df["avg_age"], errors="coerce")
    y = pd.to_numeric(df["goals_per_shot"], errors="coerce")
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3:
        base["model_note"] = "Zu wenige gueltige Datenpunkte nach Bereinigung."
        return base

    base["pearson_r_age_efficiency"] = float(valid["x"].corr(valid["y"]))

    coeff = np.polyfit(valid["x"], valid["y"], 2)
    a, b, c = float(coeff[0]), float(coeff[1]), float(coeff[2])
    age_min = float(valid["x"].min())
    age_max = float(valid["x"].max())

    if math.isclose(a, 0.0, abs_tol=1e-12):
        base["model_note"] = "Quadratisches Modell nahezu linear; kein stabiles Peak-Alter."
        return base
    if a >= 0:
        base["model_note"] = "Kurve nicht konkav; kein internes Maximum im beobachteten Bereich."
        return base

    peak_age = -b / (2 * a)
    if not (age_min <= peak_age <= age_max):
        base["model_note"] = (
            "Maximum liegt ausserhalb des beobachteten Altersbereichs "
            f"({age_min:.2f}-{age_max:.2f})."
        )
        return base

    peak_eff = a * (peak_age**2) + b * peak_age + c
    base["estimated_peak_age"] = float(peak_age)
    base["estimated_peak_goals_per_shot"] = float(peak_eff)
    base["model_note"] = "Quadratische Schaetzung innerhalb des beobachteten Altersbereichs."
    return base


def compute_rq9_optimal_age_summary(rq9_summary: pd.DataFrame) -> pd.DataFrame:
    valid = rq9_summary.dropna(subset=["avg_age", "goals_per_shot"]).copy()
    rows: list[dict] = []
    rows.append(_fit_rq9_peak(valid, scope="all_seasons", season=None))
    for season, group in valid.groupby("season", dropna=False):
        rows.append(_fit_rq9_peak(group, scope="single_season", season=str(season)))
    out = pd.DataFrame.from_records(rows)
    out = out.sort_values(["scope", "season"], kind="stable").reset_index(drop=True)
    return out


def compute_rq9_player_age_profile(config: Config, players_df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate player-level goals/shots across matches and join with player age.
    stats_rows: list[dict] = []
    for season in config.seasons:
        summary_dir = config.player_cache_dir / "_espn_match_summaries" / str(season)
        if not summary_dir.exists():
            continue
        for match_file in sorted(summary_dir.glob("*.json")):
            with match_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            for team_data in payload.get("rosters", []):
                team_name = (
                    team_data.get("team", {}).get("displayName")
                    or team_data.get("team", {}).get("name")
                    or ""
                ).strip()
                team_name = _fix_mojibake(team_name)
                for player_data in team_data.get("roster", []):
                    athlete = player_data.get("athlete", {})
                    player_id = str(athlete.get("id") or "").strip()
                    player_name = (
                        athlete.get("displayName") or athlete.get("fullName") or ""
                    ).strip()
                    if not player_id or not player_name:
                        continue
                    pstats = player_data.get("stats", []) or []
                    goals = next(
                        (_parse_int(s.get("displayValue")) for s in pstats if s.get("name") == "totalGoals"),
                        None,
                    )
                    shots = next(
                        (_parse_int(s.get("displayValue")) for s in pstats if s.get("name") == "totalShots"),
                        None,
                    )
                    if goals is None or shots is None:
                        continue
                    stats_rows.append(
                        {
                            "season": str(season),
                            "team": team_name,
                            "player": _fix_mojibake(player_name),
                            "player_id": player_id,
                            "goals": goals,
                            "shots": shots,
                        }
                    )
    if not stats_rows:
        return pd.DataFrame(
            columns=[
                "season",
                "age_int",
                "players",
                "total_goals",
                "total_shots",
                "goals_per_shot",
            ]
        )

    player_match = pd.DataFrame.from_records(stats_rows)
    player_totals = (
        player_match.groupby(["season", "team", "player", "player_id"], dropna=False)
        .agg(total_goals=("goals", "sum"), total_shots=("shots", "sum"))
        .reset_index()
    )
    player_totals["player_id"] = player_totals["player_id"].astype(str)

    age_lookup = players_df[["season", "team", "player", "player_id", "age_ref"]].copy()
    age_lookup["season"] = age_lookup["season"].astype(str)
    age_lookup["player_id"] = age_lookup["player_id"].astype(str)
    age_lookup = age_lookup.drop_duplicates(
        subset=["season", "team", "player", "player_id"], keep="first"
    )

    merged = player_totals.merge(
        age_lookup,
        how="left",
        on=["season", "team", "player", "player_id"],
    )
    merged["age_int"] = pd.to_numeric(merged["age_ref"], errors="coerce").round().astype("Int64")
    merged = merged.dropna(subset=["age_int"])
    merged = merged[merged["total_shots"] > 0].copy()

    by_age = (
        merged.groupby(["season", "age_int"], dropna=False)
        .agg(
            players=("player_id", "nunique"),
            total_goals=("total_goals", "sum"),
            total_shots=("total_shots", "sum"),
        )
        .reset_index()
    )
    by_age["goals_per_shot"] = by_age["total_goals"] / by_age["total_shots"]
    by_age = by_age.sort_values(
        ["season", "goals_per_shot", "age_int"], ascending=[True, False, True], kind="stable"
    )
    return by_age


def compute_rq9_player_best_age(player_age_profile: pd.DataFrame, min_total_shots: int = 80) -> pd.DataFrame:
    if player_age_profile.empty:
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
    eligible = player_age_profile[
        pd.to_numeric(player_age_profile["total_shots"], errors="coerce") >= int(min_total_shots)
    ].copy()
    if eligible.empty:
        return pd.DataFrame(
            [
                {
                    "season": "all",
                    "min_total_shots": int(min_total_shots),
                    "best_age_int": pd.NA,
                    "goals_per_shot": pd.NA,
                    "total_shots": pd.NA,
                    "total_goals": pd.NA,
                    "players": pd.NA,
                }
            ]
        )

    out_rows: list[dict] = []
    for season, group in eligible.groupby("season", dropna=False):
        best = group.sort_values(
            ["goals_per_shot", "total_shots", "players"],
            ascending=[False, False, False],
            kind="stable",
        ).iloc[0]
        out_rows.append(
            {
                "season": season,
                "min_total_shots": int(min_total_shots),
                "best_age_int": int(best["age_int"]),
                "goals_per_shot": float(best["goals_per_shot"]),
                "total_shots": int(best["total_shots"]),
                "total_goals": int(best["total_goals"]),
                "players": int(best["players"]),
            }
        )

    all_group = (
        eligible.groupby("age_int", dropna=False)
        .agg(
            players=("players", "sum"),
            total_goals=("total_goals", "sum"),
            total_shots=("total_shots", "sum"),
        )
        .reset_index()
    )
    all_group["goals_per_shot"] = all_group["total_goals"] / all_group["total_shots"]
    best_all = all_group.sort_values(
        ["goals_per_shot", "total_shots", "players"],
        ascending=[False, False, False],
        kind="stable",
    ).iloc[0]
    out_rows.append(
        {
            "season": "all",
            "min_total_shots": int(min_total_shots),
            "best_age_int": int(best_all["age_int"]),
            "goals_per_shot": float(best_all["goals_per_shot"]),
            "total_shots": int(best_all["total_shots"]),
            "total_goals": int(best_all["total_goals"]),
            "players": int(best_all["players"]),
        }
    )

    return pd.DataFrame.from_records(out_rows).sort_values(["season"], kind="stable")


def save_outputs(
    player_table: pd.DataFrame,
    team_summary: pd.DataFrame,
    season_summary: pd.DataFrame,
    global_summary: pd.DataFrame,
    rq9_summary: pd.DataFrame,
    rq9_peak_summary: pd.DataFrame,
    rq9_player_age_profile: pd.DataFrame,
    rq9_player_best_age: pd.DataFrame,
    output_dir: Path,
) -> None:
    player_table.to_csv(output_dir / "bundesliga_team_player_ages.csv", index=False)
    team_summary.to_csv(output_dir / "bundesliga_team_age_summary.csv", index=False)
    season_summary.to_csv(output_dir / "bundesliga_season_age_summary.csv", index=False)
    global_summary.to_csv(output_dir / "bundesliga_global_age_summary.csv", index=False)
    rq9_summary.to_csv(output_dir / "bundesliga_rq9_age_vs_efficiency.csv", index=False)
    rq9_peak_summary.to_csv(
        output_dir / "bundesliga_rq9_optimal_age_summary.csv", index=False
    )
    rq9_player_age_profile.to_csv(
        output_dir / "bundesliga_rq9_player_age_profile.csv", index=False
    )
    rq9_player_best_age.to_csv(
        output_dir / "bundesliga_rq9_player_best_age.csv", index=False
    )


def print_report(
    player_table: pd.DataFrame,
    team_summary: pd.DataFrame,
    season_summary: pd.DataFrame,
    global_summary: pd.DataFrame,
    rq9_summary: pd.DataFrame,
    rq9_peak_summary: pd.DataFrame,
    rq9_player_age_profile: pd.DataFrame,
    rq9_player_best_age: pd.DataFrame,
) -> None:
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.width", 200)

    print("\n=== SPIELER JE TEAM UND SAISON (ALTER) ===")
    for (season, team), group in player_table.groupby(["season_label", "team"], sort=True):
        print(f"\n[{season}] {team}")
        display_group = group[["player", "age_ref"]].copy()
        display_group["age_ref"] = display_group["age_ref"].round(2)
        print(display_group.to_string(index=False))
        team_avg = display_group["age_ref"].mean()
        print(f"Team-Durchschnittsalter: {team_avg:.2f}")

    print("\n=== TEAM-DURCHSCHNITTSALTER JE SAISON ===")
    team_display = team_summary.copy()
    for col in ("avg_age", "min_age", "max_age"):
        team_display[col] = team_display[col].round(2)
    print(team_display.to_string(index=False))

    print("\n=== SAISON-ZUSAMMENFASSUNG (LIGAWEIT, DEDUP JE SPIELER/SAISON) ===")
    season_display = season_summary.copy()
    for col in ("avg_age", "min_age", "max_age"):
        season_display[col] = season_display[col].round(2)
    print(season_display.to_string(index=False))

    print("\n=== GESAMT-DURCHSCHNITTSALTER UEBER ALLE TEAMS/SAISONS ===")
    global_display = global_summary.copy()
    for col in ("avg_age", "min_age", "max_age"):
        global_display[col] = global_display[col].round(2)
    print(global_display.to_string(index=False))

    print("\n=== RQ9: TEAMALTER VS EFFIZIENZ (TORE PRO SCHUSS) ===")
    rq9_display = rq9_summary.copy()
    for col in ("avg_age", "goals_per_shot"):
        rq9_display[col] = pd.to_numeric(rq9_display[col], errors="coerce").round(3)
    print(rq9_display.to_string(index=False))

    print("\n=== RQ9: GESCHAETZTES 'BESTES' ALTER ===")
    peak_display = rq9_peak_summary.copy()
    for col in (
        "pearson_r_age_efficiency",
        "estimated_peak_age",
        "estimated_peak_goals_per_shot",
    ):
        peak_display[col] = pd.to_numeric(peak_display[col], errors="coerce").round(3)
    print(peak_display.to_string(index=False))

    print("\n=== RQ9: SPIELER-ALTERSPROFIL (TORE PRO SCHUSS) ===")
    profile_display = rq9_player_age_profile.copy()
    if not profile_display.empty:
        profile_display["goals_per_shot"] = (
            pd.to_numeric(profile_display["goals_per_shot"], errors="coerce").round(3)
        )
    print(profile_display.to_string(index=False))

    print("\n=== RQ9: 'BESTES' SPIELERALTER (NACH ALTERSGRUPPE) ===")
    best_display = rq9_player_best_age.copy()
    if not best_display.empty:
        best_display["goals_per_shot"] = (
            pd.to_numeric(best_display["goals_per_shot"], errors="coerce").round(3)
        )
    print(best_display.to_string(index=False))


def build_config(args: argparse.Namespace) -> Config:
    seasons = [str(s).strip() for s in args.seasons if str(s).strip()]
    if not seasons:
        raise SystemExit("Mindestens eine Saison muss angegeben werden.")
    source_priority = [
        str(s).strip().lower() for s in args.source_priority if str(s).strip()
    ]
    if not source_priority:
        raise SystemExit("Mindestens eine Datenquelle in --source-priority angeben.")
    allowed_sources = {"fbref", "espn"}
    invalid_sources = [s for s in source_priority if s not in allowed_sources]
    if invalid_sources:
        raise SystemExit(
            "Ungueltige Datenquelle in --source-priority: "
            f"{', '.join(invalid_sources)}. Erlaubt sind: fbref, espn."
        )
    return Config(
        seasons=seasons,
        league=str(args.league).strip(),
        soccerdata_cache_dir=Path(args.soccerdata_cache_dir),
        player_cache_dir=Path(args.player_cache_dir),
        output_dir=Path(args.output_dir),
        refresh=bool(args.refresh),
        no_cache=bool(args.no_cache),
        no_store=bool(args.no_store),
        proxy=args.proxy,
        source_priority=source_priority,
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    configure_env(config)

    season_frames: list[pd.DataFrame] = []
    try:
        total_seasons = len(config.seasons)
        for idx, season in enumerate(config.seasons, start=1):
            print(format_progress("Saisons gesamt", idx, total_seasons))
            print(f"[info] Lade Saison {season} ...")
            season_frames.append(load_or_fetch_players_for_season(config, season))
    except Exception as exc:
        print(f"[error] Datenabruf fehlgeschlagen: {exc}")
        print(
            "[hint] FBref kann durch Cloudflare mit HTTP 403 blockiert werden. "
            "Dieses Skript nutzt dann automatisch ESPN als Fallback. "
            "Falls auch das scheitert, pruefe Netzwerk/Proxy (Option --proxy)."
        )
        return 1

    players_df = pd.concat(season_frames, ignore_index=True)

    player_table = compute_team_player_table(players_df)
    team_summary = compute_team_summary(players_df)
    season_summary, global_summary = compute_season_and_global_summary(players_df)
    rq9_summary = compute_rq9_age_vs_efficiency(config, team_summary)
    rq9_peak_summary = compute_rq9_optimal_age_summary(rq9_summary)
    rq9_player_age_profile = compute_rq9_player_age_profile(config, players_df)
    rq9_player_best_age = compute_rq9_player_best_age(
        rq9_player_age_profile, min_total_shots=80
    )

    save_outputs(
        player_table=player_table,
        team_summary=team_summary,
        season_summary=season_summary,
        global_summary=global_summary,
        rq9_summary=rq9_summary,
        rq9_peak_summary=rq9_peak_summary,
        rq9_player_age_profile=rq9_player_age_profile,
        rq9_player_best_age=rq9_player_best_age,
        output_dir=config.output_dir,
    )
    print_report(
        player_table,
        team_summary,
        season_summary,
        global_summary,
        rq9_summary,
        rq9_peak_summary,
        rq9_player_age_profile,
        rq9_player_best_age,
    )

    print("\n[ok] CSV-Ausgaben gespeichert in:", config.output_dir.resolve())
    print("[ok] Saison-Player-Caches in:", config.player_cache_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
