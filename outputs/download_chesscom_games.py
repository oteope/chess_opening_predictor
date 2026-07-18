#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
download_chesscom_games.py

Downloads all available PGN archives for one or more Chess.com players and
concatenates them into a single PGN file.

Usage (single player):
    python outputs/download_chesscom_games.py \
        --player hikaru \
        --output data/raw/hikaru.pgn

Usage (multiple players from a text file):
    python outputs/download_chesscom_games.py \
        --players data/players/pro_players.txt \
        --output data/raw/pro_players.pgn

Requirements:
    - requests
"""

import argparse
import os
import sys
from typing import List, Optional

import requests


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

CHESSCOM_API_BASE = "https://api.chess.com/pub/player/{player}/games/{year}/{month}"
CHESSCOM_ARCHIVES_URL = "https://api.chess.com/pub/player/{player}/games/archives"

HEADERS = {
    "User-Agent": "ChessOpeningPredictor/1.0 (Educational Machine Learning Project)"
}

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


def _ensure_output_dir(path: str) -> None:
    """
    Create the parent directory of *path* if it does not exist.

    Args:
        path: Full path to the output file.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
        print(f"[INFO] Created output directory: {parent}")


def _fetch_json(url: str) -> Optional[dict]:
    """
    Perform a GET request and return the JSON response.

    Args:
        url: URL to fetch.

    Returns:
        Parsed JSON dictionary on success, None on failure.
    """
    try:
        resp = requests.get(url, timeout=30, headers=HEADERS)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        print(f"[WARN] HTTP error while fetching {url}: {exc}", file=sys.stderr)
        return None


def _fetch_pgn_text(url: str) -> Optional[str]:
    """
    Download the PGN text from a Chess.com monthly archive URL.

    Args:
        url: URL of the monthly PGN archive.

    Returns:
        Raw PGN text on success, None on failure.
    """
    try:
        resp = requests.get(url, timeout=60, headers=HEADERS)
        resp.raise_for_status()
        # Chess.com returns plain text (PGN) for these endpoints
        return resp.text
    except requests.exceptions.RequestException as exc:
        print(f"[WARN] HTTP error while downloading PGN from {url}: {exc}",
              file=sys.stderr)
        return None


def _get_archive_urls(player: str) -> List[str]:
    """
    Retrieve the list of monthly archive URLs for a given player.

    Args:
        player: Chess.com username.

    Returns:
        List of archive URLs (one per month). Empty list if the player
        does not exist or the API call fails.
    """
    url = CHESSCOM_ARCHIVES_URL.format(player=player)
    data = _fetch_json(url)
    if data is None:
        return []
    archives = data.get("archives", [])
    if not archives:
        print(f"[WARN] No archives found for player '{player}'.")
    return archives


def _download_monthly_pgn(archive_url: str) -> Optional[str]:
    """
    Download the PGN text for a single monthly archive.

    Args:
        archive_url: Full URL of the monthly archive (e.g.
            https://api.chess.com/pub/player/hikaru/games/2024/01).

    Returns:
        PGN text on success, None on failure.
    """
    # The PGN endpoint is the same URL with "/pgn" appended
    pgn_url = archive_url + "/pgn"
    return _fetch_pgn_text(pgn_url)


def _download_player_games_to_file(player: str, out_f) -> None:
    """
    Download all monthly PGN archives for *player* and write them
    into the already opened file object *out_f*.

    Args:
        player: Chess.com username.
        out_f:  An open file object (mode 'w' or 'a') where the PGN
                text will be written.
    """
    print(f"[INFO] Fetching archive list for player '{player}'...")
    archive_urls = _get_archive_urls(player)
    if not archive_urls:
        print(f"[WARN] No archives available for player '{player}'. Skipping.")
        return

    total = len(archive_urls)
    print(f"[INFO] Found {total} monthly archive(s) for '{player}'.")

    for idx, archive_url in enumerate(archive_urls, start=1):
        # Extract year/month for display
        parts = archive_url.rstrip("/").split("/")
        year_month = f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else archive_url

        print(f"[INFO] Downloading {year_month} ({idx}/{total}) for '{player}'...")
        pgn_text = _download_monthly_pgn(archive_url)
        if pgn_text is None:
            print(f"[WARN] Skipping {year_month} (download failed).")
            continue

        # Write the PGN text, ensuring a trailing newline
        out_f.write(pgn_text)
        if not pgn_text.endswith("\n"):
            out_f.write("\n")


def download_player_games(player: str, output_path: str) -> None:
    """
    Download all monthly PGN archives for *player* and concatenate them
    into a single file at *output_path*.

    Args:
        player: Chess.com username.
        output_path: Path where the final PGN file will be saved.
    """
    _ensure_output_dir(output_path)

    with open(output_path, "w", encoding="utf-8") as out_f:
        _download_player_games_to_file(player, out_f)

    print(f"[INFO] All available games for '{player}' saved to: {output_path}")


def _read_player_list(path: str) -> List[str]:
    """
    Read a text file containing one Chess.com username per line.

    Empty lines and duplicate usernames are ignored. The order of
    first occurrence is preserved.

    Args:
        path: Path to the text file.

    Returns:
        List of unique, non‑empty usernames.
    """
    players = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            if name not in seen:
                seen.add(name)
                players.append(name)
    return players


def download_multiple_players(players: List[str], output_path: str) -> None:
    """
    Download all monthly PGN archives for every player in *players*
    and concatenate them into a single file at *output_path*.

    The download continues even if one player fails.

    Args:
        players:     List of Chess.com usernames.
        output_path: Path where the final PGN file will be saved.
    """
    _ensure_output_dir(output_path)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for player in players:
            _download_player_games_to_file(player, out_f)

    print(f"[INFO] All available games for {len(players)} player(s) "
          f"saved to: {output_path}")


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

def main() -> None:
    """
    Parse command-line arguments and start the download process.
    """
    parser = argparse.ArgumentParser(
        description="Download all PGN archives for one or more Chess.com "
                    "players and concatenate them into a single file."
    )

    # Mutually exclusive group: exactly one of --player / --players
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--player",
        help="Chess.com username (e.g. hikaru).",
    )
    group.add_argument(
        "--players",
        help="Path to a text file containing one Chess.com username per line.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output PGN file (e.g. data/raw/hikaru.pgn).",
    )

    args = parser.parse_args()

    if args.player:
        download_player_games(args.player, args.output)
    else:
        # args.players is guaranteed to be set because of the mutually
        # exclusive group
        players = _read_player_list(args.players)
        if not players:
            print("[ERROR] The players file is empty or contains no valid "
                  "usernames.",
                  file=sys.stderr)
            sys.exit(1)
        download_multiple_players(players, args.output)


if __name__ == "__main__":
    main()
