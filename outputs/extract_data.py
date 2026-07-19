#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_data.py

Script to extract features from chess games from CSV or PGN files,
generating a dataset with one-hot representation of the board at ply 20.

Usage:
    python extract_data.py --input <path> --output <path> --source <source_name>

Requirements:
    - pandas
    - python-chess
    - numpy (optional, used internally by pandas)
"""

import os
import sys
import argparse
import pandas as pd
import chess
import chess.pgn

try:
    import eco
    _ECO_AVAILABLE = True
except ImportError:
    _ECO_AVAILABLE = False

# ----------------------------------------------------------------------
# Auxiliary functions
# ----------------------------------------------------------------------

def parse_winner_csv(winner_str):
    """
    Converts the value of the 'winner' column from a CSV to an integer:
        'white' -> 1
        'draw'  -> 0
        'black' -> 2
    Any other value returns None (the row will be discarded).
    """
    w = str(winner_str).strip().lower()
    if w == 'white':
        return 1
    elif w == 'draw':
        return 0
    elif w == 'black':
        return 2
    else:
        return None


def parse_result_pgn(result_str):
    """
    Converts the result of a PGN game to an integer:
        "1-0"      -> 1
        "1/2-1/2"  -> 0
        "0-1"      -> 2
    Any other value returns None.
    """
    r = result_str.strip()
    if r == "1-0":
        return 1
    elif r == "1/2-1/2":
        return 0
    elif r == "0-1":
        return 2
    else:
        return None


def extract_one_hot(board):
    """
    Generates a dictionary with 768 keys of the form '{square}_{color}_{piece}'
    where each value is 1 if the piece is present on that square, 0 otherwise.

    The order of pieces is: Pawn, Knight, Bishop, Rook, Queen, King
    (both for whites and blacks).
    """
    # List of pieces in standard order (without color)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                   chess.ROOK, chess.QUEEN, chess.KING]
    # Short names for each type
    piece_names = ['Pawn', 'Knight', 'Bishop', 'Rook', 'Queen', 'King']

    # Initialize dictionary with zeros
    features = {}
    for square in chess.SQUARES:
        sq_name = chess.SQUARE_NAMES[square]  # e.g.: 'e4'
        for color in (chess.WHITE, chess.BLACK):
            color_name = 'W' if color == chess.WHITE else 'B'
            for ptype, pname in zip(piece_types, piece_names):
                key = f"{sq_name}_{color_name}_{pname}"
                features[key] = 0

    # Populate with the pieces from the board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            sq_name = chess.SQUARE_NAMES[square]
            color_name = 'W' if piece.color == chess.WHITE else 'B'
            ptype = piece.piece_type
            # Index of the piece type (0-5)
            idx = piece_types.index(ptype)
            pname = piece_names[idx]
            key = f"{sq_name}_{color_name}_{pname}"
            features[key] = 1

    return features


def _infer_opening_and_eco(board: chess.Board,
                           eco_db: 'eco.Eco') -> tuple:
    """
    Returns (opening_name, eco_code) using the ECO database.
    If the database is not available or the position is not recognized,
    falls back to ("Unknown", "Unknown").
    """
    if not _ECO_AVAILABLE or eco_db is None:
        return "Unknown", "Unknown"
    try:
        eco_code, opening_name = eco_db.eco_of_game(board)
        # ensure strings
        if opening_name is None:
            opening_name = "Unknown"
        if eco_code is None:
            eco_code = "Unknown"
        return opening_name, eco_code
    except Exception:
        return "Unknown", "Unknown"


def write_output_chunk(df_chunk, output_path, first_chunk):
    """
    Saves a DataFrame to the output file.
    If first_chunk is True, writes the header (mode='w').
    Otherwise, appends without header (mode='a', header=False).
    """
    if first_chunk:
        df_chunk.to_csv(output_path, mode='w', index=False)
    else:
        df_chunk.to_csv(output_path, mode='a', header=False, index=False)


# ----------------------------------------------------------------------
# Processing CSV files (Kaggle)
# ----------------------------------------------------------------------

def process_csv(input_path, output_path, source_name, max_games=None):
    """
    Reads the CSV file in chunks of 1000 rows, parses the moves,
    applies quality filters and extracts features at ply 20.
    Saves results incrementally.
    """
    first_chunk = True
    # Initialize ECO database if available (for opening inference)
    eco_db = None
    if _ECO_AVAILABLE:
        try:
            eco_db = eco.Eco()
        except Exception:
            eco_db = None
    processed = 0
    skipped = 0

    # Read the CSV in streaming mode (chunksize=1000)
    for chunk_idx, chunk in enumerate(pd.read_csv(input_path, chunksize=1000)):
        rows = []  # list of dictionaries to build the final DataFrame

        for _, row in chunk.iterrows():
            # --- Mapping columns from the original CSV file ---
            try:
                elo_white = int(row['white_rating'])
                elo_black = int(row['black_rating'])
            except (ValueError, TypeError, KeyError):
                # If it can't be converted to integer, discard the row
                skipped += 1
                continue

            # Quality filter: Elo values must be valid integers (already checked)
            # (no need for further checking)

            # Get the 'winner' column and convert it
            winner_val = parse_winner_csv(row.get('winner', ''))
            if winner_val is None:
                skipped += 1
                continue

            # Get opening name and ECO code (they may be missing)
            opening_name = row.get('opening_name', '')
            eco_code = row.get('opening_eco', '')

            # --- Parsing moves ---
            moves_str = str(row.get('moves', ''))
            if not moves_str:
                skipped += 1
                continue

            # Split by spaces
            move_list = moves_str.split()
            # We need at least 20 plies (10 full moves)
            if len(move_list) < 20:
                skipped += 1
                continue

            # Create a board and apply the first 20 moves
            board = chess.Board()
            valid = True
            for i in range(20):
                try:
                    board.push_san(move_list[i])
                except (chess.InvalidMoveError, chess.IllegalMoveError,
                        chess.AmbiguousMoveError, ValueError):
                    valid = False
                    break
            if not valid:
                skipped += 1
                continue

            # --- Infer opening/ECO if missing in PGN headers ---
            if not opening_name or not eco_code:
                inferred_opening, inferred_eco = _infer_opening_and_eco(board, eco_db)
                if not opening_name:
                    opening_name = inferred_opening
                if not eco_code:
                    eco_code = inferred_eco

            # --- Extraction of one-hot features ---
            one_hot = extract_one_hot(board)

            # --- Build output row ---
            out_row = {
                'Elo_White': elo_white,
                'Elo_Black': elo_black,
                'Elo_Difference': elo_white - elo_black,
                'Opening_Name': opening_name,
                'ECO_Code': eco_code,
                'Result': winner_val,
                'Source': source_name,
            }
            # Add the 768 one-hot columns
            out_row.update(one_hot)

            rows.append(out_row)
            processed += 1

            # Check if we have reached the limit
            if max_games is not None and processed >= max_games:
                break

        # If any rows processed in this chunk, save them
        if rows:
            df_chunk = pd.DataFrame(rows)
            write_output_chunk(df_chunk, output_path, first_chunk)
            first_chunk = False

        # Stop reading further chunks if limit reached
        if max_games is not None and processed >= max_games:
            break

    print(f"[CSV] Processing completed. Output file: {output_path}")
    print(f"Successfully processed: {processed} games")
    print(f"Skipped: {skipped} games")


# ----------------------------------------------------------------------
# Processing PGN files (Pro)
# ----------------------------------------------------------------------

def process_pgn(input_path, output_path, source_name, max_games=None):
    """
    Reads the PGN game by game using chess.pgn.read_game(),
    applies the same filters and extracts features at ply 20.
    Saves results incrementally.
    """
    first_chunk = True
    rows = []  # accumulate rows to write in batches (every 1000)
    processed = 0
    skipped = 0

    # Initialize ECO database if available (for opening inference)
    eco_db = None
    if _ECO_AVAILABLE:
        try:
            eco_db = eco.Eco()
        except Exception:
            eco_db = None

    with open(input_path, 'r', encoding='utf-8') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            # --- Mapping headers from the PGN ---
            try:
                elo_white = int(game.headers.get('WhiteElo', ''))
                elo_black = int(game.headers.get('BlackElo', ''))
            except (ValueError, TypeError):
                # If it can't be converted to integer, discard the game
                skipped += 1
                continue

            # Quality filter: Elo values must be valid integers (already checked)

            # Get result from PGN
            result_str = game.headers.get('Result', '')
            result_val = parse_result_pgn(result_str)
            if result_val is None:
                skipped += 1
                continue

            # --- Obtain the move list ---
            # Extract the sequence of moves from the mainline game
            board = game.board()
            move_list = list(game.mainline_moves())

            # We need at least 20 plies (10 full moves)
            if len(move_list) < 20:
                skipped += 1
                continue

            # Apply the first 20 moves
            valid = True
            for i in range(20):
                try:
                    board.push(move_list[i])
                except (chess.InvalidMoveError, chess.IllegalMoveError,
                        chess.AmbiguousMoveError):
                    valid = False
                    break
            if not valid:
                skipped += 1
                continue

            # --- Opening detection from the board position ---
            opening_name, eco_code = _infer_opening_and_eco(board, eco_db)

            # --- Extraction of one-hot features ---
            one_hot = extract_one_hot(board)

            # --- Build output row ---
            out_row = {
                'Elo_White': elo_white,
                'Elo_Black': elo_black,
                'Elo_Difference': elo_white - elo_black,
                'Opening_Name': opening_name,
                'ECO_Code': eco_code,
                'Result': result_val,
                'Source': source_name,
            }
            out_row.update(one_hot)

            rows.append(out_row)
            processed += 1

            # Write every 1000 rows to avoid accumulating too much memory
            if len(rows) >= 1000:
                df_chunk = pd.DataFrame(rows)
                write_output_chunk(df_chunk, output_path, first_chunk)
                first_chunk = False
                rows = []

            # Check if we have reached the limit
            if max_games is not None and processed >= max_games:
                break

    # Write the remaining rows
    if rows:
        df_chunk = pd.DataFrame(rows)
        write_output_chunk(df_chunk, output_path, first_chunk)

    print(f"[PGN] Processing completed. Output file: {output_path}")
    print(f"Successfully processed: {processed} games")
    print(f"Skipped: {skipped} games")


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extracts features from chess games (CSV or PGN) "
                    "and generates a dataset with one-hot representation at ply 20."
    )
    parser.add_argument('--input', required=True,
                        help='Path to the input file (.csv or .pgn)')
    parser.add_argument('--output', required=True,
                        help='Path to the output file (.csv)')
    parser.add_argument('--source', required=True,
                        help='Name of the data source (e.g: kaggle, pros)')
    parser.add_argument('--max-games', type=int, default=None,
                        help='Maximum number of successfully processed games (optional)')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    source_name = args.source
    max_games = args.max_games

    # Detect format by extension
    _, ext = os.path.splitext(input_path)
    ext = ext.lower()

    if ext == '.csv':
        print(f"Detected CSV format: {input_path}")
        process_csv(input_path, output_path, source_name, max_games)
    elif ext == '.pgn':
        print(f"Detected PGN format: {input_path}")
        process_pgn(input_path, output_path, source_name, max_games)
    else:
        print(f"Error: unsupported extension '{ext}'. Use .csv or .pgn.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
