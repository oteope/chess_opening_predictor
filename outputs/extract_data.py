#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_data.py

Script para extraer características de partidas de ajedrez desde archivos CSV o PGN,
generando un dataset con representación one‑hot del tablero en el ply 20.

Uso:
    python extract_data.py --input <ruta> --output <ruta> --source <nombre_fuente>

Requisitos:
    - pandas
    - python-chess
    - numpy (opcional, usado internamente por pandas)
"""

import os
import sys
import argparse
import pandas as pd
import chess
import chess.pgn

# ----------------------------------------------------------------------
# Funciones auxiliares
# ----------------------------------------------------------------------

def parse_winner_csv(winner_str):
    """
    Convierte el valor de la columna 'winner' de un CSV a entero:
        'white' -> 1
        'draw'  -> 0
        'black' -> 2
    Cualquier otro valor devuelve None (se descartará la fila).
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
    Convierte el resultado de una partida PGN a entero:
        "1-0"      -> 1
        "1/2-1/2"  -> 0
        "0-1"      -> 2
    Cualquier otro valor devuelve None.
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
    Genera un diccionario con 768 claves de la forma '{casilla}_{color}_{pieza}'
    donde cada valor es 1 si la pieza está presente en esa casilla, 0 en caso contrario.

    El orden de las piezas es: Pawn, Knight, Bishop, Rook, Queen, King
    (tanto para blancas como para negras).
    """
    # Lista de piezas en orden estándar (sin color)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                   chess.ROOK, chess.QUEEN, chess.KING]
    # Nombres cortos para cada tipo
    piece_names = ['Pawn', 'Knight', 'Bishop', 'Rook', 'Queen', 'King']

    # Inicializar diccionario con ceros
    features = {}
    for square in chess.SQUARES:
        sq_name = chess.SQUARE_NAMES[square]  # ej: 'e4'
        for color in (chess.WHITE, chess.BLACK):
            color_name = 'W' if color == chess.WHITE else 'B'
            for ptype, pname in zip(piece_types, piece_names):
                key = f"{sq_name}_{color_name}_{pname}"
                features[key] = 0

    # Poblar con las piezas actuales del tablero
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            sq_name = chess.SQUARE_NAMES[square]
            color_name = 'W' if piece.color == chess.WHITE else 'B'
            ptype = piece.piece_type
            # Índice del tipo de pieza (0‑5)
            idx = piece_types.index(ptype)
            pname = piece_names[idx]
            key = f"{sq_name}_{color_name}_{pname}"
            features[key] = 1

    return features


def write_output_chunk(df_chunk, output_path, first_chunk):
    """
    Guarda un DataFrame en el archivo de salida.
    Si first_chunk es True, escribe la cabecera (mode='w').
    En caso contrario, añade sin cabecera (mode='a', header=False).
    """
    if first_chunk:
        df_chunk.to_csv(output_path, mode='w', index=False)
    else:
        df_chunk.to_csv(output_path, mode='a', header=False, index=False)


# ----------------------------------------------------------------------
# Procesamiento de archivos CSV (Kaggle)
# ----------------------------------------------------------------------

def process_csv(input_path, output_path, source_name):
    """
    Lee el archivo CSV por chunks de 1000 filas, parsea las jugadas,
    aplica filtros de calidad y extrae las features en el ply 20.
    Guarda los resultados de forma incremental.
    """
    first_chunk = True

    # Leer el CSV en modo streaming (chunksize=1000)
    for chunk_idx, chunk in enumerate(pd.read_csv(input_path, chunksize=1000)):
        rows = []  # lista de diccionarios para construir el DataFrame final

        for _, row in chunk.iterrows():
            # --- Mapeo de columnas del CSV original ---
            try:
                elo_white = int(row['white_rating'])
                elo_black = int(row['black_rating'])
            except (ValueError, TypeError, KeyError):
                # Si no se puede convertir a entero, se descarta la fila
                continue

            # Filtro de calidad: Elos deben ser enteros válidos (ya comprobado)
            # (no se necesita más comprobación)

            # Obtener la columna 'winner' y convertirla
            winner_val = parse_winner_csv(row.get('winner', ''))
            if winner_val is None:
                continue

            # Obtener nombre de apertura y ECO (pueden faltar)
            opening_name = row.get('opening_name', '')
            eco_code = row.get('opening_eco', '')

            # --- Parseo de jugadas ---
            moves_str = str(row.get('moves', ''))
            if not moves_str:
                continue

            # Dividir por espacios
            move_list = moves_str.split()
            # Necesitamos al menos 20 plies (10 jugadas completas)
            if len(move_list) < 20:
                continue

            # Crear un tablero y aplicar las primeras 20 jugadas
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
                continue

            # --- Extracción de features one‑hot ---
            one_hot = extract_one_hot(board)

            # --- Construir fila de salida ---
            out_row = {
                'Elo_White': elo_white,
                'Elo_Black': elo_black,
                'Opening_Name': opening_name,
                'ECO_Code': eco_code,
                'Result': winner_val,
                'Source': source_name,
            }
            # Añadir las 768 columnas one‑hot
            out_row.update(one_hot)

            rows.append(out_row)

        # Si hay filas procesadas en este chunk, guardarlas
        if rows:
            df_chunk = pd.DataFrame(rows)
            write_output_chunk(df_chunk, output_path, first_chunk)
            first_chunk = False

    print(f"[CSV] Procesamiento completado. Archivo de salida: {output_path}")


# ----------------------------------------------------------------------
# Procesamiento de archivos PGN (Pro)
# ----------------------------------------------------------------------

def process_pgn(input_path, output_path, source_name):
    """
    Lee el archivo PGN partida a partida usando chess.pgn.read_game(),
    aplica los mismos filtros y extrae las features en el ply 20.
    Guarda los resultados de forma incremental.
    """
    first_chunk = True
    rows = []  # acumulamos filas para escribir en lotes (cada 1000)

    with open(input_path, 'r', encoding='utf-8') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            # --- Mapeo de headers del PGN ---
            try:
                elo_white = int(game.headers.get('WhiteElo', ''))
                elo_black = int(game.headers.get('BlackElo', ''))
            except (ValueError, TypeError):
                # Si no se puede convertir a entero, se descarta la partida
                continue

            # Filtro de calidad: Elos deben ser enteros válidos (ya comprobado)

            # Obtener resultado del PGN
            result_str = game.headers.get('Result', '')
            result_val = parse_result_pgn(result_str)
            if result_val is None:
                continue

            # Obtener nombre de apertura y ECO
            opening_name = game.headers.get('Opening', '')
            eco_code = game.headers.get('ECO', '')

            # --- Obtener la lista de jugadas ---
            # Extraemos la secuencia de movimientos desde la partida principal
            board = game.board()
            move_list = list(game.mainline_moves())

            # Necesitamos al menos 20 plies (10 jugadas completas)
            if len(move_list) < 20:
                continue

            # Aplicar las primeras 20 jugadas
            board = chess.Board()
            valid = True
            for i in range(20):
                try:
                    board.push(move_list[i])
                except (chess.InvalidMoveError, chess.IllegalMoveError,
                        chess.AmbiguousMoveError):
                    valid = False
                    break
            if not valid:
                continue

            # --- Extracción de features one‑hot ---
            one_hot = extract_one_hot(board)

            # --- Construir fila de salida ---
            out_row = {
                'Elo_White': elo_white,
                'Elo_Black': elo_black,
                'Opening_Name': opening_name,
                'ECO_Code': eco_code,
                'Result': result_val,
                'Source': source_name,
            }
            out_row.update(one_hot)

            rows.append(out_row)

            # Escribir cada 1000 filas para no acumular demasiada memoria
            if len(rows) >= 1000:
                df_chunk = pd.DataFrame(rows)
                write_output_chunk(df_chunk, output_path, first_chunk)
                first_chunk = False
                rows = []

    # Escribir las filas restantes
    if rows:
        df_chunk = pd.DataFrame(rows)
        write_output_chunk(df_chunk, output_path, first_chunk)

    print(f"[PGN] Procesamiento completado. Archivo de salida: {output_path}")


# ----------------------------------------------------------------------
# Punto de entrada principal
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extrae características de partidas de ajedrez (CSV o PGN) "
                    "y genera un dataset con representación one‑hot en el ply 20."
    )
    parser.add_argument('--input', required=True,
                        help='Ruta al archivo de entrada (.csv o .pgn)')
    parser.add_argument('--output', required=True,
                        help='Ruta al archivo de salida (.csv)')
    parser.add_argument('--source', required=True,
                        help='Nombre de la fuente de datos (ej: kaggle, pros)')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    source_name = args.source

    # Detectar formato por extensión
    _, ext = os.path.splitext(input_path)
    ext = ext.lower()

    if ext == '.csv':
        print(f"Detectado formato CSV: {input_path}")
        process_csv(input_path, output_path, source_name)
    elif ext == '.pgn':
        print(f"Detectado formato PGN: {input_path}")
        process_pgn(input_path, output_path, source_name)
    else:
        print(f"Error: extensión '{ext}' no soportada. Use .csv o .pgn.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
