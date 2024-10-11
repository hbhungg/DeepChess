import random

from tqdm import tqdm
import chess.pgn
import numpy as np


def read_games(pgn_file, con, num_games: int = 100, max_board_per_game=5):
  re = {"1-0": True, "0-1": False}
  with open(pgn_file, 'r', buffering=1024 * 1024) as pgn:
    data = []
    for i in tqdm(range(num_games)):
      game = chess.pgn.read_game(pgn)

      if game is None:
        break

      # Only using game is not draw
      result = game.headers["Result"]
      if result not in re:
        continue

      board = game.board()
      c = 0
      for idx, move in enumerate(game.mainline_moves()):
        is_capture = board.is_capture(move)
        board.push(move)
        # Only take after 5 first moves that is not a capture
        # 10% chance of taking the move
        if idx >= 5 and (is_capture is False) and random.random() <= 0.1:
          data.append((board.fen(), re[result]))
          c += 1
        if c == max_board_per_game:
          break
      if len(data) > 1000:
        cur= con.cursor()
        cur.execute('BEGIN TRANSACTION;')
        cur.executemany('INSERT INTO boards (fen, result) VALUES (?, ?)', data)
        con.commit()
        data = []


def get_bitboard(board):
  bitboard = np.zeros(64*6*2+5)

  piece_idx = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}

  for i in range(64):
    if board.piece_at(i):
      color = int(board.piece_at(i).color) + 1
      bitboard[(piece_idx[board.piece_at(i).symbol().lower()] + i * 6) * color] = 1

  bitboard[-1] = int(board.turn)
  bitboard[-2] = int(board.has_kingside_castling_rights(True))
  bitboard[-3] = int(board.has_kingside_castling_rights(False))
  bitboard[-4] = int(board.has_queenside_castling_rights(True))
  bitboard[-5] = int(board.has_queenside_castling_rights(False))

  return bitboard


if __name__ == "__main__":
  import sqlite3

  num_games = 10_000
  con = sqlite3.connect(f"dataset/dataset_{num_games}.db")
  cur = con.cursor()
  cur.execute("""CREATE TABLE IF NOT EXISTS boards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fen TEXT, 
                result BOOLEAN
              );""")
  con.commit()

  read_games("dataset/CCRL-4040.[1301281].pgn", con, num_games, 10)
  con.close()
