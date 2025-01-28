import random

from tqdm import tqdm
import chess.pgn
from dataloader import get_bitboard


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
          data.append((board.fen(), re[result], sqlite3.Binary(get_bitboard(board.fen()).data())))
          c += 1
        if c == max_board_per_game:
          break
      if len(data) > 1000:
        cur= con.cursor()
        cur.execute('BEGIN TRANSACTION;')
        cur.executemany('INSERT INTO boards (fen, result, bitboard) VALUES (?, ?, ?)', data)
        con.commit()
        data = []


if __name__ == "__main__":
  import sqlite3

  num_games = 100_000
  con = sqlite3.connect(f"dataset/b_dataset_{num_games}.db")
  cur = con.cursor()
  cur.execute("""
              CREATE TABLE IF NOT EXISTS boards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fen TEXT, 
                bitboard BLOB,
                result BOOLEAN
              );
              """)
  con.commit()

  read_games("dataset/CCRL-4040.[1301281].pgn", con, num_games, 10)
  # Add index after adding data (faster)
  cur.execute("CREATE INDEX IF NOT EXISTS idx_boards_result ON boards (result);")
  con.close()
