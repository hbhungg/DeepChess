import sqlite3
import contextlib
import numpy as np
import chess
from tinygrad import Tensor, dtypes

def get_bitboard(fen):
  board = chess.Board(fen)
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

  return Tensor(bitboard, requires_grad=False, dtype=dtypes.float32).expand(1, 773)



class BoardPairDataset:
  def __init__(self, db):
    self.conn: sqlite3.Connection = sqlite3.connect(db)

  def __len__(self) -> int:
    with contextlib.closing(self.conn.cursor()) as cur:
      w = cur.execute("SELECT count(*) FROM boards WHERE result = 1;").fetchone()[0]
      b = cur.execute("SELECT count(*) FROM boards WHERE result = 0;").fetchone()[0]
      return w*b
  
  def __getitem__(self, idx):
    return 0

  def __del__(self):
    self.conn.close()

class BoardDataset:
  def __init__(self, db):
    self.conn: sqlite3.Connection = sqlite3.connect(db)

  def __getitem__(self, idx):
    with contextlib.closing(self.conn.cursor()) as cur:
      # Support slice and single index
      cond = f"id BETWEEN {idx.start} AND {idx.stop}" if isinstance(idx, slice) else f"id = {idx}"
      ret = cur.execute(f"SELECT fen, result FROM boards WHERE {cond};").fetchall()
      boards = Tensor.cat(*[get_bitboard(x) for x, _ in ret], dim=0)
      results = Tensor([float(y) for _, y in ret])
      if ret is None: raise IndexError
      return boards, results

  def __len__(self) -> int:
    with contextlib.closing(self.conn.cursor()) as cur:
      cur.execute("SELECT count(*) FROM boards;")
      return cur.fetchone()[0]
  
  def __del__(self):
    self.conn.close()
