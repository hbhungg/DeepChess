import sqlite3
import contextlib
import numpy as np
import chess
from tinygrad import Tensor, dtypes
import random

def get_bitboard(fen):
  board = chess.Board(fen)
  SIZE = 64*6*2+5
  bitboard = np.zeros(SIZE)
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
  def __init__(self, db, train=True, split=0.8):
    with sqlite3.connect(db) as conn:
      with contextlib.closing(conn.cursor()) as cur:
        wlen, blen = [x[0] for x in cur.execute("SELECT count(*) FROM boards GROUP BY result ORDER by result DESC;").fetchall()]
        sw, ew = (0, int(wlen*split)) if train else (int(wlen*split), wlen)
        sb, eb = (0, int(blen*split)) if train else (int(blen*split), blen)
        self.w = [x[0] for x in cur.execute(f"SELECT fen FROM boards WHERE result = 1 AND id BETWEEN {sw} AND {ew};").fetchall()]
        self.b = [x[0] for x in cur.execute(f"SELECT fen FROM boards WHERE result = 0 AND id BETWEEN {sb} AND {eb};").fetchall()]

  def __len__(self) -> int: return len(self.w)*len(self.b)
  def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
    l = (idx.stop - idx.start) if isinstance(idx, slice) else 1
    w = Tensor.cat(*[get_bitboard(x) for x in random.sample(self.w, l)], dim=0)
    b = Tensor.cat(*[get_bitboard(x) for x in random.sample(self.b, l)], dim=0)
    return w, b

class BoardDataset:
  def __init__(self, db:str, train=True, split=0.8):
    with sqlite3.connect(db) as conn:
      with contextlib.closing(conn.cursor()) as cur:
        tlen = cur.execute("SELECT count(*) FROM boards;").fetchone()[0]
        s, e = (0, int(tlen*split)) if train else (int(tlen*split), tlen)
        self.data = cur.execute(f"SELECT fen, result FROM boards WHERE id BETWEEN {s} AND {e};").fetchall()

  def __len__(self) -> int: return len(self.data)
  def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
    boards = Tensor.cat(*[get_bitboard(x) for x, _ in self.data[idx]], dim=0)
    results = Tensor([float(y) for _, y in self.data[idx]])
    return boards, results

class Dataloader:
  def __init__(self, ds, batch_size): self.ds, self.batch_size = ds, batch_size
  def __iter__(self): return (self.ds[x:x+self.batch_size] for x in range(0, len(self)*self.batch_size, self.batch_size))
  def __len__(self): return len(self.ds)//self.batch_size
