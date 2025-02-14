import sqlite3
import contextlib
import numpy as np
import chess
from tinygrad import Tensor, dtypes
from tinygrad.helpers import prod
import random
from line_profiler import profile
import time

# @TinyJit
@profile
# def get_bitboard(fen:str) -> Tensor:
def get_bitboard(fen:str) -> np.ndarray:
  board = chess.Board(fen)
  # 8x8 board, each square has 6 state (for each piece), mul by 2 for color, and last 5 for misc data about game state
  SIZE = 8*8*6*2+5
  bitboard = np.zeros(SIZE, dtype=np.float32)
  piece_idx = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}

  for i in range(64):
    if p:=board.piece_at(i):
      color = int(p.color) + 1
      bitboard[(piece_idx[p.symbol().lower()] + i * 6) * color] = 1.0
  bitboard[-5:] = [
    float(board.has_queenside_castling_rights(False)),
    float(board.has_queenside_castling_rights(True)),
    float(board.has_kingside_castling_rights(False)),
    float(board.has_kingside_castling_rights(True)),
    float(board.turn)
  ]
  return bitboard
  # return Tensor(bitboard, requires_grad=False, dtype=dtypes.float32, device='CPU').expand(1, 773)

class Dataset:
  def collate_fn(self, batch):
    return Tensor.cat(*[x for x in batch], dim=0)

class BoardPairDataset(Dataset):
  def __init__(self, db, train=True, split=0.8):
    with sqlite3.connect(db) as conn:
      with contextlib.closing(conn.cursor()) as cur:
        wlen, blen = [x[0] for x in cur.execute("SELECT count(*) FROM boards GROUP BY result ORDER by result DESC;").fetchall()]
        sw, ew = (0, int(wlen*split)) if train else (int(wlen*split), wlen)
        sb, eb = (0, int(blen*split)) if train else (int(blen*split), blen)
        self.w = [x[0] for x in cur.execute(f"SELECT fen FROM boards WHERE result = 1 AND id BETWEEN {sw} AND {ew};").fetchall()]
        self.b = [x[0] for x in cur.execute(f"SELECT fen FROM boards WHERE result = 0 AND id BETWEEN {sb} AND {eb};").fetchall()]

  @profile
  def __getitem__(self, _) -> tuple[Tensor, Tensor]:
    x, y = get_bitboard(random.choice(self.w)), get_bitboard(random.choice(self.b))
    b, r = Tensor.cat(x, y), Tensor([1, 0]) if random.choice([True, False]) else Tensor.cat(y, x), Tensor([0, 1])
    return b.expand(1, -1, -1), r.expand(1, -1)
  
  def getitem(self, _) -> tuple[np.ndarray, np.ndarray]:
    x, y = get_bitboard(random.choice(self.w)), get_bitboard(random.choice(self.b))
    b, r = (np.stack([x, y]), np.array([1, 0])) if random.choice([True, False]) else (np.stack([y, x]), np.array([0, 1]))
    return b, r

  def collate_fn(self, batch):
    return Tensor.cat(*[x[0] for x in batch], dim=0), Tensor.cat(*[x[1] for x in batch], dim=0)
  
  def collate_fn_np(self, batch):
    return np.stack([x[0] for x in batch]), np.stack([x[1] for x in batch])

  def __len__(self) -> int: return len(self.w)*len(self.b)


class BoardDataset(Dataset):
  def __init__(self, db:str, train=True, split=0.8):
    with sqlite3.connect(db) as conn:
      with contextlib.closing(conn.cursor()) as cur:
        tlen = cur.execute("SELECT count(*) FROM boards;").fetchone()[0]
        s, e = (0, int(tlen*split)) if train else (int(tlen*split), tlen)
        self.data = [x[0] for x in cur.execute(f"SELECT fen FROM boards WHERE id BETWEEN {s} AND {e};").fetchall()]

  def __getitem__(self, idx) -> list[Tensor] | Tensor:
    return [get_bitboard(x) for x in self.data[idx]] if isinstance(idx, slice) else get_bitboard(self.data[idx])

  def __len__(self) -> int: return len(self.data)

import multiprocessing
from multiprocessing import shared_memory

class Dataloader:
  def __init__(self, ds:Dataset, batch_size:int, shuffle=False): 
    self.ds, self.batch_size = ds, batch_size
    self.shuffle = shuffle
    self.q = multiprocessing.Queue(maxsize=10)

  @staticmethod
  def worker(q: multiprocessing.Queue, ds:Dataset, batch_size:int, X:Tensor, y:Tensor):
    for i in range(0, len(ds), batch_size):
      batch = [ds.getitem(j) for j in range(i, i+batch_size)]
      b, r = ds.collate_fn_np(batch) 
      X[i].contiguous().realize().lazydata.base.realized.as_buffer(force_zero_copy=True)[:] = b.tobytes()
      y[i].contiguous().realize().lazydata.base.realized.as_buffer(force_zero_copy=True)[:] = r.tobytes()
      q.put(i)


  @profile
  def __iter__(self):
    import os
    if os.path.exists("/dev/shm/resnet_X"): os.unlink("/dev/shm/resnet_X")
    xsz = (self.batch_size, 2, 773)
    ysz = (self.batch_size, 2)
    xshm = shared_memory.SharedMemory(name="X", create=True, size=prod(xsz))
    yshm = shared_memory.SharedMemory(name="y", create=True, size=prod(ysz))
    X = Tensor.empty(*xsz, dtype=dtypes.float32, device=f"disk:/dev/shm/X")
    y = Tensor.empty(*ysz, dtype=dtypes.float32, device=f"disk:/dev/shm/y")
    p = multiprocessing.Process(target=self.worker, args=(self.q, self.ds, self.batch_size, X, y))
    p.daemon = True
    p.start()
    for i in range(0, len(self.ds), self.batch_size):
      try:
        idx = self.q.get()
        yield X[idx], y[idx]
      finally:
        xshm.close()
        yshm.close()
        try:
          xshm.unlink()
          yshm.unlink()
        except: pass
  def __len__(self): return len(self.ds)//self.batch_size

if __name__ == "__main__":
  a = BoardPairDataset("dataset/dataset_100000.db")
  b = BoardDataset("dataset/dataset_100000.db")
  dl1 = Dataloader(a, 256)
  # dl1 = Dataloader(a, 256)
  dl1 = iter(dl1)
  # dl2 = iter(Dataloader(b, 100))


  # @profile
  def k():
    idx = 0
    while True:
      i, v = next(dl1)
      # print(i.realize(), v.realize())
      print(i.shape, v.shape)
      time.sleep(0.5)
      if idx == 1:
        break
      idx += 1

  k()
  # for idx, i in enumerate(dl2):
  #   print(i)
  #   if idx == 10:
  #     break
