import sqlite3
import contextlib

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
      if ret is None: raise IndexError
      return ret

  def __len__(self) -> int:
    with contextlib.closing(self.conn.cursor()) as cur:
      cur.execute("SELECT count(*) FROM boards;")
      return cur.fetchone()[0]
  
  def __del__(self):
    self.conn.close()