#!/usr/bin/env python3

import chess
import chess.pgn
import numpy as np
import random


class Data_manager:
  def __init__(self, pgn_file: str):
    self.pgn_file = pgn_file
    self.pgn = open(self.pgn_file)

  def read_games(self, num_games: int = 1000):
    games = []
    black_count = 0
    white_count = 0
    for i in range(num_games):
      game = chess.pgn.read_game(self.pgn)
      if game is None:
        break
      # Only using game that result in a win
      result = game.headers["Result"]
      if result != "1/2-1/2":
        board = game.board()
        for idx, move in enumerate(game.mainline_moves()):
          # Skip the first 5 moves
          if idx < 5:
            board.push(move)
            continue
          # Only take move that is not a capture, take randomly 10% of the moves from a game
          if board.is_capture(move) is False and random.random() > 0.9:
            state = self.board_state(board)
            if result == "1-0":
              path = "dataset/npy/game_states/white/white"
              count = white_count
              white_count += 1
            elif result == "0-1":
              path = "dataset/npy/game_states/black/black"
              count = black_count
              black_count += 1
            with open(path+str(count)+'.npy', "wb") as f:
              np.save(f, state)
          board.push(move) 

  @staticmethod
  def board_state(board):
    black, white = board.occupied_co

    bitboards = np.array([
      black & board.pawns,
      black & board.knights,
      black & board.bishops,
      black & board.rooks,
      black & board.queens,
      black & board.kings,
      white & board.pawns,
      white & board.knights,
      white & board.bishops,
      white & board.rooks,
      white & board.queens,
      white & board.kings
    ], dtype=np.uint64)

    # Whos turn
    turn = board.turn
    # Castling rights
    castling_rights = board.castling_rights
    bck = Data_manager.get_bit(castling_rights, 1)
    bcq = Data_manager.get_bit(castling_rights, 8)
    wcq = Data_manager.get_bit(castling_rights, 57)
    wck = Data_manager.get_bit(castling_rights, 64)
    extra = np.array([turn, wck, wcq, bcq, bck])
    retval = np.concatenate([Data_manager.bitboards_to_array(bitboards), extra])
    return retval

  @staticmethod
  def bitboards_to_array(bb: np.ndarray) -> np.ndarray:
    bb = np.asarray(bb, dtype=np.uint64)[:, np.newaxis]
    s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
    b = (bb >> s).astype(np.uint8)
    b = np.unpackbits(b, bitorder="little")
    return b.reshape(-1)

  @staticmethod
  def get_bit(n: int, k: int) -> int:
    return (n & (1 << (k-1))) >> (k-1)


if __name__ == "__main__":
  dm = Data_manager("dataset/CCRL-4040.[1293685].pgn")
  dm.read_games()
