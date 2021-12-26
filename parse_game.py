#!/usr/bin/env python3
import chess
import chess.pgn
import numpy as np
import random


def read_games(pgn_file, save_path, num_games: int = 100):
  re = {"1-0": "{}white/{}-{}", "0-1": "{}black/{}-{}"}
  pgn = open(pgn_file)
  for i in range(num_games):
    total = 0
    game = chess.pgn.read_game(pgn)
    if game is None:
      break
    result = game.headers["Result"]
    # Only using game that result in a win
    if result in re:
      board = game.board()
      for idx, move in enumerate(game.mainline_moves()):
        is_capture = board.is_capture(move)
        board.push(move) 
        # Only take after 5 first moves that is not a capture, take randomly 10% of the moves
        if idx >= 5 and is_capture is False and random.random() > 0.9:
          total += 1
          state = board_state(board)
          with open(re[result].format(save_path, i, idx), "wb") as f:
            np.save(f, state)
    print("Game: {}, took {} board states".format(i, total))
  pgn.close()


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
  bck = get_bit(castling_rights, 1)
  bcq = get_bit(castling_rights, 8)
  wcq = get_bit(castling_rights, 57)
  wck = get_bit(castling_rights, 64)
  extra = np.array([turn, wck, wcq, bcq, bck])
  retval = np.concatenate([bitboards_to_array(bitboards), extra])
  return retval


def bitboards_to_array(bb: np.ndarray) -> np.ndarray:
  bb = np.asarray(bb, dtype=np.uint64)[:, np.newaxis]
  s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
  b = (bb >> s).astype(np.uint8)
  b = np.unpackbits(b, bitorder="little")
  return b.reshape(-1)


def get_bit(n: int, k: int) -> int:
  return (n & (1 << (k-1))) >> (k-1)


if __name__ == "__main__":
  read_games("dataset/CCRL-4040.[1301281].pgn", "dataset/npy/", 500)
