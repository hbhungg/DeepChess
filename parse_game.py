#!/usr/bin/env python3
import chess
import chess.pgn
import numpy as np
import random


def read_games(pgn_file, save_path, num_games: int = 100):
  games = []
  results = []
  re = {"1-0": 1, "0-1": 0}

  pgn = open(pgn_file)
  for i in range(num_games):
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
        # Only take after 5 first moves that is not a capture
        # And take randomly
        if idx >= 5 and is_capture is False and random.random() > 0.9:
          state = get_bitboard(board)
          games.append(state)
          results.append(re[result])

    # Print every 100 games
    if i % 1000 == 0:
      print("Game: {}/{} took {} board states".format(i, num_games, len(games)))

  with open("{}/bitboards.npy".format(save_path), "wb") as b, \
       open("{}/results.npy".format(save_path), "wb") as r:
    np.save(b, np.array(games))
    np.save(r, np.array(results))


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
  read_games("dataset/CCRL-4040.[1301281].pgn", "./dataset", 500000)
