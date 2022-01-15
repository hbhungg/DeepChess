#!/usr/bin/env python3
import chess
import chess.pgn
import numpy as np
import random


def read_games(pgn_file, save_path, num_games: int = 100):
  re = {"1-0": "{}white/{}-{}", "0-1": "{}black/{}-{}"}
  pgn = open(pgn_file)
  games = []
  results = []
  for i in range(num_games):
    game = chess.pgn.read_game(pgn)
    if game is None:
      break
    result = game.headers["Result"]
    # Only using game that result in a win
    if result in re:
      total = 0
      board = game.board()
      for idx, move in enumerate(game.mainline_moves()):
        is_capture = board.is_capture(move)
        board.push(move) 
        # Only take after 5 first moves that is not a capture, take randomly 10% of the moves
        if idx >= 5 and is_capture is False:
          total += 1
          state = get_bitboard(board)
          games.append(state)
          results.append(int(result[0]))
      print("Game: {}, result: {}, took {} board states".format(i, result, total))
  pgn.close()

  with open("dataset/games.npy", "wb") as f:
    np.save(f, np.array(games))
  with open("dataset/results.npy", "wb") as f:
    np.save(f, np.array(results))


def get_bitboard(board):
    '''
    params
    ------

    board : chess.pgn board object
        board to get state from

    returns
    -------

    bitboard representation of the state of the game
    64 * 6 + 5 dim binary numpy vector
    64 squares, 6 pieces, '1' indicates the piece is at a square
    5 extra dimensions for castling rights queenside/kingside and whose turn

    '''

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
  read_games("dataset/CCRL-4040.[1301281].pgn", "dataset/npy/", 8000)
