#!/usr/bin/env python3

from flask import Flask, render_template, request
import json
import chess
import time

import torch
import numpy as np
from model.autoencoder import Autoencoder
from model.siamese import Siamese
from parse_game import get_bitboard

app = Flask(__name__, template_folder=".")

@app.route("/")
def hello_world():
  ret = open("index.html").read()
  return ret.replace("start", board.fen())

@app.route("/move_coordinate")
def move_coordinates():
  source = request.args.get('source', default='')
  piece = request.args.get('piece', default='')
  target = request.args.get('target', default='')

  if not board.is_game_over():
    # Human move
    move = chess.Move.from_uci(source+target)
    if move in board.legal_moves:
      board.push(move)
      if not board.is_game_over():
        # Computer move
        print("Calculating move...")
        start = time.time()
        og_board = board.copy(stack=False)
        move = alphabeta(board, 4, -100, 100, False, og_board)[1]
        end = time.time()
        board.push(move)
        print("Computer moves {} in {:.2f}s".format(move, end-start))
        print("\n ---Board--- \n{}\n\nStatus: {}\n".format(board, board.outcome()))
        response = app.response_class(response=board.fen(), status=200)
    response = app.response_class(response=board.fen(), status=200)
  else:
    print("GAME OVER")
    response = app.response_class(response="game over", status=200)
  return response

def compare_board(board1, board2):
  bboard1 = torch.from_numpy(np.expand_dims(get_bitboard(board1), axis=0)).type(torch.FloatTensor)
  bboard2 = torch.from_numpy(np.expand_dims(get_bitboard(board2), axis=0)).type(torch.FloatTensor)
  _, f1 = ae(bboard1)
  _, f2 = ae(bboard2)
  f = torch.cat((f1, f2), 1)
  return si(f)[0]


def minimax(board, depth, white, orig_board):
  if depth == 0:
    return compare_board(board, orig_board)[0], None
  if white:
    best_value = -100
    for move in board.legal_moves:
      board.push(move)
      v, _ = minimax(board, depth-1, False, orig_board)
      if v >= best_value:
        best_value = v
        best_move = move
      board.pop()
    return best_value, best_move
  else:
    best_value = 100
    for move in board.legal_moves:
      board.push(move)
      v, _ = minimax(board, depth-1, True, orig_board)
      if v <= best_value:
        best_value = v
        best_move = move
      board.pop()
    return best_value, best_move


def alphabeta(board, depth, alpha, beta, white, orig_board):
  if depth == 0:
    return compare_board(board, orig_board)[0], None
  if white:
    v = -100 # very (relatively) small number
    best_move = None
    for move in board.legal_moves:
      board.push(move)
      candidate_v, _ = alphabeta(board, depth - 1, alpha, beta, False, orig_board)
      board.pop()
      if candidate_v >= v:
        v = candidate_v
        best_move = move
      else:
        pass
      alpha = max(alpha, v)
      if beta <= alpha:
        break
    return v, best_move
  else:
    v = 100 # very (relatively) large number
    best_move = None
    for move in board.legal_moves:
      board.push(move)
      candidate_v, _ = alphabeta(board, depth - 1, alpha, beta, True, orig_board)
      board.pop()
      if candidate_v <= v:
        v = candidate_v
        best_move = move
      else:
        pass
      beta = min(beta, v)
      if beta <= alpha:
        break
    return v, best_move


@app.route("/new_game")
def new_game():
  board.reset()
  return app.response_class(response=board.fen(), status=200)

if __name__ == "__main__":
  board = chess.Board()

  # Load Autoencoder
  ae = Autoencoder()
  c = torch.load("./checkpoints/ae_10.pth.tar")
  ae.load_state_dict(c["state_dict"])
  ae.eval()

  # Load Siamese
  si = Siamese()
  d = torch.load("./checkpoints/siamese/lr_0.001_decay_0.99.pth")
  si.load_state_dict(d["model_state_dict"])
  si.eval()

  app.run(host="0.0.0.0", port="5001", debug=True)

