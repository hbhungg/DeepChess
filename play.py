#!/usr/bin/env python3

from flask import Flask, render_template, request
import json
import chess
import random

app = Flask(__name__, template_folder=".")

@app.route("/")
def hello_world():
  ret = open("index.html").read()
  return ret.replace("start", board.fen())

@app.route("/move_coordinate", methods=["GET"])
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
        move = random.choice(list(board.legal_moves))
        board.push(move)
        print("\n ---Board--- \n{}\n\nStatus: {}\n".format(board, board.outcome()))
        response = app.response_class(response=board.fen(), status=200)
        return response

  print("GAME OVER")
  response = app.response_class(response="game over", status=200)
  return response

@app.route("/new_game")
def new_game():
  board.reset()
  return app.response_class(response=board.fen(), status=200)

if __name__ == "__main__":
  board = chess.Board()
  app.run(host="0.0.0.0", port="5001")
  
