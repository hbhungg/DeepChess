#!/usr/bin/env python3

from flask import Flask, render_template, request
import json

app = Flask(__name__, template_folder=".")

@app.route("/")
def hello_world():
  ret = open("index.html").read()
  return ret

@app.route("/move_coordinate", methods=["POST"])
def get_post_js_data():
  piece = request.args.get('piece', default='')
  position = request.args.get("position")
  source = request.args.get("source")
  print(piece, position, source)
  return app.response_class()

if __name__ == "__main__":
  app.run(debug=True)
  
