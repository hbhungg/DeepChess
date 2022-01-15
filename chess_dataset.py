import os 
import itertools
import numpy as np
import random

import torch
from torchvision import datasets

class ChessPairDataset(torch.utils.data.Dataset):
  def __init__(self, train=True, train_split=0.8, length=1000000) :
    games = np.load('./dataset/features.npy')
    wins = np.load('./dataset/results.npy')

    p = np.random.permutation(len(wins))
    games = games[p]
    wins = wins[p]

    if train is True:
      train_games = games[:int(len(games)*.8)]
      train_wins = wins[:int(len(games)*.8)]
      self.length = int(length * train_split)
    else:
      train_games = games[int(len(games)*.8):]
      train_wins = wins[int(len(games)*.8):]
      self.length = length - int(length * train_split)

    self.train_game_wins = train_games[train_wins == 1]
    self.train_game_losses = train_games[train_wins == 0]

  def __getitem__(self, index):
    rand_win = self.train_game_wins[
      np.random.randint(0, self.train_game_wins.shape[0])]
    rand_loss = self.train_game_losses[
      np.random.randint(0, self.train_game_losses.shape[0])]

    order = np.random.randint(0,2)
    if order == 0:
      stacked = np.hstack((rand_win, rand_loss))
      stacked = torch.from_numpy(stacked).type(torch.FloatTensor)
      label = torch.from_numpy(np.array([1, 0])).type(torch.FloatTensor)
      return (stacked, label)
    else:
      stacked = np.hstack((rand_loss, rand_win))
      stacked = torch.from_numpy(stacked).type(torch.FloatTensor)
      label = torch.from_numpy(np.array([0, 1])).type(torch.FloatTensor)
      return (stacked, label)

  def __len__(self):
      return self.length

class ChessDataset(torch.utils.data.Dataset):
  def __init__(self, path, train=True, train_split=0.8):
    if train_split < 0 or train_split > 1:
      raise ValueError("Split must be between 0 and 1")

    self.games = np.load(path)
    np,random.shuffle(self.games)

    # Split train and test set
    if train is not True:
      self.games = self.games[int(len(self.games) * train_split):]
    else:
      self.games = self.games[:int(len(self.games) * train_split)]
      
    self.length = len(self.games)

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    return torch.from_numpy(self.games[idx]).type(torch.FloatTensor)
