import numpy as np

import torch
from torchvision import datasets


class ChessPairDataset(torch.utils.data.Dataset):
  def __init__(self, features, wins, train=True, train_split=0.8, length=1000000) :

    #TODO: We dont need this right?
    p = np.random.permutation(len(wins))
    features = features[p]
    wins = wins[p]

    if train is True:
      features = features[:int(len(features)*train_split)]
      wins = wins[:int(len(wins)*train_split)]
      self.length = int(length * train_split)
    else:
      features = features[int(len(features)*train_split):]
      wins = wins[int(len(wins)*train_split):]
      self.length = length - int(length*train_split)
    
    self.game_wins = features[wins == 1]
    self.game_losses = features[wins == -1]

  #TODO: Might change this so that it is guarantee that sample will not be repeat each call
  def __getitem__(self, index):
    rand_win = self.game_wins[
      np.random.randint(0, self.game_wins.shape[0])]
    rand_loss = self.game_losses[
      np.random.randint(0, self.game_losses.shape[0])]

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
  def __init__(self, games, train=True, train_split=0.8):
    if train_split < 0 or train_split > 1:
      raise ValueError("Split must be between 0 and 1")
    
    self.games = games

    # Split train and test set
    if train is not True:
      self.games = self.games[int(len(self.games) * train_split):]
      self.p = list(range(0, len(self.games)))
    else:
      self.games = self.games[:int(len(self.games) * train_split)]
      self.p = list(range(0, len(self.games)))

    np.random.shuffle(self.p)
    self.length = len(self.games)

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    return torch.from_numpy(self.games[self.p[idx]]).type(torch.FloatTensor)
