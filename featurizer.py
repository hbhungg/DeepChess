#!/usr/bin/env python3
import torch
import numpy as np
from torch.utils.data import DataLoader
from model.autoencoder import Autoencoder


class BitboardDataset:
  def __init__(self, games):
    self.games = games
    self.length = len(self.games)

  def __getitem__(self, idx):
    return torch.from_numpy(self.games[idx]).type(torch.FloatTensor)

  def __len__(self):
    return self.length
    
def featurize(game, model):
  _, enc = model(game)
  return enc.detach().numpy()

if __name__ == "__main__":
  games = np.load("./dataset/bitboards.npy", mmap_mode="c")
  bd = BitboardDataset(games)
  bdloader = DataLoader(bd, batch_size=128)

  model = Autoencoder()
  c = torch.load("./checkpoints/autoencoder/lr_0.005_decay_0.95.pt")
  model.load_state_dict(c["model_state_dict"])
  model.eval()

  featurized = []
  for idx, game in enumerate(bdloader):
    featurized.append(featurize(game, model))
    if idx % 1000 == 0:
      print("Featurizing board {}/{}".format(idx, len(bdloader)))

  featurized = np.vstack(featurized)

  with open("./dataset/features.npy", "wb") as f:
    np.save(f, featurized)
