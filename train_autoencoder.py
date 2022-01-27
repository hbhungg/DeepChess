import torch
from torch.utils.data import DataLoader
import numpy as np

from model.autoencoder import Autoencoder
from chess_dataset import ChessDataset


def train(model, train_data, val_data, epochs, loss_f, patient, lr, decay, save_path):
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  history = {"val_loss": [], "train_loss": []}
  curr_patient = patient

  # Train loop
  for epoch in range(epochs):
    running_loss = 0
    model.train()
    for batch_idx, states in enumerate(train_data):
      optimizer.zero_grad()

      reconstructed, _ = model(states)
      loss = loss_f(reconstructed, states)
      loss.backward()
      optimizer.step()

      # Calculate training statistics
      running_loss += loss.item()
      # Print every 1000
      if batch_idx % 100 == 0:
        print("Training epoch {}: {}/{}\t{:.0%}\tLoss: {:.3f}".format(epoch, 
                batch_idx, 
                len(train_data), 
                batch_idx/len(train_data), 
                loss.item()/len(states)))

    val_loss = validation(model, val_data, loss_f)
    # Decay 
    for p in optimizer.param_groups:
      p["lr"] *= decay

    # Save training data
    history["train_loss"].append(running_loss/len(train_data.dataset))
    history["val_loss"].append(val_loss)

    # Save model every epochs
    save(model, save_path, lr, decay, epoch)

    # Early stopping with loss and patient epoch on validation result
    if len(history["val_loss"]) > 1:
      if val_loss - history["val_loss"][-2] > 0:
        curr_patient -= 1
        if curr_patient < 0:
          print("Early stopping")
          return history
  return history


def validation(model, val_data, loss_f):
  running_loss = 0
  model.eval()
  for states in val_data:
    reconstructed, _ = model(states)
    loss = loss_f(reconstructed, states) 
    running_loss += loss.item()
  print("====> Test loss: {:.3f}".format(running_loss/len(val_data.dataset)))
  return running_loss/len(val_data.dataset)


def save(model, path, lr, decay, epoch):
  p = "{}/lr_{}_decay_{}.pt".format(path, lr, decay)
  torch.save({
              "epoch": epoch,
              "model_state_dict": model.state_dict()
             }, p)
  print("Model saved on {}".format(p))



if __name__ == "__main__":
  # Model
  model = Autoencoder() 

  # Dataset
  white_wins = np.load("./dataset/white/white_wins.npy", mmap_mode="c")
  black_wins = np.load("./dataset/black/black_wins.npy", mmap_mode="c")

  train_data = ChessDataset(white_wins, black_wins, train_split=0.8)
  test_data = ChessDataset(white_wins, black_wins, train=False, train_split=0.8)

  print(len(train_data), len(test_data))
  trainloader = DataLoader(train_data, batch_size=128, shuffle=True) 
  testloader = DataLoader(test_data, batch_size=128, shuffle=False) 

  # Train
  print("Start training")
  epochs = 1
  lr = 5e-4
  decay = 0.95
  save_path="./checkpoints/autoencoder"
  loss_f = torch.nn.BCELoss(size_average=False)
  resume = False
  if resume:
    c = torch.load("./checkpoints/autoencoder/lr_0.005_decay_0.95.pt")
    model.load_state_dict(c["model_state_dict"])
  his = train(model=model,
              train_data=trainloader,
              val_data=testloader, 
              epochs=epochs,
              patient=2,
              loss_f=loss_f,
              lr=lr,
              decay=decay,
              save_path=save_path)

  from utils import plot
  plot(his)
