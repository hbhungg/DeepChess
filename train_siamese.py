import torch
from torch.utils.data import DataLoader
import numpy as np

from model.siamese import Siamese
from chess_dataset import ChessPairDataset


#TODO: Might merge this and the autoencoder training loop, since both of them are roughly similar
def train(model, train_data, val_data, epochs, loss_f, lr, decay, save_path, patient, delta):
  """
  Pytorch supervise training loop
  
  Parameters:
    train_data: Training data, iterable object (torch.utils.data.DataLoader)
    val_data:   Validtion data, iterable object (torch.utils.data.DataLoader)
    epochs:     Training epochs
    loss_f:     Torch loss functions (eg. torch.nn.BCELoss)
    lr:         Learning rate for optimizer
    decay:      Decay rate for optimizer
    save_path:  Model save path (will be save every epochs)
    patient:    How many non-improvement epochs to stop early
    delta:      How much improvement is need to be consider an improvement

  Returns:
    history:    A dict of training statistics
  """

  optimizer = torch.optim.Adam(model.parameters(), 
                              lr=lr)
  history = {"val_loss": [], "train_loss": [], "val_acc": [], "train_acc":[]}
  curr_patient = patient

  # Train loop
  for epoch in range(epochs):
    correct, total, running_loss = 0, 0, 0
    model.train()
    for batch_idx, (stack, result) in enumerate(train_data):
      optimizer.zero_grad()

      predict = model(stack)
      loss = loss_f(predict.type(torch.float), result.type(torch.float))
      loss.backward()
      optimizer.step()

      # Calculate training statistics
      running_loss += loss.item()
      total += result.shape[0]
      correct += sum(1 for x in (torch.argmax(result, 1) == torch.argmax(predict, 1)) if x)
      accuracy = correct/total

      if batch_idx % 500 == 0:
        print("Training epoch: {} {}/{}\t{:.0%}\tLoss: {:3f}  Accuracy: {:3f}".format(epoch,
                batch_idx, 
                len(train_data), 
                batch_idx/len(train_data),
                loss.item()/len(stack), 
                accuracy))

    # Validation
    val_loss, val_acc = validation(model, val_data, loss_f)

    # Decay 
    for p in optimizer.param_groups:
      p["lr"] *= decay

    # Save training statistics
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["train_loss"].append(running_loss/len(train_data.dataset))
    history["train_acc"].append(accuracy)

    # Save model every epoch
    save(model, save_path, lr, decay, epoch)

    # Early stopping on validation result
    if len(history["val_acc"]) > 1:
      if history["val_acc"][-2] - history["val_acc"][-1] > delta:
        print("Not improve with delta {}, {} more epochs".format(delta, curr_patient))
        curr_patient -= 1
        if curr_patient < 0:
          print("Early stopping")
          return history
      else:
        print("Improve with delta {}".format(delta))
        curr_patient = patient
  return history


def validation(model, val_data, loss_f):
  correct, total, running_loss = 0, 0, 0
  model.eval()
  for stack, result in val_data:
    predict = model(stack)

    # Calculate validation statistics
    correct += sum(1 for x in (torch.argmax(result, 1) == torch.argmax(predict, 1)) if x)
    total += result.shape[0]
    accuracy = correct/total
    loss = loss_f(predict.type(torch.float), result.type(torch.float))
    running_loss += loss.item()
  print("====> Test loss: {:3f}  Accuracy: {:3f}".format(running_loss/len(val_data.dataset), accuracy))
  return running_loss/len(val_data.dataset), accuracy


def save(model, path, lr, decay, epoch):
  p = "{}/lr_{}_decay_{}.pth".format(path, lr, decay)
  torch.save({
              "epoch": epoch,
              "model_state_dict": model.state_dict()
             }, p)
  print("Model saved on {}".format(p))


if __name__ == "__main__":
  # Model
  model = Siamese()

  # Dataset
  features = np.load("./dataset/features.npy", mmap_mode="r")
  wins = np.load("./dataset/results.npy", mmap_mode="r")
  train_data = ChessPairDataset(features, wins, length=1000000)
  test_data = ChessPairDataset(features, wins, train=False, length=1000000)
  trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
  testloader = DataLoader(test_data, batch_size=64, shuffle=False)

  # Train
  print("Start training")
  epochs = 1
  lr = 0.001
  decay = 0.99
  save_path = "./checkpoints/siamese"
  loss_f = torch.nn.BCELoss(size_average=False)
  resume = True
  if resume:
    c = torch.load("./checkpoints/siamese/lr_0.001_decay_0.99.pth")
    model.load_state_dict(c["model_state_dict"])
  his = train(model=model,
              train_data=trainloader,
              val_data=testloader,
              epochs=epochs,
              loss_f=loss_f,
              lr=lr,
              decay=decay,
              patient=1,
              delta=0.01,
              save_path=save_path)

  from utils import plot
  plot(his)
