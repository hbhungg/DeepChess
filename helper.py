import torch
from tqdm import tqdm

def train_supervise(model, train_data, val_data, epochs, patient):
  loss_f = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  history = []
  curr_patient = patient
  for epoch in range(epochs):
    correct = 0
    total = 0
    for board1, board2, result in (t := tqdm(train_data)):
      predict = model(board1, board2)
      total += result.shape[0]
      correct += sum(1 for x in (torch.argmax(result, 1) == torch.argmax(predict, 1)) if x)
      accuracy = correct/total
      optimizer.zero_grad()
      loss = loss_f(predict.type(torch.float), result.type(torch.float))
      loss.backward()
      optimizer.step()
      
      t.set_description("Epoch: {} | Loss: {} | Accuracy: {}".format(epoch, loss, accuracy))
    val_loss, _ = validation_supervise(model, val_data)
    # Learning rate multiply by 0.99 after each epochs
    optimizer.param_groups[0]['lr'] *= 0.99

    # Early stopping with loss and patient epoch on validation result
    if len(history) > 0:
      if val_loss - history[-1] > 0:
        curr_patient -= 1      
        if curr_patient < 0:
          print("Early stopping")
          return history
      else:
        curr_patient = patient
    history.append(val_loss)
  return history

def validation_supervise(model, val_data):
  loss_f = torch.nn.BCELoss()
  correct = 0
  total = 0
  for board1, board2, result in (t:= tqdm(val_data)):
    predict = model(board1, board2)
    correct += sum(1 for x in (torch.argmax(result, 1) == torch.argmax(predict, 1)) if x)
    total += result.shape[0]
    accuracy = correct/total
    loss = loss_f(predict.type(torch.float), result.type(torch.float))
    t.set_description("Loss: {} | Accuracy: {}".format(loss, accuracy))
  return loss, accuracy



def train_autoencoder(model, train_data, val_data, epochs, patient):
  loss_f = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
  history = []
  curr_patient = patient
  for epoch in range(epochs):
    for states in (t := tqdm(train_data)):
      reconstructed = model(states)
      optimizer.zero_grad()
      loss = loss_f(reconstructed, states)
      loss.backward()
      optimizer.step()
      t.set_description("Epoch: {} | Loss: {}".format(epoch, loss))
    val_loss = validation_autoencoder(model, val_data)

    # Early stopping with loss and patient epoch on validation result
    if len(history) > 0:
      if val_loss - history[-1] > 0:
        curr_patient -= 1
        if curr_patient < 0:
          print("Early stopping")
          return history
    history.append(val_loss)
  return history

def validation_autoencoder(model, val_data):
  loss_f = torch.nn.MSELoss()
  for states in (t := tqdm(val_data)):
    reconstructed = model(states)
    loss = loss_f(reconstructed, states) 
    t.set_description("Val loss: {}".format(loss))
  return loss
