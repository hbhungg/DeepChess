import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_supervise(model, train_data, val_data, epochs, patient, lr):
  loss_f = torch.nn.BCELoss(size_average=False)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  history = {"val_loss": [], "train_loss": [], "val_acc": [], "train_acc":[]}
  curr_patient = patient
  for epoch in range(epochs):
    correct = 0
    total = 0
    running_loss = 0
    for stack, result in (t := tqdm(train_data)):
      optimizer.zero_grad()

      predict = model(stack)
      total += result.shape[0]
      correct += sum(1 for x in (torch.argmax(result, 1) == torch.argmax(predict, 1)) if x)
      accuracy = correct/total

      loss = loss_f(predict.type(torch.float), result.type(torch.float))
      running_loss += loss.item()
      loss.backward()
      optimizer.step()
      
      t.set_description("Epoch: {} | Loss: {:3f} | Accuracy: {:3f}".format(epoch, loss.item()/len(stack), accuracy))
    val_loss, val_acc = validation_supervise(model, val_data)
    # Learning rate multiply by 0.99 after each epochs
    # optimizer.param_groups[0]['lr'] *= 0.99

    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["train_loss"].append(running_loss/len(train_data.dataset))
    history["train_acc"].append(accuracy)

    # Early stopping with on validation result
    if len(history["val_acc"]) > 1:
      if val_acc < history["val_acc"][-2]:
        curr_patient -= 1      
        if curr_patient < 0:
          print("Early stopping")
          return history
      else:
        curr_patient = patient
  return history

def validation_supervise(model, val_data):
  loss_f = torch.nn.BCELoss(size_average=False)
  correct = 0
  total = 0
  running_loss = 0
  for stack, result in (t:= tqdm(val_data)):
    predict = model(stack)
    correct += sum(1 for x in (torch.argmax(result, 1) == torch.argmax(predict, 1)) if x)
    total += result.shape[0]
    accuracy = correct/total
    loss = loss_f(predict.type(torch.float), result.type(torch.float))
    running_loss += loss.item()
    t.set_description("Loss: {:3f} | Accuracy: {:3f}".format(loss/len(stack), accuracy))
  return running_loss, accuracy



def train_autoencoder(model, train_data, val_data, epochs, patient, lr):
  loss_f = torch.nn.BCEWithLogitsLoss(size_average=False)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  history = {"val_loss": [], "train_loss": []}
  curr_patient = patient
  for epoch in range(epochs):
    running_loss = 0
    for states in (t := tqdm(train_data)):
      optimizer.zero_grad()

      reconstructed = model(states)
      loss = loss_f(reconstructed, states)
      loss.backward()
      running_loss += loss.item()
      optimizer.step()
      t.set_description("Epoch: {} | Loss: {:.3f}".format(epoch, loss.item()/len(states)))
    val_loss = validation_autoencoder(model, val_data)
    optimizer.param_groups[0]['lr'] *= 0.98

    # Save training data
    history["train_loss"].append(running_loss/len(train_data.dataset))
    history["val_loss"].append(val_loss)

    # Early stopping with loss and patient epoch on validation result
    if len(history["val_loss"]) > 1:
      if val_loss - history["val_loss"][-2] > 0:
        curr_patient -= 1
        if curr_patient < 0:
          print("Early stopping")
          return history
  return history

def validation_autoencoder(model, val_data):
  loss_f = torch.nn.BCEWithLogitsLoss(size_average=False)
  running_loss = 0
  for states in (t := tqdm(val_data)):
    reconstructed = model(states)
    loss = loss_f(reconstructed, states) 
    running_loss += loss.item()
    t.set_description("Loss: {:3f}".format(loss/len(states)))
  return running_loss/len(val_data.dataset)

def plot(history):
  # Plot loss
  plt.plot(history["train_loss"])
  plt.plot(history["val_loss"])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'valuation'], loc='upper left')
  plt.show()

  # Plot accuracy (if history has)
  if "train_acc" in history and "val_acc" in history:
    plt.plot(history["train_acc"])
    plt.plot(history["val_acc"])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

