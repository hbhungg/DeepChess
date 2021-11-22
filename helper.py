import torch
from tqdm import tqdm

def train(model, train_data, epochs):
  loss_f = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  for epoch in range(epochs):
    total = 0
    for board1, board2, result in (t := tqdm(train_data)):
      predict = model(board1, board2)
      accuracy = sum(1 for x in (torch.argmax(result, 1) == torch.argmax(predict, 1)) if x)/result.shape[0]
      optimizer.zero_grad()
      loss = loss_f(predict.type(torch.float), result.type(torch.float))
      loss.backward()
      optimizer.step()
      
      t.set_description("Epoch: {} | Loss: {} | Accuracy: {}".format(epoch, loss, accuracy))
