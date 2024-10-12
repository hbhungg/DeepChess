from model import Autoencoder
from tinygrad import nn, TinyJit
from tinygrad.tensor import Tensor
from dataloader import BoardDataset


x = Tensor.rand(64, 773, requires_grad=False)

@TinyJit
@Tensor.train()
def train(opt, model, x:Tensor):
  opt.zero_grad()
  y = model.forward(x)
  loss = (y.binary_crossentropy(x, reduction="mean")).backward()
  opt.step()
  return loss

if __name__ == "__main__":

  model = Autoencoder()
  dataset = BoardDataset("./dataset/dataset_100000.db")
  opt = nn.optim.Adam(nn.state.get_parameters(model), lr=1e-4)

  EPOCH = 1
  BATCH_SIZE = 30
  train_l, test_l = [], []

  split_idx = int(len(dataset) * 0.8)

  for i in range(EPOCH): 
    # Train loop
    for x in range(0, split_idx, BATCH_SIZE):
      X = dataset[x:x+BATCH_SIZE-1][0]
      if X.shape[0] != BATCH_SIZE: continue # STOOPID
      loss = train(opt, model, X).item()
      train_l.append(loss)
      if x % 1000 == 0:
        print(loss)

    # Validation loop
    running_loss = 0
    for x in range(split_idx, len(dataset), BATCH_SIZE):
      X = dataset[x:x+BATCH_SIZE-1][0]
      if X.shape[0] != BATCH_SIZE: continue # STOOPID
      y = model.forward(X)
      loss = (y.binary_crossentropy(X, reduction="mean")).numpy()
      running_loss += loss
    print("Validation test: ", running_loss/(len(dataset)-split_idx))

  from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
  state_dict = get_state_dict(model)
  safe_save(state_dict, "autoencoder_100k.sft")

  import matplotlib.pyplot as plt

  plt.plot(train_l)
  plt.show()

