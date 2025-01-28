import math
from model import Autoencoder, DeepChess
from tinygrad import nn, TinyJit
from tinygrad.tensor import Tensor
from dataloader import BoardDataset, BoardPairDataset, Dataloader
import mlflow
from mlflow.data.dataset import Dataset
from tinygrad.nn.state import safe_save, get_state_dict
from tinygrad.nn.optim import Optimizer
from tqdm import tqdm
from pprint import pprint

@TinyJit
@Tensor.train()
def train(opt:Optimizer, model:DeepChess, x:Tensor, y:Tensor):
  opt.zero_grad()
  ret = model.forward(x)
  loss = (ret.binary_crossentropy(y, reduction="mean")).backward()
  opt.step()
  return loss


if __name__ == "__main__":
  DATASET_NAME = "dataset_100000"
  EPOCH = 5
  BATCH_SIZE = 512
  LR = 1e-3
  ADAM_ESP = 1e-8
  TRAINING_STEP = 100
  VAL_STEP = 10

  train_ds = Dataloader(BoardPairDataset(f"dataset/{DATASET_NAME}.db", True), BATCH_SIZE)
  val_ds = Dataloader(BoardPairDataset(f"dataset/{DATASET_NAME}.db", False, split=0.8), BATCH_SIZE)
  print(len(val_ds))

  autoencoder = Autoencoder()
  model = DeepChess(autoencoder)
  # pprint(nn.state.get_parameters(model))
  opt = nn.optim.Adam(nn.state.get_parameters(model), lr=LR, eps=ADAM_ESP)

  for epoch in range(EPOCH):
    for idx, (X, y) in enumerate(tbar:=tqdm(train_ds, total=TRAINING_STEP)):
      loss = train(opt, model, X, y).item()
      assert math.isnan(loss) is False
      if idx % 3 == 0:
        # mlflow.log_metric(key="train_loss", value=loss, step=epoch*len(train_ds)+idx)
        tbar.set_description(f"Epoch: {epoch} - Loss: {loss}")
      # This dataset is huge, just simulate epoch
      if idx == TRAINING_STEP:
        break
    # Validation loop
    running_loss = 0
    for idx, (X, y) in enumerate(tqdm(val_ds, total=VAL_STEP)):
      ret = model.forward(X)
      loss = (ret.binary_crossentropy_logits(y, reduction="mean")).numpy()
      running_loss += loss
      if idx == 10:
        break
    val_mean_loss = running_loss/VAL_STEP
    print(f"Epoch: {epoch} - Val Loss: {val_mean_loss}")
    # mlflow.log_metric(key="val_loss", value=val_mean_loss, step=epoch)