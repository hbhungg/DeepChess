import math
from model import Autoencoder
from tinygrad import nn, TinyJit
from tinygrad.tensor import Tensor
from dataloader import BoardDataset, Dataloader
import mlflow
from mlflow.data.dataset import Dataset
from tinygrad.nn.state import safe_save, get_state_dict
from tinygrad.nn.optim import Optimizer
from tqdm import tqdm

@TinyJit
@Tensor.train()
def train(opt:Optimizer, model:Autoencoder, x:Tensor):
  opt.zero_grad()
  y = model.forward(x)
  loss = (y.binary_crossentropy_logits(x, reduction="mean")).backward()
  opt.step()
  return loss

if __name__ == "__main__":
  # TRACKING_URI="http://mlflow.lab.home"
  TRACKING_URI="http://localhost:5000"
  mlflow.set_tracking_uri(TRACKING_URI)
  mlflow.set_experiment("/deepchess-autoencoder")

  DATASET_NAME = "dataset_100000"
  EPOCH = 10
  BATCH_SIZE = 256
  LR = 1e-3
  ADAM_ESP = 1e-8

  model = Autoencoder()
  train_ds = Dataloader(BoardDataset(f"./dataset/{DATASET_NAME}.db", True), batch_size=BATCH_SIZE, shuffle=True)
  val_ds = Dataloader(BoardDataset(f"./dataset/{DATASET_NAME}.db", False), batch_size=BATCH_SIZE, shuffle=False)
  opt = nn.optim.Adam(nn.state.get_parameters(model), lr=LR, eps=ADAM_ESP)

  with mlflow.start_run():
    run = mlflow.active_run()
    mlflow.log_params({"epoch": EPOCH, "batch_size": BATCH_SIZE, "lr": LR, "ADAM_ESP": ADAM_ESP})

    # Train loop
    for epoch in range(EPOCH):
      for idx, x in enumerate(tbar:=tqdm(train_ds, total=len(train_ds))):
        loss = train(opt, model, x).item()
        assert math.isnan(loss) is False
        if idx % 3 == 0:
          mlflow.log_metric(key="train_loss", value=loss, step=epoch*len(train_ds)+idx)
          tbar.set_description(f"Epoch: {epoch} - Loss: {loss}")

      # Validation loop
      running_loss = 0
      for x in tqdm(val_ds, total=len(val_ds)):
        y = model.forward(x)
        loss = (y.binary_crossentropy_logits(x, reduction="mean")).numpy()
        running_loss += loss
      val_mean_loss = running_loss/len(val_ds)
      print(f"Epoch: {epoch} - Val Loss: {val_mean_loss}")
      mlflow.log_metric(key="val_loss", value=val_mean_loss, step=epoch)

    state_dict = get_state_dict(model)
    PATH = f"/tmp/{run.info.run_id}.sft"
    safe_save(state_dict, PATH)
    mlflow.log_artifact(local_path=PATH, run_id=run.info.run_id)
