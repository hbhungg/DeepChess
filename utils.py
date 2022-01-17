import matplotlib.pyplot as plt


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
