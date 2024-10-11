from dataloader import BoardDataset, BoardPairDataset

if __name__ == "__main__":
  a = BoardDataset("dataset/dataset_10000.db")
  print(len(a))
  print(a[100])
  print(a[100:120])

  b = BoardPairDataset("dataset/dataset_10000.db")
  print(len(b))
  print(b[100])
  print(b[100:120])
