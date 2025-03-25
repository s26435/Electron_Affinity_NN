import torch

batch_size = 256

SAVE_DIC_LOC = "src/dict.json"
SAVE_SCALER_LOC = "src/scaler.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 1000