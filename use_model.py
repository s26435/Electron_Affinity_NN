import torch
import sys
import pandas as pd
import numpy as np

from descryptors import get_one
from utils import tokenize_formula
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model
    model = torch.load("src/model.pth", weights_only=False, map_location=device)
    # Convert model to float32
    model = model.float()
    model.eval()
    print("Załadowano model!")

    if len(sys.argv) > 1:
        formula = sys.argv[1]
    else:
        formula = input("Podaj formułę: ")

    df = get_one(formula).drop(columns=["formula"])
    X = torch.concat(
        [
            torch.tensor(tokenize_formula(formula), dtype=torch.float32),
            torch.tensor(df.values).flatten().float()
        ],
        dim=0
    ).reshape(1, -1).to(device)

    # Load the scaler
    scaler_loaded = joblib.load('src/scaler.pkl')

    # Scale in numpy, then convert to float32
    X_scaled = scaler_loaded.transform(X.cpu().numpy()).astype(np.float32)
    X_scaled = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    # Run the model
    result = model(X_scaled)
    print(f"Przewidziane EA dla {formula} = {result.item()}")

if __name__ == "__main__":
    main()
