import os.path

from utils import *
from hiper_params import device, num_epochs
from descryptors import get_one

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, help="Nazwa modelu:\n-`class` - Classificator (>3.6 eV)\n- `preg` - positive only Regressor\n-`reg` - Regressor", required=True)
parser.add_argument("--action", type=str, help="`train` - jeśli chcesz trenować nowy model\n`use` - jeśli chcesz użyć już wytrenowany", required=True)
parser.add_argument("--target", type=str, help="jeśli trenujesz model tu wzór do predykcji np. `--target CO2`")

args = parser.parse_args()

def train_new_model(model_type):
    print(f"Using device: {device}")
    model, criterion, optimizer, scheduler, learning_rate = get_model_with_params(model_type)
    dataloader_train, dataloader_test = get_datasets(model_type)
    model, loss = train_model(model, criterion, optimizer, scheduler, dataloader_train)
    print(f"Training loss: {loss}")
    eval_model(model, model_type, dataloader_test)
    save_model(model, model_type)

def use_model(model_type):
    path = f"src/{model_type}_model.pth"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(path, weights_only=False, map_location=device)
    model = model.float()
    model.eval()
    print("Załadowano model!")

    if args.target is not None and len(args.target) > 1:
        formula = args.target
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

    scaler_loaded = joblib.load('src/scaler.pkl')

    X_scaled = scaler_loaded.transform(X.cpu().numpy()).astype(np.float32)
    X_scaled = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    result = model(X_scaled)
    if model_type in ["reg", "preg"]:
        print(f"Przewidziane EA dla {formula} = {result.item()}")
    elif model_type == "class":
        ans = "" if result.item() > 0.5 else "nie"
        print(f"Przewidziane EA dla {formula}  go jako {ans} mającego EA > 3.6 eV")

if __name__ == "__main__":
    if args.action == "train":
        train_new_model(args.model)
    elif args.action == "use":
        try:
            use_model(args.model)
        except FileNotFoundError as e:
            print(e)
            ans = input("Czy chcesz nauczyć model od nowa? [t/n]")
            if ans.lower() == "t":
                train_new_model(args.model)
                use_model(args.model)
            else:
                print("Kończenie pracy programu")
                exit(0)