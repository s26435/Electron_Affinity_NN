import numpy as np
import pandas as pd
import json
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from tqdm import tqdm

from hiper_params import SAVE_DIC_LOC, SAVE_SCALER_LOC, device, batch_size, num_epochs

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



from models import ElectronAffinityRegressor, ElectronAffinityClassifier


def augment_dataframe(df, noise_std=0.01, multiplier=10):

    augmented_data = []

    for _ in range(multiplier - 1):
        noisy_df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        noisy_df[numeric_cols] += np.random.normal(loc=0, scale=noise_std, size=df[numeric_cols].shape)

        augmented_data.append(noisy_df)

    augmented_df = pd.concat([df] + augmented_data, ignore_index=True)

    return augmented_df



token_dict = []

def get_token_trans(target_token):
    for i, token in enumerate(token_dict):
        if token == target_token:
            return i

def tokenize_formula(formula):
    if len(token_dict) == 0:
        load_dic(SAVE_DIC_LOC)

    if len(formula) > 64:
        formula = formula[:64]

    tokenized = np.zeros(64)
    for i, token in enumerate(formula):
        if token.isnumeric():
            tokenized[i] = int(token)
        else:
            if token not in token_dict:
                token_dict.append(token)

            tokenized[i] = get_token_trans(token) + 10

    save_dic(SAVE_DIC_LOC)
    return tokenized

def save_dic(filename: str = "token_dict.json") -> None:
    """Zapisuje token_dict do pliku JSON."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(token_dict, f, ensure_ascii=False, indent=4)

def load_dic(filename: str = "token_dict.json") -> None:
    """Wczytuje token_dict z pliku JSON."""
    global token_dict
    try:
        with open(filename, "r", encoding="utf-8") as f:
            token_dict = json.load(f)
    except FileNotFoundError:
        token_dict = []

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def get_datasets(model_type: str, file_path: str = "descriptors.csv") -> tuple[DataLoader, DataLoader]:
    if model_type == "reg":
        df = pd.read_csv(file_path)
        augmented_df = augment_dataframe(df, multiplier=20)
        tokenized = augmented_df['formula'].apply(lambda x: tokenize_formula(x))
        tokenz = pd.DataFrame(tokenized.tolist())
        X = pd.concat([tokenz, augmented_df.iloc[:, 3:]], axis=1).values
        Y = augmented_df[' EA'].values.reshape(-1, 1)
    elif model_type == "preg":
        df = pd.read_csv(file_path)
        df = df[df[" EA"] >= 0]
        augmented_df = augment_dataframe(df, multiplier=20)
        tokenized = augmented_df['formula'].apply(lambda x: tokenize_formula(x))
        tokenz = pd.DataFrame(tokenized.tolist())
        X = pd.concat([tokenz, augmented_df.iloc[:, 3:]], axis=1).values
        Y = augmented_df[' EA'].values.reshape(-1, 1)
    elif model_type == "class":
        df = pd.read_csv(file_path)
        augmented_df = augment_dataframe(df, multiplier=20)
        augmented_df[" EA"] = (augmented_df[" EA"] > 3.6).astype(int)
        tokenized = augmented_df['formula'].apply(lambda x: tokenize_formula(x))
        tokenz = pd.DataFrame(tokenized.tolist())
        X = pd.concat([tokenz, augmented_df.iloc[:, 3:]], axis=1).values
        Y = augmented_df[' EA'].values.reshape(-1, 1)
    else:
        raise ValueError(f"model_type must be reg or preg got {model_type}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, SAVE_SCALER_LOC)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to(device)
    dataset_train = TensorDataset(X_train_tensor, Y_train_tensor)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_test = TensorDataset(X_test_tensor, Y_test_tensor)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return dataloader_train, dataloader_test

def get_model_with_params(model_type: str):
    if model_type == "reg":
        model = ElectronAffinityRegressor(True).to(device)
        criterion = nn.MSELoss()
        learning_rate = 0.00005
    elif model_type == "preg":
        model = ElectronAffinityRegressor(False).to(device)
        criterion = nn.MSELoss()
        learning_rate = 0.00005
    elif model_type == "class":
        model = ElectronAffinityClassifier().to(device)
        criterion = nn.BCELoss()
        learning_rate = 0.00005
    else:
        raise ValueError(f"model_type must be reg or preg got {model_type}")

    init_weights(model)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    return model, criterion, optimizer, scheduler, learning_rate


def train_model(model, criterion, optimizer, scheduler, dataloader_train):
    best_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 20
    pbar = tqdm(range(num_epochs), total=num_epochs, ncols=200, desc="Training", unit="epoch")
    for _ in pbar:
        model.train()
        total_loss = 0
        for x_batch, y_batch in dataloader_train:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        current_lr = scheduler.get_last_lr()
        avg_loss = total_loss / len(dataloader_train)
        scheduler.step(avg_loss)
        pbar.set_postfix(loss=avg_loss, best_los=best_loss, current_learning_rate=current_lr)
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    return model, best_loss


def eval_model(model, model_type,  dataloader_test):
    model.eval()
    y_true, y_pred_list = [], []

    with torch.no_grad():
        for x_batch, y_batch in dataloader_test:
            y_pred = model(x_batch.to(device))
            y_true.extend(y_batch.cpu().numpy().flatten())
            y_pred_list.extend(y_pred.cpu().numpy().flatten())

    y_true = np.array(y_true)
    y_pred_list = np.array(y_pred_list)

    if model_type in ["reg", "preg"]:
        mae = mean_absolute_error(y_true, y_pred_list)
        mse = mean_squared_error(y_true, y_pred_list)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred_list)
        mape = np.mean(np.abs((y_true - y_pred_list) / (y_true + 1e-6))) * 100
        smape = np.mean(100 * np.abs(y_true - y_pred_list) / ((np.abs(y_true) + np.abs(y_pred_list)) / 2))

        print(f"Test Loss (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape:.2f}%")
    elif model_type == "class":
        y_pred_classes = (y_pred_list > 0.5).astype(int)

        accuracy = accuracy_score(y_true, y_pred_classes)
        precision = precision_score(y_true, y_pred_classes, zero_division=0)
        recall = recall_score(y_true, y_pred_classes, zero_division=0)
        f1 = f1_score(y_true, y_pred_classes, zero_division=0)
        cm = confusion_matrix(y_true, y_pred_classes)

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
    else:
        raise ValueError(f"model_type must be reg or preg got {model_type}")

def save_model(model, model_type):
    if model_type == "reg":
        path = "src/reg_model.pth"
    elif model_type == "preg":
        path = "src/preg_model.pth"
    elif model_type == "class":
        path = "src/class_model.pth"
    else:
        raise ValueError(f"model_type must be reg or preg got {model_type}")

    torch.save(model, path)