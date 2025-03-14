import numpy as np
import pandas as pd
import json

import torch.nn as nn

SAVE_DIC_LOC = "src/dict.json"


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
