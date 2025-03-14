import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error

import utils
from models import *
from descryptors import *
from utils import *

import joblib

batch_size = 256
num_epochs = 1000
learning_rate = 0.00005

# sprawdzanie czy dostępne jest trenowanie na GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Przygotowywanie modelu
model= ElectronAffinityRegressor().to(device)
utils.init_weights(model)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# przygotowywanie danych,
df = pd.read_csv("descriptors.csv")
augmented_df = augment_dataframe(df, multiplier=20)
tokenized = augmented_df['formula'].apply(lambda x: tokenize_formula(x))
tokenz = pd.DataFrame(tokenized.tolist())
X = pd.concat([tokenz, augmented_df.iloc[:, 3:]], axis=1).values
Y = augmented_df[' EA'].values.reshape(-1, 1)

# normalizacja danych
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, 'src/scaler.pkl')


# dzielenie danych na zbiory testowe i treningowe
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to(device)
dataset_train = TensorDataset(X_train_tensor, Y_train_tensor)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataset_test = TensorDataset(X_test_tensor, Y_test_tensor)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# trening
best_loss = float('inf')
early_stop_counter = 0
early_stop_patience = 20
current_lr = learning_rate
pbar = tqdm(range(num_epochs), total=num_epochs, ncols=200, desc="Training", unit="epoch")
for epoch in pbar:
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


# ewaluacja modelu
model.eval()
y_true, y_pred_list = [], []

with torch.no_grad():
    for x_batch, y_batch in dataloader_test:
        y_pred = model(x_batch.to(device))
        y_true.extend(y_batch.cpu().numpy().flatten())
        y_pred_list.extend(y_pred.cpu().numpy().flatten())

y_true = np.array(y_true)
y_pred_list = np.array(y_pred_list)

mae = mean_absolute_error(y_true, y_pred_list)
mse = mean_squared_error(y_true, y_pred_list)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred_list)
mape = np.mean(np.abs((y_true - y_pred_list) / (y_true + 1e-6))) * 100
smape = np.mean(100 * np.abs(y_true - y_pred_list) / ((np.abs(y_true) + np.abs(y_pred_list)) / 2))

# wyświetlanie parametrów ewaluacji
print(f"Test Loss (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape:.2f}%")

# zapis modelu
torch.save(model, "src/model.pth")