import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import time
from model import LSTMRULPredictor

# ----------------- CONFIG -----------------
WINDOW_SIZE = 30
BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
PATIENCE = 10
RUL_CLIP = 125
DATA_DIR = "../data"
RESULTS_DIR = "../results"
# ------------------------------------------

FEATURE_COLS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
    "sensor_2_mean5", "sensor_2_std5", "sensor_3_mean5", "sensor_3_std5",
    "sensor_4_mean5", "sensor_4_std5", "sensor_7_mean5", "sensor_7_std5",
    "sensor_8_mean5", "sensor_8_std5", "sensor_9_mean5", "sensor_9_std5",
    "sensor_11_mean5", "sensor_11_std5", "sensor_12_mean5", "sensor_12_std5",
    "sensor_13_mean5", "sensor_13_std5", "sensor_14_mean5", "sensor_14_std5",
    "sensor_15_mean5", "sensor_15_std5", "sensor_17_mean5", "sensor_17_std5",
    "sensor_20_mean5", "sensor_20_std5", "sensor_21_mean5", "sensor_21_std5"
]


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────
def create_sequences_train(df, window_size, feature_cols):
    """
    For training: slide a window over each engine's cycles.
    Each sequence of `window_size` cycles predicts the RUL
    at the last cycle of that window.
    """
    X, y = [], []
    for engine_id, group in df.groupby("engine_id"):
        group = group.sort_values("cycle").reset_index(drop=True)
        features = group[feature_cols].values
        targets = group["RUL"].values
        for i in range(len(group) - window_size + 1):
            X.append(features[i:i + window_size])
            y.append(targets[i + window_size - 1])
    return np.array(X), np.array(y)


def create_sequences_test(df, window_size, feature_cols):
    """
    For testing: take the LAST `window_size` cycles of each engine.
    The true RUL is the RUL at that last cycle.
    """
    X, y = [], []
    for engine_id, group in df.groupby("engine_id"):
        group = group.sort_values("cycle").reset_index(drop=True)
        features = group[feature_cols].values
        targets = group["RUL"].values
        if len(group) < window_size:
            pad = np.zeros((window_size - len(group), len(feature_cols)))
            features = np.vstack([pad, features])
        X.append(features[-window_size:])
        y.append(targets[-1])
    return np.array(X), np.array(y)


class RULDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────
# NASA Scoring Function
# ─────────────────────────────────────────
def nasa_score(y_true, y_pred):
    """
    Official NASA scoring function from the CMAPSS paper.
    Penalizes late predictions more heavily than early ones.
    Lower is better.
    """
    d = y_pred - y_true
    score = np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))
    return score


# ─────────────────────────────────────────
# Training
# ─────────────────────────────────────────
def train_lstm(model, train_loader, val_loader, num_epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"No improve: {epochs_no_improve}/{PATIENCE}")

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    print(f"\nBest Val Loss: {best_val_loss:.4f}")
    return model


# ─────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────
def evaluate(y_true, y_pred, label="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    score = nasa_score(y_true, y_pred)

    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"{'='*40}")
    print(f"  RMSE:        {rmse:.4f} cycles")
    print(f"  MAE:         {mae:.4f} cycles")
    print(f"  R²:          {r2:.4f}")
    print(f"  NASA Score:  {score:.2f} (lower is better)")
    print(f"{'='*40}")

    return {"rmse": rmse, "mae": mae, "r2": r2, "nasa_score": score}

# ─────────────────────────────────────────
# MC Dropout Inference
# Runs T stochastic forward passes with dropout
# ACTIVE to estimate prediction uncertainty
# ─────────────────────────────────────────
def mc_dropout_inference(model, test_loader, device, T=50):
    """
    Runs T forward passes with dropout enabled.
    Returns array of shape (T, n_samples) — one prediction
    per pass per engine. Mean = prediction, Std = uncertainty.
    """
    # Enable dropout at inference by setting model to train mode
    # but disabling gradient computation
    model.train()

    all_passes = []
    with torch.no_grad():
        for t in range(T):
            preds_t = []
            for X_batch, _ in test_loader:
                preds = model(X_batch.to(device)).cpu().numpy()
                preds_t.extend(preds)
            all_passes.append(preds_t)

    model.eval()  # restore eval mode after MC passes
    return np.array(all_passes)  # shape: (T, n_engines)


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_featured.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_featured.csv"))

    # Create sequences
    print("\nCreating sequences...")
    X_train_full, y_train_full = create_sequences_train(train_df, WINDOW_SIZE, FEATURE_COLS)
    X_test, y_test = create_sequences_test(test_df, WINDOW_SIZE, FEATURE_COLS)

    print(f"Train sequences: {X_train_full.shape}")
    print(f"Test sequences:  {X_test.shape}")

    # Train/val split (80/20)
    split = int(0.8 * len(X_train_full))
    X_train, y_train = X_train_full[:split], y_train_full[:split]
    X_val, y_val = X_train_full[split:], y_train_full[split:]

    # Dataloaders
    train_loader = DataLoader(RULDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(RULDataset(X_val, y_val),
                            batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(RULDataset(X_test, y_test),
                             batch_size=BATCH_SIZE, shuffle=False)

    # ---- Train LSTM ----
    print("\nTraining LSTM...")
    start = time.time()
    model = LSTMRULPredictor(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    model = train_lstm(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, device)
    elapsed = time.time() - start
    print(f"Training time: {elapsed:.1f}s")

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "lstm_rul.pth"))

    # ---- Evaluate LSTM (standard) ----
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(device)).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    lstm_metrics = evaluate(all_true, all_preds, label="LSTM (ours)")

    # ---- MC Dropout Inference (uncertainty quantification) ----
    print("\nRunning MC Dropout inference (T=50 forward passes)...")
    mc_preds = mc_dropout_inference(model, test_loader, device, T=50)
    mc_mean = mc_preds.mean(axis=0)   # shape: (n_engines,)
    mc_std  = mc_preds.std(axis=0)    # shape: (n_engines,) — this is our uncertainty

    print("\n--- MC Dropout Results ---")
    print(f"Mean prediction error vs standard eval: "
          f"{np.abs(mc_mean - all_preds).mean():.4f} cycles")

    mc_metrics = evaluate(all_true, mc_mean, label="LSTM + MC Dropout (averaged)")

    # Save uncertainty estimates per engine
    uncertainty_df = pd.DataFrame({
        "engine_id": range(1, len(all_true) + 1),
        "true_RUL": all_true,
        "pred_RUL_mean": mc_mean,
        "pred_RUL_std": mc_std,
        "error": np.abs(mc_mean - all_true)
    })
    uncertainty_df.to_csv(os.path.join(RESULTS_DIR, "uncertainty_estimates.csv"), index=False)

    print(f"\nAverage uncertainty (std): {mc_std.mean():.4f} cycles")
    print(f"Engines with high uncertainty (std > 15): "
          f"{(mc_std > 15).sum()} / {len(mc_std)}")
    print(f"Uncertainty estimates saved to {RESULTS_DIR}/uncertainty_estimates.csv")

    # ---- Baseline 1: Mean RUL ----
    mean_pred = np.full_like(all_true, y_train.mean())
    evaluate(all_true, mean_pred, label="Baseline: Mean RUL")

    # ---- Baseline 2: Linear Regression ----
    X_train_flat = X_train[:, -1, :]
    X_test_flat = X_test[:, -1, :]
    lr = LinearRegression()
    lr.fit(X_train_flat, y_train)
    lr_preds = lr.predict(X_test_flat)
    evaluate(all_true, lr_preds, label="Baseline: Linear Regression")

    # ---- Baseline 3: Random Forest ----
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train_flat, y_train)
    rf_preds = rf.predict(X_test_flat)
    evaluate(all_true, rf_preds, label="Baseline: Random Forest")

    # ---- Save results ----
    results = pd.DataFrame([
        {"Model": "LSTM", **lstm_metrics},
        {"Model": "Mean Baseline",
         "rmse": np.sqrt(mean_squared_error(all_true, mean_pred)),
         "mae": mean_absolute_error(all_true, mean_pred),
         "r2": r2_score(all_true, mean_pred),
         "nasa_score": nasa_score(all_true, mean_pred)},
        {"Model": "Linear Regression",
         "rmse": np.sqrt(mean_squared_error(all_true, lr_preds)),
         "mae": mean_absolute_error(all_true, lr_preds),
         "r2": r2_score(all_true, lr_preds),
         "nasa_score": nasa_score(all_true, lr_preds)},
        {"Model": "Random Forest",
         "rmse": np.sqrt(mean_squared_error(all_true, rf_preds)),
         "mae": mean_absolute_error(all_true, rf_preds),
         "r2": r2_score(all_true, rf_preds),
         "nasa_score": nasa_score(all_true, rf_preds)},
    ])
    results.to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"), index=False)
    print(f"\nMetrics saved to {RESULTS_DIR}/metrics_comparison.csv")

    return lstm_metrics


if __name__ == "__main__":
    main()