import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import sys

from model import LSTMRULPredictor
from train import (
    create_sequences_test, RULDataset, mc_dropout_inference,
    nasa_score, FEATURE_COLS, WINDOW_SIZE, HIDDEN_SIZE,
    NUM_LAYERS, DROPOUT, DATA_DIR, RESULTS_DIR
)

PLOTS_DIR = "../results/plots"


def load_model_and_data(device):
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_featured.csv"))
    X_test, y_test = create_sequences_test(test_df, WINDOW_SIZE, FEATURE_COLS)
    test_loader = DataLoader(RULDataset(X_test, y_test), batch_size=64, shuffle=False)

    model = LSTMRULPredictor(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(RESULTS_DIR, "lstm_rul.pth"),
        map_location=device
    ))
    model.eval()

    # Standard predictions
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(device)).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    # MC Dropout predictions
    mc_preds = mc_dropout_inference(model, test_loader, device, T=50)
    mc_mean = mc_preds.mean(axis=0)
    mc_std = mc_preds.std(axis=0)

    return all_true, all_preds, mc_mean, mc_std


def plot_predicted_vs_actual(y_true, y_pred, save_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="steelblue", label="Engines")
    ax.plot([0, 125], [0, 125], "r--", linewidth=1.5, label="Perfect prediction")

    # Error bands
    ax.fill_between([0, 125], [0 - 15, 125 - 15], [0 + 15, 125 + 15],
                    alpha=0.1, color="red", label="±15 cycle band")

    ax.set_xlabel("True RUL (cycles)", fontsize=12)
    ax.set_ylabel("Predicted RUL (cycles)", fontsize=12)
    ax.set_title("Predicted vs Actual RUL — LSTM Model", fontsize=13)
    ax.legend()
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 130)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.text(5, 115, f"R² = {r2:.4f}\nRMSE = {rmse:.2f} cycles",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "predicted_vs_actual.png"), dpi=150)
    plt.close()
    print("Saved: predicted_vs_actual.png")


def plot_error_distribution(y_true, y_pred, save_dir):
    errors = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of errors
    axes[0].hist(errors, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
    axes[0].axvline(errors.mean(), color="orange", linestyle="--",
                    linewidth=1.5, label=f"Mean error: {errors.mean():.2f}")
    axes[0].set_xlabel("Prediction Error (cycles)", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Error Distribution", fontsize=13)
    axes[0].legend()

    # Absolute error per engine
    abs_errors = np.abs(errors)
    engine_ids = np.arange(1, len(y_true) + 1)
    colors = ["red" if e > 20 else "steelblue" for e in abs_errors]
    axes[1].bar(engine_ids, abs_errors, color=colors, alpha=0.7, width=0.8)
    axes[1].axhline(abs_errors.mean(), color="orange", linestyle="--",
                    linewidth=1.5, label=f"Mean MAE: {abs_errors.mean():.2f}")
    axes[1].set_xlabel("Engine ID", fontsize=12)
    axes[1].set_ylabel("Absolute Error (cycles)", fontsize=12)
    axes[1].set_title("Per-Engine Absolute Error", fontsize=13)
    axes[1].legend()

    plt.suptitle("LSTM RUL Prediction Error Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_distribution.png"), dpi=150)
    plt.close()
    print("Saved: error_distribution.png")


def plot_uncertainty(y_true, mc_mean, mc_std, save_dir):
    engine_ids = np.arange(1, len(y_true) + 1)
    errors = np.abs(mc_mean - y_true)

    # Sort by uncertainty for cleaner visualization
    sort_idx = np.argsort(mc_std)
    sorted_ids = engine_ids[sort_idx]
    sorted_mean = mc_mean[sort_idx]
    sorted_true = y_true[sort_idx]
    sorted_std = mc_std[sort_idx]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Predictions with uncertainty bands sorted by uncertainty
    axes[0].fill_between(range(len(sorted_ids)),
                         sorted_mean - 2 * sorted_std,
                         sorted_mean + 2 * sorted_std,
                         alpha=0.3, color="steelblue", label="95% confidence interval")
    axes[0].fill_between(range(len(sorted_ids)),
                         sorted_mean - sorted_std,
                         sorted_mean + sorted_std,
                         alpha=0.4, color="steelblue", label="68% confidence interval")
    axes[0].plot(range(len(sorted_ids)), sorted_mean, "b-",
                 linewidth=1, label="MC Dropout mean prediction")
    axes[0].plot(range(len(sorted_ids)), sorted_true, "r.",
                 markersize=4, label="True RUL")
    axes[0].set_xlabel("Engine (sorted by uncertainty)", fontsize=12)
    axes[0].set_ylabel("RUL (cycles)", fontsize=12)
    axes[0].set_title("MC Dropout Predictions with Uncertainty Bands", fontsize=13)
    axes[0].legend(loc="upper left")

    # Plot 2: Uncertainty vs absolute error (does high uncertainty = high error?)
    axes[1].scatter(mc_std, errors, alpha=0.6, s=25, color="steelblue")
    axes[1].axvline(15, color="red", linestyle="--",
                    linewidth=1.5, label="High uncertainty threshold (std=15)")
    axes[1].set_xlabel("Prediction Uncertainty (std, cycles)", fontsize=12)
    axes[1].set_ylabel("Absolute Error (cycles)", fontsize=12)
    axes[1].set_title("Uncertainty vs Prediction Error\n"
                      "(well-calibrated model: high uncertainty → high error)", fontsize=13)
    axes[1].legend()

    # Correlation annotation
    corr = np.corrcoef(mc_std, errors)[0, 1]
    axes[1].text(0.05, 0.92, f"Pearson r = {corr:.3f}",
                 transform=axes[1].transAxes, fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "uncertainty_analysis.png"), dpi=150)
    plt.close()
    print("Saved: uncertainty_analysis.png")


def plot_model_comparison(save_dir):
    metrics_df = pd.read_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ["steelblue", "lightcoral", "lightgreen", "lightsalmon"]
    models = metrics_df["Model"].tolist()

    for ax, metric, title, better in zip(
        axes,
        ["rmse", "mae", "r2"],
        ["RMSE (cycles) ↓", "MAE (cycles) ↓", "R² Score ↑"],
        ["lower", "lower", "higher"]
    ):
        bars = ax.bar(models, metrics_df[metric], color=colors,
                      edgecolor="white", alpha=0.85)
        ax.set_title(title, fontsize=12)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)

        # Highlight best bar
        best_idx = metrics_df[metric].idxmin() if better == "lower" \
            else metrics_df[metric].idxmax()
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2)

        # Value labels on bars
        for bar, val in zip(bars, metrics_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Model Comparison — NASA CMAPSS FD001", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_comparison.png"), dpi=150)
    plt.close()
    print("Saved: model_comparison.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading model and data...")
    y_true, y_pred, mc_mean, mc_std = load_model_and_data(device)

    print("\nGenerating plots...")
    plot_predicted_vs_actual(y_true, y_pred, PLOTS_DIR)
    plot_error_distribution(y_true, y_pred, PLOTS_DIR)
    plot_uncertainty(y_true, mc_mean, mc_std, PLOTS_DIR)
    plot_model_comparison(PLOTS_DIR)

    print(f"\nAll plots saved to {PLOTS_DIR}/")
    print("Files:")
    for f in os.listdir(PLOTS_DIR):
        print(f"  - {f}")


if __name__ == "__main__":
    main()