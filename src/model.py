import torch
import torch.nn as nn


# ─────────────────────────────────────────
# Locked Dropout (Variational Dropout)
# Same mask applied across all timesteps
# From: Merity et al. (2017) "Regularizing and
# Optimizing LSTM Language Models" (AWD-LSTM paper)
# ─────────────────────────────────────────
class LockedDropout(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # x: (batch, seq_len, features)
        # Create mask on first timestep only, then expand across all timesteps
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)
        mask = mask / (1 - self.p)  # scale to preserve expected value
        mask = mask.expand_as(x)
        return x * mask


# ─────────────────────────────────────────
# LSTM with LayerNorm + Locked Dropout
# ─────────────────────────────────────────
class LSTMRULPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 dropout=0.3, locked_dropout=0.3):
        """
        input_size:      number of features per timestep
        hidden_size:     LSTM hidden units
        num_layers:      stacked LSTM layers
        dropout:         standard dropout between LSTM layers
        locked_dropout:  variational dropout applied across timesteps
        """
        super(LSTMRULPredictor, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Input projection + LayerNorm on raw features
        self.input_norm = nn.LayerNorm(input_size)

        # Locked dropout applied to input
        self.locked_drop = LockedDropout(p=locked_dropout)

        # Stacked LSTM layers (one at a time so we can apply
        # LayerNorm + LockedDropout between each layer)
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            self.lstm_layers.append(
                nn.LSTM(input_dim, hidden_size, batch_first=True)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_size))

        # Regressor head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)

        # Normalize inputs
        out = self.input_norm(x)

        # Apply locked dropout to input
        out = self.locked_drop(out)

        # Pass through each LSTM layer with LayerNorm + LockedDropout
        for i, (lstm, norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            out, _ = lstm(out)
            out = norm(out)
            # Apply locked dropout between layers (not after last layer)
            if i < self.num_layers - 1:
                out = self.locked_drop(out)

        # Take last timestep
        last_out = out[:, -1, :]

        return self.regressor(last_out).squeeze(-1)


if __name__ == "__main__":
    model = LSTMRULPredictor(input_size=42, hidden_size=64, num_layers=2)
    dummy = torch.randn(32, 30, 42)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test locked dropout is active in train mode
    model.train()
    out1 = model(dummy)
    out2 = model(dummy)
    print(f"Outputs differ in train mode (locked dropout active): {not torch.allclose(out1, out2)}")

    # Test MC Dropout behavior
    model.eval()
    out_eval1 = model(dummy)
    out_eval2 = model(dummy)
    print(f"Outputs identical in eval mode (no dropout): {torch.allclose(out_eval1, out_eval2)}")