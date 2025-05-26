import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class LSTMRegressor(nn.Module):
    """
    PyTorch LSTM model for time series regression (price forecasting).
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        """
        Args:
            input_size (int): Number of features per timestep.
            hidden_size (int): Number of LSTM units.
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout rate between LSTM layers.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch, 1).
        """
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last output
        out = self.fc(out)
        return out.squeeze(-1)

    @staticmethod
    def create_windowed_dataset(
        data: np.ndarray, targets: np.ndarray, window: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create windowed dataset for LSTM.
        Args:
            data (np.ndarray): Feature array of shape (n_samples, n_features).
            targets (np.ndarray): Target array of shape (n_samples,).
            window (int): Number of timesteps per sample.
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y) where X is (n_samples-window, window, n_features), y is (n_samples-window,)
        """
        X, y = [], []
        for i in range(len(data) - window):
            X.append(data[i:i+window])
            y.append(targets[i+window])
        return np.array(X), np.array(y)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: Optional[str] = None,
        verbose: bool = True,
        progress_callback: Optional[callable] = None
    ) -> None:
        """
        Train the LSTM regressor.
        Args:
            X (np.ndarray): Input of shape (n_samples, seq_len, n_features).
            y (np.ndarray): Targets of shape (n_samples,).
            epochs (int): Number of epochs.
            batch_size (int): Batch size.
            lr (float): Learning rate.
            device (str, optional): 'cuda' or 'cpu'.
            verbose (bool): Print progress.
            progress_callback (callable, optional): Callback for progress reporting.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        self.train()
        for epoch in range(epochs):
            losses = []
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if progress_callback is not None:
                progress_callback(epoch + 1, epochs, np.mean(losses))
            elif verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - Loss: {np.mean(losses):.6f}")

    def predict(self, X: np.ndarray, device: Optional[str] = None) -> np.ndarray:
        """
        Predict using the trained LSTM regressor.
        Args:
            X (np.ndarray): Input of shape (n_samples, seq_len, n_features).
            device (str, optional): 'cuda' or 'cpu'.
        Returns:
            np.ndarray: Predicted values of shape (n_samples,).
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval()
        self.to(device)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self(X_tensor).cpu().numpy()
        return preds

    def predict_with_uncertainty(self, X: np.ndarray, n_iter: int = 20, device: Optional[str] = None) -> Tuple[float, float]:
        """
        Predict with uncertainty estimation using MC Dropout.

        Args:
            X (np.ndarray): Input of shape (1, seq_len, n_features) or (n_samples, seq_len, n_features).
            n_iter (int): Number of stochastic forward passes.
            device (str, optional): 'cuda' or 'cpu'.

        Returns:
            Tuple[float, float]: (mean prediction, std deviation of predictions)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train()  # Enable dropout
        self.to(device)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        preds = []
        with torch.no_grad():
            for _ in range(n_iter):
                pred = self(X_tensor).cpu().numpy()
                preds.append(pred)
        preds = np.stack(preds, axis=0)  # (n_iter, n_samples)
        mean_pred = np.mean(preds, axis=0).squeeze()
        std_pred = np.std(preds, axis=0).squeeze()
        # If single sample, return scalars
        if mean_pred.shape == ():
            return float(mean_pred), float(std_pred)
        return mean_pred, std_pred

    def save(self, path: str) -> None:
        """
        Save the LSTM model state_dict to a file.
        Args:
            path (str): Path to save the model (.pt file).
        """
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2) -> 'LSTMRegressor':
        """
        Load an LSTM model from a state_dict file.
        Args:
            path (str): Path to the saved model (.pt file).
            input_size (int): Number of features per timestep.
            hidden_size (int): Number of LSTM units.
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout rate between LSTM layers.
        Returns:
            LSTMRegressor: Loaded model.
        """
        model = cls(input_size, hidden_size, num_layers, dropout)
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        return model 