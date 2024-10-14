from sklearn.ensemble import RandomForestRegressor
from perturbench.models import PerturbationModel
from perturbench.dataset import PerturbationDataset
import torch
import pathlib
import numpy as np
import scipy as sp
import os
import uuid
import joblib


class RandomForestModel(PerturbationModel):
    """Uses a RandomForestRegressor for predictions."""

    def __init__(self, device: torch.device, model_name: str = "RandomForest Model", **kwargs) -> None:
        self.kwargs = kwargs
        self.device = device
        self.model = RandomForestRegressor(**kwargs)
        self.model_name = model_name
        self.predictions = None
        self.true_values = None
        self.istrained = False
        self.ispredicted = False

    def train(self, data: PerturbationDataset) -> None:
        if not self.istrained:  # Check if the model is already trained
            print(f"Training {self.model_name}...")
            raw_data = data.raw_counts
            if sp.sparse.issparse(raw_data):
                raw_data = raw_data.toarray()
            self.true_values = raw_data
            X = raw_data  # Features (e.g., cells)
            y = raw_data.mean(axis=1)  # Example target: mean expression per cell (customize as needed)

            self.model.fit(X, y)
            self.istrained = True  # Set the flag after training
            print(f"Model {self.model_name} is trained successfully")
        else:
            print(f"Model {self.model_name} is already trained, skipping...")

    def predict(self, data: PerturbationDataset, perturbation: list) -> sp.sparse.csr_matrix:
        if not self.ispredicted:  # Check if the model has already generated predictions
            raw_data = data.raw_counts
            if sp.sparse.issparse(raw_data):
                raw_data = raw_data.toarray()

            self.predictions = self.model.predict(raw_data)
            self.ispredicted = True  # Set the flag after predictions
            print(f"Model {self.model_name} generated predictions successfully.")
        else:
            print(f"Model {self.model_name} has already generated predictions, skipping...")

        return sp.sparse.csr_matrix(self.predictions)

    def calculate_metrics(self):
        if self.predictions is None or self.true_values is None:
            raise ValueError("Predictions or true values are not set.")

        mse = np.mean((self.predictions - self.true_values.mean(axis=1)) ** 2)
        mae = np.mean(np.abs(self.predictions - self.true_values.mean(axis=1)))
        r2 = 1 - (np.sum((self.predictions - self.true_values.mean(axis=1)) ** 2) /
                  np.sum((self.true_values.mean(axis=1) - np.mean(self.true_values.mean(axis=1))) ** 2))

        return {"mse": mse, "mae": mae, "r2": r2}

    def save(self) -> pathlib.Path:
        path = pathlib.Path(os.path.join("artefacts", "models", str(uuid.uuid4())))
        os.makedirs(path, exist_ok=True)
        model_path = path / f"{self.model_name}.joblib"
        joblib.dump(self.model, model_path)  # Save the model using joblib
        print(f"Model {self.model_name} saved to {model_path}")
        return model_path

    def load(self, path: pathlib.Path) -> None:
        self.model = joblib.load(path / f"{self.model_name}.joblib")  # Load the model using joblib
        self.istrained = True  # Set the flag to indicate the model is now trained
        print(f"Model {self.model_name} loaded from {path}")
