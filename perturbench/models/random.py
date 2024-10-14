import numpy as np
import scipy as sp
import pathlib
import torch
from typing import List, Any
import uuid
import os

from perturbench.models import PerturbationModel
from perturbench.dataset import PerturbationDataset


class RandomModel(PerturbationModel):
    """Samples from a normal distribution"""
    def __init__(self, device: torch.device, model_name: str = "Random Model", **kwargs) -> None:
        self.kwargs = kwargs
        self.device = device
        self.model = "initialised!"
        self.model_name = model_name  # Add model_name attribute
        self.predictions = None
        self.true_values = None
    def train(self, data: PerturbationDataset) -> None:
        self.model = "trained!"
        print(f"{self.model} successfully.")

    def predict(self, data: PerturbationDataset, perturbation: List[str]) -> sp.sparse.csr_matrix:
        raw_data = data.raw_counts  # Ensure this method exists
        shape = raw_data.shape

        # Generate random predictions based on the shape of the raw data
        self.predictions = np.random.normal(size=shape)

        # Convert raw_data to dense if it is a sparse matrix
        if sp.sparse.issparse(raw_data):
            self.true_values = raw_data.toarray()  # Assuming raw_data is a sparse matrix
        else:
            self.true_values = raw_data  # If it's already dense

        print(f"Predictions for {self.model} generated successfully.")

        # Return predictions in sparse matrix format
        return sp.sparse.csr_matrix(self.predictions)

    def calculate_metrics(self):
        if self.predictions is None or self.true_values is None:
            raise ValueError("Predictions or true values are not set.")

        # Calculate metrics
        mse = np.mean((self.predictions - self.true_values) ** 2)
        mae = np.mean(np.abs(self.predictions - self.true_values))
        r2 = 1 - (np.sum((self.predictions - self.true_values) ** 2) /
                  np.sum((self.true_values - np.mean(self.true_values)) ** 2))

        # Return or print the metrics
        return {"mse": mse, "mae": mae, "r2": r2}

    def save(self) -> pathlib.Path:
        path = pathlib.Path(os.path.join("artefacts", "models", str(uuid.uuid4())))
        os.makedirs(path, exist_ok=True)  # Create the directory if it doesn't exist
        with open(os.path.join(path, "model.txt"), "w") as file:
            file.write(self.model)
        return path

    def load(self, path: pathlib.Path) -> None:
        with open(os.path.join(path, "model.txt"), "r") as file:
            self.model = file.read()