# Define the RandomModel with model_name
# PerturbationModel abstract class
from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np
import scipy as sp
import pandas as pd
import anndata as ad
import torch
import pathlib
import uuid
import os
from perturbench.dataset import PerturbationDataset
class PerturbationModel(ABC):
    """Class responsible for instantiating a model, training a model and performing a prediction."""

    @abstractmethod
    def __init__(self, device: torch.cuda.device, **kwargs) -> None:
        self.model_name = ''
        pass

    @abstractmethod
    def train(self, data: PerturbationDataset) -> None:
        pass

    @abstractmethod
    def predict(self, data: PerturbationDataset, perturbation: List[str]) -> sp.sparse.csr_matrix:
        """
        :param data:
            A PerturbationDataset where all cells are unperturbed (i.e. baseline), from which
            to make a prediction.
        :param perturbation:
            List of perturbations to predict where perturbations
            are encoded as described in PerturbationDataset.
        :return:
        """
        pass

    @abstractmethod
    def save(self) -> pathlib.Path:
        pass

    @abstractmethod
    def load(self, path: pathlib.Path) -> None:
        pass