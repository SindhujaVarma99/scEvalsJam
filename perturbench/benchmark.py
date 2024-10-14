from typing import List, Tuple
from perturbench.models import PerturbationModel
from perturbench.dataset import PerturbationDataset

class PerturbationBenchmark:
    """Responsible for comparing performance across different scenarios/models."""

    def __init__(self,
                 models: List[PerturbationModel] = [],
                 datasets: List[PerturbationDataset] = [],
                 scenarios: List = [],  # List to hold scenarios
                 metrics: List[str] = ['r2', 'mse', 'mae'],
                 gene_subset: List[str] = ['all_genes'],
                 **kwargs):
        self.models = models
        self.datasets = datasets
        self.scenarios = scenarios
        self.metrics = metrics
        self.results = []
    def add_model(self, model: PerturbationModel):
        """Add a model to the list of perturbation benchmark, only if not already added."""
        if model not in self.models:
            self.models.append(model)
            print(f"Added model: {model.model_name}")
        else:
            print(f"Model {model.model_name} already added!")

    def add_data(self, dataset: PerturbationDataset):
        """Add a dataset to the list of perturbation benchmark."""
        self.datasets.append(dataset)

    def add_scenario(self, scenario):
        """Add a scenario to the list of perturbation benchmark."""
        self.scenarios.append(scenario)

    def train(self, train_data: PerturbationDataset):
        """Train each model in the list of perturbation benchmark."""
        for model in self.models:
            if not model.istrained:
                model.train(train_data)
            else:
                print(f"{model.model_name} is already trained, skipping...")

    def predict(self, test_data: PerturbationDataset, perturbation: List[str]):
        """Predict each model in the list of perturbation benchmark."""
        for model in self.models:
            if not model.ispredicted:
                model.predict(test_data, perturbation)
            else:
                print(f"{model.model_name} has already generated predictions, skipping...")


    def calculate_metrics(self):
        """Calculate metrics for each model in the list of perturbation benchmark."""
        for model in self.models:
            # Ensure metrics are not calculated twice
            if not any(result[0] == model.model_name for result in self.results):
                metrics = model.calculate_metrics()
                self.results.append((model.model_name, metrics))
                print(f"Model {model.model_name} metrics are calculated successfully.")
            else:
                print(f"Metrics for {model.model_name} already exist, skipping...")

    def run(self, train_data: PerturbationDataset, test_data: PerturbationDataset, perturbation: List[str]):
        """Run the training, prediction, and metric calculation for each model in the list of perturbation benchmark."""
        self.train(train_data)
        self.predict(test_data, perturbation)
        self.calculate_metrics()

    def summary(self):
        """Print a summary of the results, including scenarios used."""
        print("Benchmark Summary:")
        for scenario in self.scenarios:
            print(f"Scenario: {scenario.__class__.__name__}")

        for model_name, metrics in self.results:
            print(f"Model: {model_name}")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value}")

    def compare_models(self):
        """Compares the performance of the models and prints the order along with ranks."""
        # Create a list of tuples with model names and their metrics
        performance = []
        for model_name, metrics in self.results:
            performance.append((model_name, metrics['mse'], metrics['mae'], metrics['r2']))

        # Sort the models based on MSE first, then MAE, and finally R²
        performance.sort(key=lambda x: (x[1], x[2], -x[3]))

        print("\nModel Performance Comparison:")
        print("{:<5} {:<25} {:<15} {:<15} {:<15}".format("Rank", "Model", "MSE", "MAE", "R²"))
        print("-" * 80)
        for rank, model in enumerate(performance, start=1):
            print("{:<5} {:<25} {:<15.4f} {:<15.4f} {:<15.4f}".format(rank, model[0], model[1], model[2], model[3]))

