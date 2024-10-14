import pandas as pd
import numpy as np
import scipy as sp
import anndata as ad
import torch
from sklearn.preprocessing import LabelEncoder
from perturbench.dataset import PerturbationDataset
from perturbench.scenarios import RandomSplitScenario
from perturbench.metrics import AverageDifferenceMetric
from perturbench.benchmark import PerturbationBenchmark
from perturbench.models import RandomForestModel, SVMModel, GradientBoostingModel

# Load the data from the .gct file
# Load the data from the .gct file
file_path = "data/CCLE_miRNA_20181103.gct"  # Update this line

# Read the GCT file while skipping the first two metadata lines
with open(file_path) as f:
    f.readline()  # Skip version line
    f.readline()  # Skip dimensions line
    column_names = f.readline().strip().split('\t')  # Read the column names

# Read the data into a DataFrame, skipping the first three lines for header
raw_counts = pd.read_csv(file_path, sep='\t', comment='#', skiprows=3, header=None, names=column_names)

# Set the first column as the index (which contains identifiers)
raw_counts.set_index(column_names[0], inplace=True)

# Convert numeric columns to float
raw_counts = raw_counts.select_dtypes(include=[np.number]).astype(float)

# Convert to sparse matrix
raw_counts_sparse = sp.sparse.csr_matrix(raw_counts.values)

# Generate actual values for perturbations, sex, and cell type
n_cells = raw_counts.shape[0]
perturbation_choices = ["GENETIC:MYC", "GENETIC:AKT", "GENETIC:PD1", None]
perturbations = np.random.choice(perturbation_choices, size=n_cells)
sex = np.random.choice(["male", "female"], size=n_cells)
cell_type = np.random.choice(["cell_type1", "cell_type2"], size=n_cells)

# Encode perturbations
label_encoder = LabelEncoder()
encoded_perturbations = label_encoder.fit_transform(perturbations)

# Create AnnData object
anndata = ad.AnnData(raw_counts_sparse, obs=pd.DataFrame({"perturbation": encoded_perturbations, "sex": sex, "cell_type": cell_type}))

# Create the perturbation dataset
perturbation_dataset = PerturbationDataset(anndata, "perturbation", ["sex", "cell_type"])

# Create the scenario
my_favourite_scenario = RandomSplitScenario()

# Initialize the models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_forest_model = RandomForestModel(device=device)
gradient_boosting_model = GradientBoostingModel(device=device)
svm_model = SVMModel(device=device)

# Choose the metric
my_favourite_metric = AverageDifferenceMetric()

# Initialize the benchmark
benchmark = PerturbationBenchmark()

# Register datasets, scenarios, and metrics
benchmark.add_data(perturbation_dataset)
benchmark.add_scenario(my_favourite_scenario)

# Ensure each model is only added once!
if "RandomForest Model" not in [m.model_name for m in benchmark.models]:
    benchmark.add_model(random_forest_model)
if "Gradient Boosting Model" not in [m.model_name for m in benchmark.models]:
    benchmark.add_model(gradient_boosting_model)
if "SVM Model" not in [m.model_name for m in benchmark.models]:
    benchmark.add_model(svm_model)
print(f"Models after adding: {[m.model_name for m in benchmark.models]}")

# Run the benchmarking - this should only train each model once!
benchmark.run(train_data=perturbation_dataset, test_data=perturbation_dataset, perturbation=encoded_perturbations)

# View the results as a table
benchmark.summary()

# Compare and print the order of model performance
benchmark.compare_models()
