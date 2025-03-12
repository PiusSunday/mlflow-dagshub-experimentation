# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import logging
import sys
import warnings
from urllib.parse import urlparse

# Import DAGsHub and initialize it with the repository owner and repository name
import dagshub
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Initialize DAGsHub with your credentials and enable MLFlow tracking
dagshub.init(
    repo_owner="PiusSunday", repo_name="mlflow-dagshub-experimentation", mlflow=True
)

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)  # Change to INFO or DEBUG for more details
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    """Calculates and returns RMSE, MAE, and R2 metrics."""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output
    np.random.seed(40)  # Set random seed for reproducibility

    # URL to the wine quality dataset
    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"

    try:
        # Load the dataset from the URL
        data = pd.read_csv(csv_url, sep=";")
        logger.info(f"Successfully loaded dataset from {csv_url}")
    except Exception as e:
        logger.error(f"Unable to download training & test CSV: {e}")
        sys.exit(1)  # Exit if dataset loading fails

    # Split the dataset into training and testing sets (75% train, 25% test)
    train, test = train_test_split(data, test_size=0.25, random_state=42)

    # Separate features (X) and target (y)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Get alpha and l1_ratio from command-line arguments, or use defaults
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # MLFlow Experiment Setup
    experiment_name = "MLFlow Experimentation with DagsHub Experimentation"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
        logger.info(f"Created new MLFlow experiment: {experiment_name}")
    mlflow.set_experiment(experiment_name)

    # Start MLFlow Run
    with mlflow.start_run():
        logger.info(f"Starting MLFlow run with alpha={alpha}, l1_ratio={l1_ratio}")

        # Train the ElasticNet model
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Make predictions on the test set
        predicted_qualities = lr.predict(test_x)

        # Evaluate model performance
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # Log metrics and parameters
        print(f"Elasticnet model (alpha={alpha:.6f}, l1_ratio={l1_ratio:.6f}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Infer model signature and log the model
        predictions = lr.predict(train_x)
        signature = infer_signature(train_x, predictions)

        # Configure MLFlow tracking URI for DagsHub
        remote_server_uri = (
            "https://dagshub.com/PiusSunday/mlflow-dagshub-experimentation.mlflow"
        )
        mlflow.set_tracking_uri(remote_server_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Log the model with an input example to infer the signature
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr,
                "model",
                registered_model_name="ElasticnetWineModel",
                input_example=train_x[:1],
            )
            logger.info("Model registered to MLFlow Model Registry.")
        else:
            mlflow.sklearn.log_model(lr, "model", input_example=train_x[:1])
            logger.info("Model logged to MLFlow.")

        logger.info("MLFlow run completed.")
