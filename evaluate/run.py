#!/usr/bin/env python
import argparse
import itertools
import logging
import pandas as pd
import wandb
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="test")

    # --- Start modification for fetching test_data artifact with 'latest' alias ---
    logger.info("Downloading and reading test artifact")
    test_data_artifact_name = args.test_data
    if ':' not in test_data_artifact_name:
        test_data_artifact_name = f"{test_data_artifact_name}:latest"
    logger.info(f"Attempting to fetch test data artifact: {test_data_artifact_name}")

    try:
        # Assuming the test data is of type 'dataset'
        test_data_artifact = run.use_artifact(test_data_artifact_name, type='segregated data')
        test_data_path = test_data_artifact.file()
    except wandb.errors.CommError as e:
        logger.error(f"WandB Communication Error: Could not fetch test data artifact {test_data_artifact_name}. "
                     f"Please ensure it exists and has the specified alias/version. Error: {e}")
        run.finish(exit_code=1)
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching test data: {e}")
        run.finish(exit_code=1)
        return
    # --- End modification for fetching test_data artifact ---

    df = pd.read_csv(test_data_path, low_memory=False)

    # Extract the target from the features
    logger.info("Extracting target from dataframe")
    X_test = df.copy()
    y_test = X_test.pop("genre")

    # --- Start modification for fetching model_export artifact with 'latest' alias ---
    logger.info("Downloading and reading the exported model")
    model_export_artifact_name = args.model_export
    if ':' not in model_export_artifact_name:
        model_export_artifact_name = f"{model_export_artifact_name}:latest"
    logger.info(f"Attempting to fetch model artifact: {model_export_artifact_name}")

    try:
        # Assuming the model export is of type 'model'
        model_export_artifact = run.use_artifact(model_export_artifact_name, type='model_export')
        model_export_path = model_export_artifact.download() # Use .download() for directories/models
    except wandb.errors.CommError as e:
        logger.error(f"WandB Communication Error: Could not fetch model artifact {model_export_artifact_name}. "
                     f"Please ensure it exists and has the specified alias/version. Error: {e}")
        run.finish(exit_code=1)
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching model: {e}")
        run.finish(exit_code=1)
        return
    # --- End modification for fetching model_export artifact ---

    pipe = mlflow.sklearn.load_model(model_export_path)

    # Note: If 'genre' column is also used in preprocessor transformers,
    # you might need to adjust 'used_columns' logic or ensure 'genre' is not in X_test at this stage
    # before passing to predict_proba.
    used_columns = list(itertools.chain.from_iterable([x[2] for x in pipe['preprocessor'].transformers]))
    
    # Ensure that 'used_columns' does not include the target column if it was used in preprocessing but needs to be absent for prediction
    # This might require careful handling depending on your preprocessing pipeline
    # If your preprocessor expects 'genre' as input, remove this next line.
    # Otherwise, it might cause issues if 'genre' is still in X_test.
    # For now, assuming genre is not expected by the preprocessor for features.
    if "genre" in used_columns:
        used_columns.remove("genre") # Remove target column if it sneaked into 'used_columns'
    
    pred_proba = pipe.predict_proba(X_test[used_columns])

    logger.info("Scoring")
    score = roc_auc_score(y_test, pred_proba, average="macro", multi_class="ovo")

    run.summary["AUC"] = score
    logger.info(f"AUC: {score:.4f}")

    logger.info("Computing confusion matrix")
    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_estimator(
        pipe,
        X_test[used_columns], # Ensure X_test only contains features expected by the model
        y_test,
        ax=sub_cm,
        normalize="true",
        values_format=".1f",
        xticks_rotation=90,
    )
    fig_cm.tight_layout()

    run.log(
        {
            "confusion_matrix": wandb.Image(fig_cm)
        }
    )
    
    logger.info("Testing complete.")
    run.finish() # Finish the W&B run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the provided model on the test artifact",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--model_export",
        type=str,
        help="Fully-qualified artifact name for the exported model to evaluate (e.g., 'model_pipe'). "
             "The script will automatically attempt to fetch the ':latest' version.",
        required=True,
    )

    parser.add_argument(
        "--test_data",
        type=str,
        help="Fully-qualified artifact name for the test data (e.g., 'processed_data_test'). "
             "The script will automatically attempt to fetch the ':latest' version.",
        required=True,
    )

    args = parser.parse_args()

    go(args)
