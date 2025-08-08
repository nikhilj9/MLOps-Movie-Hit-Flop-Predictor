#!/usr/bin/env python3
"""
MLflow Experiment Tracking and Model Registry Verification Script
This script tests both experiment tracking and model registry functionality
"""

import os

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from zsrc.core.config import load_config
from zsrc.core.logging import setup_logging

logger = setup_logging()


def test_current_mlflow_setup():
    """Test the current MLflow configuration and identify issues."""
    print("=== MLflow Configuration Verification ===\n")

    # Load your config
    config = load_config()

    print(f"Config MLflow tracking URI: {config.mlflow.tracking_uri}")
    print(f"Config experiment name: {config.mlflow.experiment_name}")
    print(f"Config model name: {config.model.name}")
    print(f"Config artifact location: {config.mlflow.artifact_location}")

    # Check what MLflow is currently configured to use
    current_tracking_uri = mlflow.get_tracking_uri()
    print(f"\nCurrent MLflow tracking URI: {current_tracking_uri}")

    # Check if MLflow database exists
    if "sqlite" in current_tracking_uri:
        db_path = current_tracking_uri.replace("sqlite:///", "")
        db_exists = os.path.exists(db_path)
        print(f"SQLite database exists ({db_path}): {db_exists}")
        if db_exists:
            print(f"Database size: {os.path.getsize(db_path)} bytes")

    # Check mlruns directory
    mlruns_exists = os.path.exists("mlruns")
    print(f"mlruns directory exists: {mlruns_exists}")
    if mlruns_exists:
        experiments = os.listdir("mlruns")
        print(f"Experiments in mlruns: {experiments}")

    # Initialize MLflow client
    try:
        client = MlflowClient()
        print("\nMLflow client initialized successfully")

        # List all experiments
        experiments = client.search_experiments()
        print(f"Number of experiments found: {len(experiments)}")

        for exp in experiments:
            print(f"  - Experiment: {exp.name} (ID: {exp.experiment_id})")

        # Check for your specific experiment
        try:
            experiment = client.get_experiment_by_name(config.mlflow.experiment_name)
            if experiment:
                print(f"\nYour experiment '{config.mlflow.experiment_name}' exists!")
                print(f"  Experiment ID: {experiment.experiment_id}")
                print(f"  Lifecycle stage: {experiment.lifecycle_stage}")

                # Check runs in this experiment
                runs = client.search_runs([experiment.experiment_id])
                print(f"  Number of runs: {len(runs)}")

                for run in runs[:3]:  # Show first 3 runs
                    print(
                        f"    Run {run.info.run_id[:8]}... - Status: {run.info.status}"
                    )
            else:
                print(
                    f"\nExperiment '{config.mlflow.experiment_name}' does not exist yet"
                )

        except Exception as e:
            print(f"Error getting experiment: {e}")

        # Check model registry
        print("\n=== Model Registry Check ===")

        try:
            registered_models = client.search_registered_models()
            print(f"Number of registered models: {len(registered_models)}")

            for model in registered_models:
                print(f"  Model: {model.name}")
                versions = client.get_latest_versions(model.name)
                for version in versions:
                    print(
                        f"    Version {version.version} - Stage: {version.current_stage}"
                    )

            # Check for your specific model
            try:
                model_versions = client.get_latest_versions(config.model.name)
                if model_versions:
                    print(f"\nYour model '{config.model.name}' exists in registry!")
                    for version in model_versions:
                        print(
                            f"  Version {version.version} - Stage: {version.current_stage}"
                        )
                else:
                    print(f"\nModel '{config.model.name}' not found in registry")
            except Exception as e:
                print(f"Model '{config.model.name}' not found in registry: {e}")

        except Exception as e:
            print(f"Error getting register models: {e}")

    except Exception as e:
        print(f"Error initializing MLflow client: {e}")
        return False

    return True


if __name__ == "__main__":
    print("MLflow Experiment Tracking and Model Registry Verification")
    print("=" * 60)

    success = test_current_mlflow_setup()

    if success:
        print("\n MLflow client connection successful")
    else:
        print("\n MLflow client connection failed")

    print("\nNext steps:")
    print("1. If you see experiments and models, your setup is working")
    print("2. If not, we'll run the training pipeline to create them")
    print("3. Then we'll test model registry functionality")
