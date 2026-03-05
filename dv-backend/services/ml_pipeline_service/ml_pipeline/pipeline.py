"""
Main machine learning pipeline module.
"""

# Library imports
import os
import sys
import traceback
from typing import Any, Callable, Dict, Optional

import pandas as pd
# Pipeline imports
from ml_pipeline.data_loader import CSVLoader, DemoDataLoader
from ml_pipeline.data_profiler import generate_complete_profile
from ml_pipeline.llm_agents import (DataSamplingAgent, ModelTrainingAgent,
                                    TypeConversionAgent)
from ml_pipeline.preprocess import (auto_cast_and_encode,
                                    generate_and_run_preprocessing_code,
                                    generate_preprocessing_insights,
                                    handle_null)
from ml_pipeline.utils import pretty_print_stats


class PipelineLogger:
    """Simple logger that writes to stdout for log capture"""

    def info(self, message: str):
        print(message, flush=True)
        sys.stdout.flush()

    def error(self, message: str):
        print(f"ERROR: {message}", flush=True)
        sys.stderr.flush()


# Global pipeline logger
_pipeline_logger = PipelineLogger()


def run_pipeline(X: pd.DataFrame, target_field: str, status_callback: Optional[Callable[[str, str], None]] = None, sample_percentage: float = None) -> tuple[Dict[str, Any], Any]:
    """
    Run the ML pipeline with optional status callbacks.

    Args:
        X: Input DataFrame
        target_field: Target column name
        status_callback: Optional callback function for status updates
                        Signature: callback(stage_name: str, status: str)
                        where status is one of: "start", "complete", "error"
        sample_percentage: Optional percentage (0-100) to sample the data.
                          If None, uses full dataset. If provided, forces sampling.

    Returns:
        tuple: (metrics, model)
    """
    # Ignore warnings for cleaner output
    import warnings
    warnings.filterwarnings("ignore")

    def update_status(stage_name: str, status: str):
        """Helper function to update status via callback or print."""
        if status_callback:
            status_callback(stage_name, status)
        else:
            if status == "start":
                _pipeline_logger.info(f"Starting {stage_name}...")
            elif status == "complete":
                _pipeline_logger.info(f"-- {stage_name} - Done --")
            elif status == "error":
                _pipeline_logger.error(f"-- {stage_name} - Failed --")

    try:
        # Print start message
        _pipeline_logger.info("Starting ML Pipeline...")
        _pipeline_logger.info(f"Dataset shape: {X.shape}")
        _pipeline_logger.info(f"Target column: {target_field}")

        # Stage 1: Auto Cast and Encode (Deterministic preprocessing)
        update_status("Type Conversion", "start")
        _pipeline_logger.info("Applying deterministic auto cast and encode...")
        X, encoding_report = auto_cast_and_encode(X)
        _pipeline_logger.info(
            f"Encoding complete. Processed {len(encoding_report)} columns")

        # Optional: LLM-based type conversion for additional refinement
        # type_converter = TypeConversionAgent()
        # X = type_converter.convert_types(X)
        update_status("Type Conversion", "complete")

        # Handle missing values (not tracked as separate stage)
        X = handle_null(X, strategy='mean')

        # Stage 2: Compute the sampled data frame
        update_status("Data Sampling", "start")
        sampler_agent = DataSamplingAgent()
        sampled_df = sampler_agent.sample_data(
            X, target_column=target_field, sample_percentage=sample_percentage)
        update_status("Data Sampling", "complete")
        _pipeline_logger.info(f'Sampled Data Shape: {sampled_df.shape}')
        _pipeline_logger.info(f'Population Shape: {X.shape}')

        # Stage 3: Complete data profile for the sampled DataFrame
        update_status(
            "Profile Generation (Sampled)", "start")
        complete_profile = generate_complete_profile(
            sampled_df, target_column=target_field
        )
        update_status(
            "Profile Generation (Sampled)", "complete")

        # Stage 4: Generate preprocessing insights
        update_status("Preprocessing Insights Generation", "start")
        insights = generate_preprocessing_insights(complete_profile)
        update_status("Preprocessing Insights Generation", "complete")

        # Stage 5: Apply preprocessing code
        update_status("Preprocessing Code Execution", "start")
        sampled_df = generate_and_run_preprocessing_code(
            complete_profile,
            insights,
            sampled_df
        )
        update_status("Preprocessing Code Execution", "complete")

        # Stage 6: Generate complete profile for the preprocessed DataFrame
        update_status(
            "Profile Generation (Preprocessed)", "start")
        complete_profile = generate_complete_profile(
            sampled_df, target_column=target_field
        )
        update_status(
            "Profile Generation (Preprocessed)", "complete")

        # Stage 7 & 8: Model Recommendation (LLM) + Model Training (AutoGluon)
        # Note: Both stages are tracked inside train_with_autogluon()
        # It will call status_callback for "Model Recommendation" and "Model Training"
        training_agent = ModelTrainingAgent()
        _pipeline_logger.info("Training with AutoGluon (phase2 approach)...")
        metrics, model = training_agent.train_with_autogluon(
            sampled_df=sampled_df,
            target_field=target_field,
            status_callback=status_callback  # Pass callback for both stages
        )
        _pipeline_logger.info("AutoGluon training completed!")

        _pipeline_logger.info("✅ ML Pipeline completed successfully.")
        return metrics, model

    except Exception as e:
        # If there's an error, we need to determine which stage failed
        # Print detailed stack trace for debugging
        _pipeline_logger.error(f"Pipeline failed: {e}")
        _pipeline_logger.error("Stack trace:")
        _pipeline_logger.error(traceback.format_exc())
        raise e


def main() -> None:
    """
    Main function to run the ML pipeline.
    """

    # Simulation I - UCI ML Demo Dataset
    try:
        print("\n🏁 Starting Simulation I - UCI ML Demo Dataset")
        dataset, target_field = DemoDataLoader.load_diabetes_dataset()
        metrics, model = run_pipeline(dataset, target_field=target_field)
        pretty_print_stats(metrics, model)
    except Exception as e:
        print(f"🔴 An error occurred during Simulation I: {e}")
        print("\n🔍 Stack trace:")
        print(traceback.format_exc())
        exit(1)

    # Simulation II - Local Weather Dataset
    try:
        print("\n🏁 Starting Simulation II - Local Weather Dataset")
        path = os.path.join(os.getcwd(), "simulation_datasets/weatherAUS.csv")
        dataset, target_field = CSVLoader.load_local(path), 'RainTomorrow'
        metrics, model = run_pipeline(dataset, target_field=target_field)
        pretty_print_stats(metrics, model)
    except Exception as e:
        print(f"🔴 An error occurred during Simulation II: {e}")
        print("\n🔍 Stack trace:")
        print(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()
