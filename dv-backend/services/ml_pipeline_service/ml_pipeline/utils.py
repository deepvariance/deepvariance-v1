"""
File for utility functions in the ML pipeline.
"""


def pretty_print_stats(metrics, model) -> str:
    """
    Pretty print the model metrics and model details, also returns a plain text summary.

    Args:
        metrics (dict): Dictionary containing model metrics.
        model (object): Trained model object.
    """

    plain_text = "Model Evaluation Metrics:\n"

    # Display metrics in a readable format
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
        plain_text += f"{metric}: {value}\n"

    # Display type of model
    print("\nTrained Model Type:")
    print(type(model).__name__)
    plain_text += f"\nTrained Model Type: {type(model).__name__}"

    return plain_text
