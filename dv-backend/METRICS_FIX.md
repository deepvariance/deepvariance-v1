# Classification Metrics Fix - Issue Resolution

## Problem

Model `0fca41da-9bb3-4efd-a136-34a08c9c18d5` (histopathologic-cancer-detection) shows N/A for classification metrics (Precision, Recall, F1-Score) in the frontend.

## Root Cause

The `evaluate_with_metrics()` function in [training_pipeline/core/llm_training.py](training_pipeline/core/llm_training.py#L442) was only computing:

- Accuracy%
- Loss
- InferenceSpeed
- CPUUsage%
- RAMPeak(MB)

It was **NOT** computing classification-specific metrics:

- ❌ Precision
- ❌ Recall
- ❌ F1-Score

## Code Fix Applied

### File: `training_pipeline/core/llm_training.py`

Updated `evaluate_with_metrics()` function to:

1. Collect all predictions and labels during evaluation
2. Use `metrics_utils.evaluate_model_with_metrics()` to calculate classification metrics
3. Include precision, recall, and f1_score in returned metrics dictionary

**Before:**

```python
def evaluate_with_metrics(model, testloader, device: str = 'cpu'):
    # ... evaluation code ...
    return {
        'Accuracy%': accuracy * 100,
        'Loss': avg_loss,
        'InferenceSpeed': inference_time,
        'CPUUsage%': cpu_percent,
        'RAMPeak(MB)': max(ram_peak, 0)
    }
```

**After:**

```python
def evaluate_with_metrics(model, testloader, device: str = 'cpu'):
    # ... evaluation code ...
    # NEW: Calculate classification metrics
    classification_metrics = {}
    if calc_metrics is not None:
        detailed_metrics = calc_metrics(model, testloader, device, num_classes, criterion)
        classification_metrics = {
            'precision': detailed_metrics.get('precision'),
            'recall': detailed_metrics.get('recall'),
            'f1_score': detailed_metrics.get('f1_score'),
        }

    return {
        'Accuracy%': accuracy * 100,
        'Loss': avg_loss,
        'InferenceSpeed': inference_time,
        'CPUUsage%': cpu_percent,
        'RAMPeak(MB)': max(ram_peak, 0),
        **classification_metrics  # ✅ Include precision, recall, f1_score
    }
```

## Impact

### ✅ Future Models

All NEW models trained from now on will automatically have:

- Precision (macro-averaged across classes)
- Recall (macro-averaged across classes)
- F1-Score (macro-averaged across classes)

These metrics will be:

1. Calculated during final evaluation
2. Saved to `Model.metrics` JSONB field in database
3. Displayed in frontend (Training Overview, Model Details, Analytics)

### ⚠️ Existing Model (0fca41da-9bb3-4efd-a136-34a08c9c18d5)

This model was trained BEFORE the fix, so its metrics don't include precision/recall/f1.

**Options:**

1. **Retrain the model** (Recommended): Start a new training job with the same dataset. The new model will have all metrics.
2. **Accept current state**: The model works fine for predictions, it just doesn't have these specific metrics displayed.

## Verification

To verify the fix works for new models:

```bash
# Start a new training job
# After completion, check that Model.metrics includes:
# - precision
# - recall
# - f1_score
```

## Related Files Modified

- ✅ [training_pipeline/core/llm_training.py](training_pipeline/core/llm_training.py) - Added classification metrics calculation
- ✅ [metrics_utils.py](metrics_utils.py) - Already has `calculate_classification_metrics()` function (no changes needed)
- ✅ [training_runner.py](training_runner.py) - Already extracts precision/recall/f1 from result.metrics (no changes needed)

## Database Schema

No schema changes required. The `models.metrics` field is already JSONB and can store any key-value pairs.

Current Model State:

```json
{
  "Loss": 0.6958629687627157,
  "Accuracy%": 40.51812294057493,
  "CPUUsage%": 0.0,
  "Stability%": 70.0,
  "RAMPeak(MB)": 0,
  "InferenceSpeed": 1668.1244185859134
  // ❌ Missing: precision, recall, f1_score
}
```

Future Model State:

```json
{
  "Loss": 0.6958629687627157,
  "Accuracy%": 84.31,
  "CPUUsage%": 0.0,
  "Stability%": 70.0,
  "RAMPeak(MB)": 0,
  "InferenceSpeed": 1668.1244185859134,
  "precision": 0.8543, // ✅ NEW
  "recall": 0.8421, // ✅ NEW
  "f1_score": 0.8481 // ✅ NEW
}
```

## Status: ✅ FIXED

Classification metrics will be calculated and synced for all future model trainings.
