"""
Task Type Inference
Smart detection of ML task type based on dataset characteristics
"""
import re
from pathlib import Path
from typing import Optional, Dict, Any, List


def detect_supported_tasks_at_upload(
    dataset_path: Path,
    domain: str,
    structure: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Detect supported tasks at dataset upload time based on file structure
    
    Simple rules:
    - ZIP (Vision) with class folders → Classification only
    - ZIP with annotations → Detection only
    - CSV with categorical target → Classification + Regression
    - CSV with continuous target → Regression only
    
    Args:
        dataset_path: Path to uploaded dataset
        domain: Dataset domain (vision, tabular, etc.)
        structure: Parsed structure information
        
    Returns:
        Dictionary with supported_tasks and recommended_task
    """
    supported_tasks = []
    recommended_task = None
    reasoning = []
    
    if domain == 'vision':
        # Check for class-based folder structure
        if 'classes' in structure and isinstance(structure.get('classes'), list):
            num_classes = len(structure['classes'])
            if num_classes >= 2:
                supported_tasks = ['classification']
                recommended_task = 'classification'
                reasoning.append(f"Vision dataset with {num_classes} class folders")
        
        # Check for annotations (detection)
        elif 'annotations' in structure or 'labels' in structure:
            supported_tasks = ['detection']
            recommended_task = 'detection'
            reasoning.append("Dataset contains annotation files for object detection")
        
        # Check for masks (segmentation → detection)
        elif 'masks' in structure:
            supported_tasks = ['detection']
            recommended_task = 'detection'
            reasoning.append("Dataset contains mask files for segmentation")
        
        # Unknown structure - default to classification
        else:
            supported_tasks = ['classification']
            recommended_task = 'classification'
            reasoning.append("Default vision task")
    
    elif domain == 'tabular':
        # CSV files can support multiple tasks
        target_info = structure.get('target_column')
        
        if target_info:
            unique_values = target_info.get('unique_values', 0)
            dtype = target_info.get('dtype', '').lower()
            
            # Categorical target (2-100 unique values)
            if 2 <= unique_values <= 100:
                supported_tasks = ['classification', 'regression']
                recommended_task = 'classification'
                reasoning.append(f"Target column has {unique_values} unique values (categorical)")
            
            # Continuous target (many unique values or float type)
            elif unique_values > 100 or 'float' in dtype:
                supported_tasks = ['regression']
                recommended_task = 'regression'
                reasoning.append("Target column is continuous (regression)")
            
            # Binary classification
            elif unique_values == 2:
                supported_tasks = ['classification']
                recommended_task = 'classification'
                reasoning.append("Binary classification task")
            
            else:
                # Default: both possible
                supported_tasks = ['classification', 'regression']
                recommended_task = 'classification'
                reasoning.append("Target column supports multiple tasks")
        else:
            # No target info - allow both
            supported_tasks = ['classification', 'regression']
            recommended_task = 'classification'
            reasoning.append("Tabular dataset (task depends on target column)")
    
    else:
        # Other domains - default to classification
        supported_tasks = ['classification']
        recommended_task = 'classification'
        reasoning.append(f"Default task for {domain} domain")
    
    return {
        'supported_tasks': supported_tasks,
        'recommended_task': recommended_task,
        'reasoning': reasoning
    }


def infer_task_from_dataset(dataset: Dict[str, Any]) -> str:
    """
    Infer the most appropriate task type based on dataset characteristics
    
    Args:
        dataset: Dataset dictionary with metadata
        
    Returns:
        Task type: 'classification', 'regression', 'detection', 'clustering'
    """
    name = dataset.get('name', '').lower()
    domain = dataset.get('domain', '').lower()
    structure = dataset.get('structure', {})
    description = dataset.get('description', '').lower()
    
    # Priority 1: Explicit task keywords in name
    if any(kw in name for kw in ['classification', 'classifier', 'classes']):
        return 'classification'
    
    if any(kw in name for kw in ['regression', 'prediction', 'forecast', 'price', 'sales']):
        return 'regression'
    
    if any(kw in name for kw in ['detection', 'detect', 'yolo', 'rcnn', 'ssd']):
        return 'detection'
    
    if any(kw in name for kw in ['segmentation', 'segment', 'mask', 'semantic']):
        return 'detection'  # We'll map segmentation to detection for now
    
    # Priority 2: Known dataset patterns
    if any(kw in name for kw in [
        'mnist', 'cifar', 'imagenet', 'fashion', 'flower', 'animal', 'pet',
        'cat', 'dog', 'bird', 'plant', 'leaf', 'fruit', 'vehicle',
        'credit', 'fraud', 'spam', 'sentiment', 'emotion', 'toxicity',
        'churn', 'diagnosis', 'disease', 'symptom'
    ]):
        return 'classification'
    
    if any(kw in name for kw in [
        'house', 'housing', 'apartment', 'stock', 'energy', 'consumption',
        'temperature', 'weather', 'sales', 'revenue', 'income', 'salary',
        'age', 'rating', 'score', 'time_series', 'timeseries'
    ]):
        return 'regression'
    
    if any(kw in name for kw in [
        'face', 'person', 'people', 'pedestrian', 'vehicle',
        'object', 'coco', 'voc', 'anomaly', 'defect', 'crack'
    ]):
        return 'detection'
    
    # Priority 3: Dataset structure analysis (for vision datasets)
    if domain == 'vision' and structure:
        # Check if it has class folders (classification)
        if 'classes' in structure and isinstance(structure.get('classes'), list):
            num_classes = len(structure.get('classes', []))
            if num_classes >= 2:
                return 'classification'
        
        # Check for annotation files (detection)
        if any(key in structure for key in ['annotations', 'labels', 'bboxes']):
            return 'detection'
        
        # Check for mask/segmentation structure
        if any(key in structure for key in ['masks', 'segmentation']):
            return 'detection'
    
    # Priority 4: Description keywords
    if description:
        if any(kw in description for kw in ['classify', 'categorize', 'category', 'label']):
            return 'classification'
        
        if any(kw in description for kw in ['predict', 'estimate', 'forecast', 'continuous']):
            return 'regression'
        
        if any(kw in description for kw in ['detect', 'locate', 'bounding box', 'object']):
            return 'detection'
    
    # Default: classification for vision, regression for tabular
    if domain == 'vision':
        return 'classification'
    elif domain == 'tabular':
        # Could be either - default to classification
        return 'classification'
    
    return 'classification'


def get_task_confidence(dataset: Dict[str, Any]) -> Dict[str, float]:
    """
    Get confidence scores for each task type
    
    Returns:
        Dictionary with task types and confidence scores (0-1)
    """
    name = dataset.get('name', '').lower()
    domain = dataset.get('domain', '').lower()
    structure = dataset.get('structure', {})
    
    scores = {
        'classification': 0.3,  # Default baseline
        'regression': 0.2,
        'detection': 0.1,
        'clustering': 0.1,
    }
    
    # Boost scores based on keywords
    classification_keywords = ['classification', 'classifier', 'mnist', 'cifar', 'imagenet', 'fraud', 'sentiment']
    regression_keywords = ['regression', 'prediction', 'forecast', 'price', 'sales', 'temperature']
    detection_keywords = ['detection', 'detect', 'yolo', 'rcnn', 'face', 'object']
    
    for kw in classification_keywords:
        if kw in name:
            scores['classification'] += 0.3
    
    for kw in regression_keywords:
        if kw in name:
            scores['regression'] += 0.3
    
    for kw in detection_keywords:
        if kw in name:
            scores['detection'] += 0.3
    
    # Boost based on structure
    if domain == 'vision' and structure:
        if 'classes' in structure:
            scores['classification'] += 0.2
        if 'annotations' in structure:
            scores['detection'] += 0.2
    
    # Normalize scores
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}
    
    return scores


# Example usage and test cases
EXAMPLE_DATASETS = {
    'mnist_classification': {
        'name': 'MNIST Handwritten Digits',
        'domain': 'vision',
        'structure': {'classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']},
        'expected_task': 'classification'
    },
    'house_prices': {
        'name': 'House Prices Prediction',
        'domain': 'tabular',
        'structure': {'target': 'price', 'features': ['sqft', 'bedrooms', 'location']},
        'expected_task': 'regression'
    },
    'face_detection': {
        'name': 'Face Detection Dataset',
        'domain': 'vision',
        'structure': {'annotations': 'yolo'},
        'expected_task': 'detection'
    },
    'credit_risk': {
        'name': 'Credit Risk Assessment',
        'domain': 'tabular',
        'structure': {'classes': ['approved', 'rejected']},
        'expected_task': 'classification'
    },
    'flower_types': {
        'name': 'Flower Classification',
        'domain': 'vision',
        'structure': {'classes': ['rose', 'tulip', 'daisy', 'sunflower']},
        'expected_task': 'classification'
    },
    'stock_forecast': {
        'name': 'Stock Price Forecast',
        'domain': 'tabular',
        'description': 'Predict future stock prices',
        'expected_task': 'regression'
    },
}


if __name__ == '__main__':
    print("Testing Task Inference")
    print("=" * 60)
    
    for dataset_id, dataset_info in EXAMPLE_DATASETS.items():
        inferred = infer_task_from_dataset(dataset_info)
        expected = dataset_info['expected_task']
        status = '✓' if inferred == expected else '✗'
        
        print(f"\n{status} {dataset_info['name']}")
        print(f"  Expected: {expected}")
        print(f"  Inferred: {inferred}")
        
        confidences = get_task_confidence(dataset_info)
        print(f"  Confidence: {confidences}")
