"""
Dataset Validation Module

Validates dataset structure and requirements based on domain and task type.
Ensures datasets meet minimum requirements for training.
"""
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class DatasetValidator:
    """Validates dataset structure and requirements"""

    # Minimum requirements
    MIN_SAMPLES_CLASSIFICATION = 50
    MIN_SAMPLES_REGRESSION = 50
    MIN_SAMPLES_CLUSTERING = 50
    MIN_SAMPLES_DETECTION = 20
    MIN_SAMPLES_AUTOML = 100  # AutoML requires more data for reliable training
    MIN_CLASSES = 2
    MIN_IMAGES_PER_CLASS = 10
    VALID_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif'}
    MIN_IMAGE_SIZE = (32, 32)

    @staticmethod
    def validate_dataset(
        dataset_path: Path,
        domain: str,
        task: Optional[str] = None
    ) -> Dict:
        """
        Main validation entry point

        Args:
            dataset_path: Path to dataset directory
            domain: Dataset domain (vision, tabular)
            task: Optional task type (classification, regression, clustering, detection)

        Returns:
            Dict with validation results and metadata

        Raises:
            ValidationError: If validation fails
        """
        if domain == "vision":
            return DatasetValidator._validate_vision_dataset(dataset_path, task)
        elif domain == "tabular":
            return DatasetValidator._validate_tabular_dataset(dataset_path, task)
        else:
            raise ValidationError(
                f"Domain '{domain}' is not currently supported. "
                f"Only 'vision' and 'tabular' datasets are supported."
            )

    @staticmethod
    def _validate_vision_dataset(dataset_path: Path, task: Optional[str]) -> Dict:
        """Validate vision dataset structure"""

        # Check for train/test split
        has_train = (dataset_path / "train").exists()
        has_test = (dataset_path / "test").exists()

        if not has_train or not has_test:
            raise ValidationError(
                "Vision dataset must have 'train' and 'test' directories. "
                "Please split your dataset before uploading."
            )

        train_dir = dataset_path / "train"
        test_dir = dataset_path / "test"

        if task == "classification":
            return DatasetValidator._validate_vision_classification(train_dir, test_dir)
        elif task == "detection":
            return DatasetValidator._validate_vision_detection(train_dir, test_dir)
        else:
            # Generic vision validation
            train_images = DatasetValidator._count_images(train_dir)
            test_images = DatasetValidator._count_images(test_dir)

            if train_images == 0:
                raise ValidationError(
                    "No valid images found in train directory")
            if test_images == 0:
                raise ValidationError(
                    "No valid images found in test directory")

            return {
                "valid": True,
                "train_samples": train_images,
                "test_samples": test_images,
                "total_samples": train_images + test_images,
                "has_split": True,
                "message": "Vision dataset validated successfully"
            }

    @staticmethod
    def _validate_vision_classification(train_dir: Path, test_dir: Path) -> Dict:
        """Validate vision classification dataset (ImageFolder structure)"""

        # Get class folders in train directory
        train_classes = [d.name for d in train_dir.iterdir(
        ) if d.is_dir() and not d.name.startswith('.')]
        test_classes = [d.name for d in test_dir.iterdir(
        ) if d.is_dir() and not d.name.startswith('.')]

        if len(train_classes) < DatasetValidator.MIN_CLASSES:
            raise ValidationError(
                f"Classification requires at least {DatasetValidator.MIN_CLASSES} classes. "
                f"Found {len(train_classes)} in train directory."
            )

        if set(train_classes) != set(test_classes):
            raise ValidationError(
                "Train and test directories must have the same classes. "
                f"Train: {train_classes}, Test: {test_classes}"
            )

        # Validate each class
        class_counts = {}
        total_train = 0
        total_test = 0

        for class_name in train_classes:
            train_class_dir = train_dir / class_name
            test_class_dir = test_dir / class_name

            train_images = DatasetValidator._count_images(train_class_dir)
            test_images = DatasetValidator._count_images(test_class_dir)

            if train_images < DatasetValidator.MIN_IMAGES_PER_CLASS:
                raise ValidationError(
                    f"Class '{class_name}' has only {train_images} training images. "
                    f"Minimum required: {DatasetValidator.MIN_IMAGES_PER_CLASS}"
                )

            class_counts[class_name] = {
                "train": train_images,
                "test": test_images,
                "total": train_images + test_images
            }

            total_train += train_images
            total_test += test_images

        return {
            "valid": True,
            "task": "classification",
            "num_classes": len(train_classes),
            "classes": train_classes,
            "class_distribution": class_counts,
            "train_samples": total_train,
            "test_samples": total_test,
            "total_samples": total_train + total_test,
            "has_split": True,
            "message": f"Classification dataset validated: {len(train_classes)} classes, {total_train + total_test} images"
        }

    @staticmethod
    def _validate_vision_detection(train_dir: Path, test_dir: Path) -> Dict:
        """Validate object detection dataset (images + annotations)"""

        # Check for images and annotations
        train_images = DatasetValidator._count_images(train_dir)
        test_images = DatasetValidator._count_images(test_dir)

        if train_images < DatasetValidator.MIN_SAMPLES_DETECTION:
            raise ValidationError(
                f"Detection requires at least {DatasetValidator.MIN_SAMPLES_DETECTION} training images. "
                f"Found {train_images}"
            )

        # Check for annotation files (common formats)
        train_annotations = (
            (train_dir / "annotations.json").exists() or
            (train_dir / "labels").exists() or
            any(train_dir.glob("*.xml"))
        )

        test_annotations = (
            (test_dir / "annotations.json").exists() or
            (test_dir / "labels").exists() or
            any(test_dir.glob("*.xml"))
        )

        if not train_annotations:
            raise ValidationError(
                "No annotation files found in train directory. "
                "Expected: annotations.json, labels/ folder, or .xml files"
            )

        if not test_annotations:
            raise ValidationError(
                "No annotation files found in test directory"
            )

        return {
            "valid": True,
            "task": "detection",
            "train_samples": train_images,
            "test_samples": test_images,
            "total_samples": train_images + test_images,
            "has_split": True,
            "has_annotations": True,
            "message": f"Detection dataset validated: {train_images + test_images} annotated images"
        }

    @staticmethod
    def _validate_tabular_dataset(dataset_path: Path, task: Optional[str]) -> Dict:
        """Validate tabular dataset structure (CSV or Parquet)"""
        from file_utils import get_data_files

        # Check for train/test files (CSV or Parquet)
        train_files = (
            list(dataset_path.glob("train.csv")) +
            list(dataset_path.glob("train.parquet")) +
            list(dataset_path.glob("train.pq"))
        )
        test_files = (
            list(dataset_path.glob("test.csv")) +
            list(dataset_path.glob("test.parquet")) +
            list(dataset_path.glob("test.pq"))
        )

        # Also check for single data file with train/test split indicator
        single_file = None
        data_files = get_data_files(dataset_path)

        # Filter out train/test files to find single file
        excluded_names = ['train.csv', 'test.csv',
                          'train.parquet', 'test.parquet', 'train.pq', 'test.pq']
        for data_file in data_files:
            if data_file.name not in excluded_names:
                single_file = data_file
                break

        if not (train_files and test_files) and not single_file:
            raise ValidationError(
                "Tabular dataset must have either 'train' and 'test' files (CSV or Parquet), "
                "or a single data file with train/test split indicator column. "
                "Please split your dataset before uploading."
            )

        if train_files and test_files:
            return DatasetValidator._validate_split_files(train_files[0], test_files[0], task)
        else:
            return DatasetValidator._validate_single_file(single_file, task)

    @staticmethod
    def _validate_split_files(train_file: Path, test_file: Path, task: Optional[str]) -> Dict:
        """Validate train/test data files (CSV or Parquet)"""
        import pandas as pd

        from file_utils import read_dataframe

        # Read train and test files
        train_df = read_dataframe(train_file)
        test_df = read_dataframe(test_file)

        train_headers = list(train_df.columns)
        test_headers = list(test_df.columns)
        train_rows = len(train_df)
        test_rows = len(test_df)

        # Validate headers match
        if train_headers != test_headers:
            raise ValidationError(
                "Train and test files must have the same columns"
            )

        # Validate minimum samples
        min_samples = DatasetValidator.MIN_SAMPLES_CLASSIFICATION
        if task in ["classification", "regression", "clustering"]:
            if train_rows < min_samples:
                raise ValidationError(
                    f"{task.capitalize()} requires at least {min_samples} training samples. "
                    f"Found {train_rows}"
                )

        num_features = len(train_headers)

        # Task-specific validation
        if task == "classification":
            # Assume last column is target
            if num_features < 2:
                raise ValidationError(
                    "Classification requires at least 1 feature and 1 target column")
        elif task == "regression":
            if num_features < 2:
                raise ValidationError(
                    "Regression requires at least 1 feature and 1 target column")
        elif task == "clustering":
            if num_features < 2:
                raise ValidationError(
                    "Clustering requires at least 2 feature columns")

        return {
            "valid": True,
            "task": task,
            "train_samples": train_rows,
            "test_samples": test_rows,
            "total_samples": train_rows + test_rows,
            "num_features": num_features,
            "columns": train_headers,
            "has_split": True,
            "message": f"Tabular dataset validated: {train_rows + test_rows} samples, {num_features} features"
        }

    # Train/Test split ratio for single CSV/Parquet files (configurable)
    # Format: (train_ratio, test_ratio) where train_ratio + test_ratio = 1.0
    DEFAULT_TRAIN_TEST_SPLIT = (0.8, 0.2)  # 80% train, 20% test

    @staticmethod
    def _validate_single_file(data_file: Path, task: Optional[str]) -> Dict:
        """Validate single CSV or Parquet file (with or without split indicator)"""
        import pandas as pd

        from file_utils import read_dataframe

        # Read file using format-agnostic reader
        df = read_dataframe(data_file)
        headers = list(df.columns)
        total_rows = len(df)

        if total_rows == 0:
            raise ValidationError("Data file is empty (no data rows)")

        # Check for split indicator column (optional)
        split_indicators = ['split', 'set', 'subset', 'train_test']
        split_col = None
        for col in split_indicators:
            if col in [h.lower() for h in headers]:
                split_col = headers[[h.lower()
                                     for h in headers].index(col.lower())]
                break

        if split_col:
            # Has split indicator - use it
            train_count = df[split_col].str.lower().isin(
                ['train', 'training']).sum()
            test_count = df[split_col].str.lower().isin(
                ['test', 'testing']).sum()

            if train_count == 0 or test_count == 0:
                raise ValidationError(
                    f"Split indicator column must contain both train and test samples. "
                    f"Found {train_count} train, {test_count} test"
                )

            return {
                "valid": True,
                "task": task,
                "train_samples": int(train_count),
                "test_samples": int(test_count),
                "total_samples": total_rows,
                "num_features": len(headers) - 1,  # Exclude split column
                "columns": headers,
                "has_split": True,
                "split_column": split_col,
                "message": f"Tabular dataset validated: {total_rows} samples, {len(headers)-1} features (pre-split)"
            }
        else:
            # No split indicator - will split automatically using DEFAULT_TRAIN_TEST_SPLIT
            train_ratio, test_ratio = DatasetValidator.DEFAULT_TRAIN_TEST_SPLIT
            train_count = int(total_rows * train_ratio)
            test_count = total_rows - train_count

            return {
                "valid": True,
                "task": task,
                "train_samples": train_count,  # Estimated
                "test_samples": test_count,    # Estimated
                "total_samples": total_rows,
                "num_features": len(headers),
                "columns": headers,
                "has_split": False,  # Will be split automatically
                "auto_split": True,
                "split_ratio": f"{int(train_ratio*100)}/{int(test_ratio*100)}",
                "message": f"Tabular dataset validated: {total_rows} samples, {len(headers)} features (will auto-split {int(train_ratio*100)}/{int(test_ratio*100)})"
            }

    @staticmethod
    def _count_images(directory: Path) -> int:
        """Count valid image files in directory"""
        count = 0
        for file in directory.rglob("*"):
            if file.suffix.lower() in DatasetValidator.VALID_IMAGE_FORMATS:
                # Try to verify it's a valid image
                try:
                    with Image.open(file) as img:
                        width, height = img.size
                        if width >= DatasetValidator.MIN_IMAGE_SIZE[0] and \
                           height >= DatasetValidator.MIN_IMAGE_SIZE[1]:
                            count += 1
                except Exception:
                    # Skip invalid images
                    continue
        return count

    @staticmethod
    def validate_automl_requirements(csv_path: Path, target_column: str) -> Dict:
        """
        Validate CSV dataset meets AutoML requirements

        Args:
            csv_path: Path to CSV file
            target_column: Target column name for prediction

        Returns:
            Dict with validation results

        Raises:
            ValidationError: If validation fails
        """
        if not csv_path.exists():
            raise ValidationError(f"CSV file not found: {csv_path}")

        try:
            # Read CSV to analyze
            import pandas as pd
            df = pd.read_csv(csv_path)

            # Check minimum rows
            if len(df) < DatasetValidator.MIN_SAMPLES_AUTOML:
                raise ValidationError(
                    f"AutoML requires minimum {DatasetValidator.MIN_SAMPLES_AUTOML} rows. "
                    f"Found: {len(df)} rows"
                )

            # Check target column exists
            if target_column not in df.columns:
                raise ValidationError(
                    f"Target column '{target_column}' not found in dataset. "
                    f"Available columns: {', '.join(df.columns)}"
                )

            # Check target has valid distribution
            target_unique = df[target_column].nunique()
            if target_unique < 2:
                raise ValidationError(
                    f"Target column '{target_column}' must have at least 2 unique values. "
                    f"Found: {target_unique}"
                )

            # Check if target is all NaN
            if df[target_column].isna().all():
                raise ValidationError(
                    f"Target column '{target_column}' contains only missing values"
                )

            # Determine problem type
            target_dtype = df[target_column].dtype
            is_numeric = pd.api.types.is_numeric_dtype(target_dtype)

            if is_numeric and target_unique > 20:
                problem_type = "regression"
            else:
                problem_type = "classification"

            # Calculate missing value percentage
            missing_pct = (df.isna().sum().sum() /
                           (len(df) * len(df.columns))) * 100

            return {
                "valid": True,
                "rows": len(df),
                "columns": len(df.columns),
                "target_column": target_column,
                "target_classes": target_unique,
                "problem_type": problem_type,
                "missing_percentage": round(missing_pct, 2),
                "column_names": list(df.columns),
                "message": f"AutoML validation passed: {len(df)} rows, {len(df.columns)} columns, {problem_type} task"
            }

        except pd.errors.EmptyDataError:
            raise ValidationError("CSV file is empty")
        except pd.errors.ParserError as e:
            raise ValidationError(f"CSV parsing error: {str(e)}")
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Validation error: {str(e)}")
