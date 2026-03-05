"""
Core LLM Training Module
Handles LLM-powered CNN generation and training
"""

import importlib.util
import json
import os
import random
import re
# Import metrics utilities
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from groq import Groq
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from metrics_utils import evaluate_model_with_metrics as calc_metrics
except ImportError:
    calc_metrics = None
    print("[WARNING] metrics_utils not found, metrics calculation will be limited")


# === LLM System Prompt Template ===
SYSTEM_PROMPT_TEMPLATE = """
You are an expert in designing LIGHTWEIGHT Convolutional Neural Networks (CNNs) for image classification tasks using PyTorch.

Dataset information:
- Input shape: ({input_channels}, {H}, {W})
- Number of classes: {{num_classes}}
- Number of training images: {{num_images}}

Your task is to generate a complete, working PyTorch CNN class named 'GeneratedCNN' that:
1. Inherits from nn.Module
2. Has __init__ and forward methods
3. Uses the provided hyperparameters (learning rate, dropout, etc.)
4. Is appropriate for the dataset size and complexity
5. Returns logits (no softmax in forward)

CRITICAL CONSTRAINTS FOR MEMORY EFFICIENCY:
- **MAXIMUM 32 channels** in first conv layer
- **MAXIMUM 64 channels** in second conv layer (if needed)
- **MAXIMUM 128 channels** in any layer
- **Use MaxPool2d after EVERY conv block** to reduce dimensions quickly
- **Limit to 2-3 conv layers maximum** for small datasets
- **Keep FC layers small** (max 256 neurons in hidden layer)

Architectural guidelines:
- Use Conv2d layers with kernel_size=3, padding=1
- Include BatchNorm2d and dropout for regularization
- Use MaxPool2d(2,2) to reduce spatial dimensions aggressively
- After convolutions, flatten and use Linear layers
- Pattern: [Conv(32)->BN->ReLU->Pool] -> [Conv(64)->BN->ReLU->Pool] -> Flatten -> FC(128) -> FC(num_classes)
- For small datasets (<10k images), use only 2 conv layers with 32 and 64 channels
- For larger datasets, add one more conv layer with max 128 channels

Height/width calculations (for reference):
- After one 3x3 conv (padding=1): same size
- After MaxPool2d(2,2): H -> {H2}, W -> {W2}
- After two pools: H -> {H4}, W -> {W4}

Output format: Provide ONLY the Python class code, wrapped in ```python ``` markers.
Do NOT include training loops, optimizer code, or data loading.
Just the GeneratedCNN class definition that follows the memory constraints above.
"""


def is_valid_image_file(path: str) -> bool:
    """Check if file is a valid image (excludes macOS metadata files)"""
    VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    name = os.path.basename(path).lower()
    if name.startswith('._') or name == '.ds_store':
        return False
    return name.endswith(VALID_EXTS)


def find_class_root(root: Path) -> Path:
    """Find the directory level that contains class subfolders with image files"""
    entries = [root / e for e in os.listdir(root)]
    for e in entries:
        if e.is_dir():
            files = os.listdir(e)
            if any(fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')) for fname in files):
                return root
    for e in entries:
        if e.is_dir():
            subentries = [e / s for s in os.listdir(e)]
            for s in subentries:
                if s.is_dir():
                    files = os.listdir(s)
                    if any(fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')) for fname in files):
                        return e
    return root


def build_transforms(resize_to: tuple):
    """Build training and validation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def load_dataset(dataset_path: Path, resize_to: tuple = (224, 224), num_workers: int = 0):
    """
    Load dataset from path

    Returns:
        tuple: (train_dataset, val_dataset, testloader, image_shape, num_classes, num_images)
    """
    train_t, val_t = build_transforms(resize_to)
    class_root = find_class_root(dataset_path)

    print(f"[LLM Training] Using class root: {class_root}")

    full_dataset = ImageFolder(
        root=class_root,
        transform=val_t,
        is_valid_file=is_valid_image_file
    )

    if len(full_dataset) == 0:
        raise RuntimeError(
            "No valid images found. Check dataset root and file extensions.")

    # Random split 80/20 for train/validation
    num_images_total = len(full_dataset)
    val_ratio = 0.20
    val_size = int(num_images_total * val_ratio)
    train_size = num_images_total - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply different transforms
    train_dataset.dataset.transform = train_t
    val_dataset.dataset.transform = val_t

    # Get metadata
    sample_img, _ = full_dataset[0]
    image_shape = tuple(sample_img.shape)
    num_classes = len(full_dataset.classes)

    # Create test loader
    testloader = DataLoader(
        val_dataset,
        batch_size=min(1000, len(val_dataset)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(
        f"[LLM Training] Image shape: {image_shape}, Classes: {num_classes}, Training images: {train_size}")

    return train_dataset, val_dataset, testloader, image_shape, num_classes, train_size


def get_hyperparameter_suggestions(iteration: int, best_acc: float, last_config: Optional[Dict] = None) -> Dict[str, Any]:
    """Get hyperparameter suggestions based on current progress"""
    if iteration == 0 or best_acc < 0.5:
        return {
            'lr': 1e-3,
            'batch_size': 32,
            'optimizer': 'Adam',
            'dropout_rate': 0.2,
            'epochs': 3
        }
    elif best_acc < 0.8:
        return {
            'lr': random.choice([1e-3, 5e-4, 2e-4]),
            'batch_size': random.choice([32, 64]),
            'optimizer': random.choice(['Adam', 'SGD']),
            'dropout_rate': random.choice([0.1, 0.2]),
            'epochs': random.choice([4, 5])
        }
    else:
        return {
            'lr': random.choice([5e-4, 2e-4]),
            'batch_size': random.choice([16, 32]),
            'optimizer': 'Adam',
            'dropout_rate': random.choice([0.1, 0.15]),
            'epochs': 5
        }


def extract_python_code(text: str) -> str:
    """Extract Python code from LLM response"""
    # Try python-specific code block
    python_pattern = r'```python\s*(.*?)\s*```'
    match = re.search(python_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic code block
    generic_pattern = r'```\s*(.*?)\s*```'
    match = re.search(generic_pattern, text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if 'class GeneratedCNN' in code:
            return code

    # Fallback: extract lines that look like code
    lines = text.split('\n')
    code_lines = []
    in_class = False
    for line in lines:
        if 'class GeneratedCNN' in line:
            in_class = True
        if in_class:
            code_lines.append(line)

    if code_lines:
        return '\n'.join(code_lines)

    raise ValueError("Could not extract Python code from LLM response")


def get_user_prompt(image_shape: tuple, num_classes: int, num_images: int,
                    hyperparams: Dict, previous_accuracy: Optional[float] = None,
                    previous_error: Optional[str] = None, previous_config: Optional[Dict] = None) -> str:
    """Generate user prompt for LLM"""
    prompt = f"""
Generate a CNN architecture for image classification.

Dataset details:
- Input shape: {image_shape}
- Number of classes: {num_classes}
- Number of training samples: {num_images}

Hyperparameters to use:
- Dropout rate: {hyperparams.get('dropout_rate', 0.2)}
- (Learning rate {hyperparams.get('lr')}, batch size {hyperparams.get('batch_size')}, optimizer {hyperparams.get('optimizer')} will be used during training)
"""

    if previous_accuracy is not None:
        prompt += f"\nPrevious best accuracy: {previous_accuracy:.4f}"
        if previous_config:
            prompt += f"\nPrevious config: {previous_config}"

    if previous_error:
        prompt += f"\n\nPrevious attempt failed with error:\n{previous_error}\n\nPlease fix the architecture to avoid this error."
    else:
        prompt += "\n\nPlease generate an improved architecture."

    prompt += "\n\nProvide ONLY the GeneratedCNN class code."

    return prompt


def call_llm(system_prompt: str, user_prompt: str, api_key: str) -> str:
    """
    Call GROQ API to generate CNN architecture

    Args:
        system_prompt: System prompt for LLM
        user_prompt: User prompt for LLM
        api_key: GROQ API key

    Returns:
        Generated architecture code as string
    """
    if not api_key:
        raise ValueError("GROQ_API_KEY is required but not provided")

    try:
        # Try creating client without extra parameters (newer groq versions)
        client = Groq(api_key=api_key)
    except TypeError:
        # Fallback for older versions
        from groq import Client
        client = Client(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=2048
    )

    return response.choices[0].message.content


def load_model_from_code(model_code: str, model_file: Path):
    """Load model class from generated code"""
    # Ensure imports
    if 'import torch' not in model_code:
        model_code = 'import torch\n' + model_code
    if 'import torch.nn as nn' not in model_code:
        model_code = 'import torch.nn as nn\n' + model_code
    if 'import torch.nn.functional as F' not in model_code and 'F.' in model_code:
        model_code = 'import torch.nn.functional as F\n' + model_code

    # Save to file
    with open(model_file, 'w') as f:
        f.write(model_code)

    # Load module
    spec = importlib.util.spec_from_file_location(
        "generated_model", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'GeneratedCNN'):
        raise AttributeError(
            "Generated code must contain a class named 'GeneratedCNN'")

    return module.GeneratedCNN


def train_and_evaluate(model_cls, train_dataset, val_dataset, hyperparams: Dict, device: str = 'cpu', num_classes: int = 10):
    """Train model and return validation accuracy and metrics"""
    model = model_cls().to(device)

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=(device == 'cuda')
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=min(1000, len(val_dataset)),
        shuffle=False,
        num_workers=0
    )

    # Setup optimizer
    optimizer_name = hyperparams.get('optimizer', 'Adam')
    lr = hyperparams.get('lr', 1e-3)

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    # Training loop
    epochs = hyperparams.get('epochs', 3)
    final_loss = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

            # Limit batches per epoch for faster iteration
            if batch_count > 300:
                break

        final_loss = running_loss / max(1, batch_count)
        print(f"  Epoch {epoch + 1}/{epochs}, Loss: {final_loss:.4f}")

    # Comprehensive validation with metrics
    if calc_metrics is not None:
        metrics = calc_metrics(model, val_loader, device,
                               num_classes, criterion)
        return {
            'accuracy': metrics['accuracy'],
            'loss': metrics['loss'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
        }
    else:
        # Fallback: simple accuracy calculation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return {
            'accuracy': correct / total if total > 0 else 0.0,
            'loss': val_loss / len(val_loader) if len(val_loader) > 0 else 0.0,
            'precision': None,
            'recall': None,
            'f1_score': None,
        }


def evaluate_with_metrics(model, testloader, device: str = 'cpu'):
    """Evaluate model and return comprehensive metrics including classification metrics"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    # Collect predictions and labels for classification metrics
    all_preds = []
    all_labels = []

    # Measure inference time
    start_time = time.time()

    # Track memory
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions and labels for classification metrics
            all_preds.append(predicted)
            all_labels.append(labels)

    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    accuracy = correct / total
    avg_loss = total_loss / len(testloader)
    inference_time = (end_time - start_time) * 1000 / total  # ms per sample
    cpu_percent = process.cpu_percent()
    ram_peak = mem_after - mem_before

    # Calculate classification metrics if calc_metrics is available
    classification_metrics = {}
    if calc_metrics is not None:
        try:
            # Concatenate all predictions and labels
            all_preds_tensor = torch.cat(all_preds)
            all_labels_tensor = torch.cat(all_labels)

            # Determine number of classes
            num_classes = len(torch.unique(all_labels_tensor))

            # Use metrics_utils to calculate classification metrics
            detailed_metrics = calc_metrics(
                model, testloader, device, num_classes, criterion)
            classification_metrics = {
                'precision': detailed_metrics.get('precision'),
                'recall': detailed_metrics.get('recall'),
                'f1_score': detailed_metrics.get('f1_score'),
            }
        except Exception as e:
            print(f"[WARNING] Could not calculate classification metrics: {e}")
            classification_metrics = {
                'precision': None,
                'recall': None,
                'f1_score': None,
            }
    else:
        classification_metrics = {
            'precision': None,
            'recall': None,
            'f1_score': None,
        }

    return {
        'Accuracy%': accuracy * 100,
        'Loss': avg_loss,
        'InferenceSpeed': inference_time,
        'CPUUsage%': cpu_percent,
        'RAMPeak(MB)': max(ram_peak, 0),
        **classification_metrics  # Include precision, recall, f1_score
    }


def run_llm_training(
    dataset_path: Path,
    model_id: str,
    groq_api_key: str,
    max_iterations: int = 10,
    target_accuracy: float = 1.0,
    device: str = 'cpu',
    resize_to: tuple = (224, 224),
    num_workers: int = 0,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Run LLM-powered CNN training

    Args:
        dataset_path: Path to dataset directory
        model_id: Model ID for file naming
        groq_api_key: GROQ API key for LLM calls
        max_iterations: Maximum training iterations
        target_accuracy: Target accuracy to achieve
        device: Device to train on ('cpu', 'cuda', 'mps')
        resize_to: Image resize dimensions
        num_workers: Number of data loader workers
        progress_callback: Optional callback for progress updates

    Returns:
        Dict containing:
            - success: bool
            - best_accuracy: float
            - best_config: dict
            - model_path: Path
            - metrics: dict
            - experiment_history: list
    """
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is required for LLM training")
    print(f"[LLM Training] Starting training for model {model_id}")
    print(f"[LLM Training] Dataset: {dataset_path}")
    print(
        f"[LLM Training] Max iterations: {max_iterations}, Target: {target_accuracy}")

    # Load dataset
    train_dataset, val_dataset, testloader, image_shape, num_classes, num_images = load_dataset(
        dataset_path, resize_to, num_workers
    )

    # Prepare system prompt
    H2 = image_shape[1] // 2
    W2 = image_shape[2] // 2
    H4 = image_shape[1] // 4
    W4 = image_shape[2] // 4

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        input_channels=image_shape[0],
        H=image_shape[1],
        W=image_shape[2],
        H2=H2,
        W2=W2,
        H4=H4,
        W4=W4
    )

    # Create directories
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    results_dir = Path('results') / model_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_acc = 0.0
    best_loss = float('inf')  # Initialize to infinity
    best_model_code = ""
    best_config = None
    last_error = None
    experiment_history = []

    for iteration in range(max_iterations):
        print(
            f"\n[LLM Training] === Iteration {iteration + 1}/{max_iterations} ===")

        # Get hyperparameter suggestions
        current_config = get_hyperparameter_suggestions(
            iteration, best_acc, best_config)
        print(f"[LLM Training] Hyperparameters: {current_config}")

        try:
            # Generate CNN architecture via LLM
            user_prompt = get_user_prompt(
                image_shape, num_classes, num_images, current_config,
                None if iteration == 0 else best_acc,
                last_error, best_config
            )

            print(f"[LLM Training] Calling LLM to generate architecture...")
            raw_response = call_llm(system_prompt, user_prompt, groq_api_key)
            model_code = extract_python_code(raw_response)

            # Save generated model
            generated_model_file = models_dir / \
                f"generated_model_{model_id}.py"
            model_cls = load_model_from_code(model_code, generated_model_file)
            print(f"[LLM Training] Model code saved to {generated_model_file}")

            # Train and evaluate
            print(f"[LLM Training] Training model...")
            result = train_and_evaluate(
                model_cls, train_dataset, val_dataset, current_config, device, num_classes)

            acc = result['accuracy']
            loss = result['loss']
            precision = result.get('precision')
            recall = result.get('recall')
            f1_score = result.get('f1_score')

            print(
                f"[LLM Training] Validation - Acc: {acc:.4f}, Loss: {loss:.4f}")
            if precision is not None:
                print(
                    f"[LLM Training] Metrics - Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1_score:.4f}")

            # Track best loss (update if this loss is better)
            if loss is not None and loss < best_loss:
                best_loss = loss

            # Report progress with full metrics
            if progress_callback:
                try:
                    progress_callback({
                        'type': 'progress',
                        'iteration': iteration + 1,
                        'total_iterations': max_iterations,
                        'current_accuracy': acc,
                        'best_accuracy': max(best_acc, acc),
                        'current_loss': loss,
                        'best_loss': best_loss if best_loss != float('inf') else loss,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1_score,
                        # Include hyperparameters used for this iteration
                        'hyperparameters': current_config,
                        'status': 'training',
                        'message': f'Iteration {iteration + 1}/{max_iterations} - Acc: {acc:.4f}'
                    })
                except Exception as callback_error:
                    print(
                        f"[LLM Training] Warning: Progress callback failed: {callback_error}")

            experiment_history.append({
                'iteration': iteration + 1,
                'accuracy': acc,
                'loss': loss,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'config': current_config.copy(),
                'success': True
            })

            # Update best model
            if acc > best_acc:
                best_acc = acc
                best_model_code = model_code
                best_config = current_config.copy()
                print(f"[LLM Training] ✓ New best accuracy: {best_acc:.4f}")

            last_error = None

            # Check if target reached
            if best_acc >= target_accuracy:
                print(
                    f"[LLM Training] Target accuracy {target_accuracy} reached!")
                break

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            last_error = str(e)
            print(f"[LLM Training] ✗ Iteration failed: {e}")
            print(f"[LLM Training] Error traceback:\n{error_trace}")
            experiment_history.append({
                'iteration': iteration + 1,
                'accuracy': 0.0,
                'config': current_config.copy(),
                'success': False,
                'error': str(e),
                'traceback': error_trace
            })
            continue

    # Save results
    if best_model_code:
        best_model_file = models_dir / f"best_model_{model_id}.py"
        with open(best_model_file, 'w') as f:
            f.write(best_model_code)

        timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M")
        history_file = results_dir / f"experiment_history_{timestamp}.json"
        with open(history_file, 'w') as f:
            json.dump(experiment_history, f, indent=2)

        print(f"\n[LLM Training] === Training Complete ===")
        print(f"[LLM Training] Best accuracy: {best_acc:.4f}")
        print(f"[LLM Training] Best config: {best_config}")
        print(f"[LLM Training] Model saved to: {best_model_file}")

        # Compute final metrics (with error handling)
        metrics = {}
        try:
            model_cls = load_model_from_code(best_model_code, best_model_file)
            model = model_cls().to(device)
            metrics = evaluate_with_metrics(model, testloader, device)
            print(f"[LLM Training] Final metrics computed successfully")
        except Exception as e:
            print(
                f"[LLM Training] Warning: Could not compute final metrics: {e}")
            # Use basic metrics if evaluation fails
            metrics = {
                'Accuracy%': best_acc * 100,
                'Loss': best_loss if best_loss != float('inf') else 0.0,
                'InferenceSpeed': 0.0,
                'CPUUsage%': 0.0,
                'RAMPeak(MB)': 0.0
            }

        # Calculate stability
        num_success = sum(
            1 for e in experiment_history if e.get('success', False))
        stability = (num_success / len(experiment_history)) * \
            100 if experiment_history else 100.0
        metrics['Stability%'] = stability

        # Final progress callback
        if progress_callback:
            try:
                progress_callback({
                    'type': 'final_metrics',
                    'Accuracy%': best_acc * 100,
                    'config': best_config,
                    'hyperparameters': best_config,
                    **metrics
                })
            except Exception as e:
                print(f"[LLM Training] Warning: Progress callback failed: {e}")

        return {
            'success': True,
            'best_accuracy': best_acc,
            'best_config': best_config,
            'model_path': best_model_file,
            'metrics': metrics,
            'experiment_history': experiment_history
        }

    else:
        print("[LLM Training] No successful models generated")
        return {
            'success': False,
            'error': 'No successful models generated',
            'experiment_history': experiment_history
        }
