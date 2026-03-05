# %%
import argparse
import importlib.util
import json
import os
import random
import re
import time
from datetime import datetime

import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from groq import Groq
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder

# === Configure Groq client ===
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

_groq_key = os.getenv("GROQ_API_KEY")
if not _groq_key:
    raise ValueError(
        "❌ GROQ_API_KEY is not set!\n"
        "Please add it to your .env file or set it in your environment.\n"
        "Get your API key from: https://console.groq.com/keys"
    )

client = Groq(api_key=_groq_key)

# ensure a valid directory exists
os.makedirs("./data", exist_ok=True)
print("Working dir:", os.getcwd())

# ========= PARAMETERS ===========
# Resize images for training (originals in histopathology are 768x768). Smaller sizes speed up experiments.
RESIZE_TO = (224, 224)  # (H, W). Use 224 or 128 depending on GPU capacity.

image_shape = None   # will be set after dataset is created
num_classes = None
num_images = None
dataset_name = None
target_accuracy = 1.0
max_iterations = 10
# Some filesystems prefer 0 workers; override via --num-workers if desired
DEFAULT_NUM_WORKERS = 0
DATA_NUM_WORKERS = DEFAULT_NUM_WORKERS
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========= DATA ================

# --- Safe ImageFolder creation with is_valid_file to ignore macOS resource files like "._*" and ".DS_Store"
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def is_valid_image_file(path):
    name = os.path.basename(path).lower()
    # exclude macOS metadata/resource fork files that start with "._" or .DS_Store
    if name.startswith('._') or name == '.ds_store':
        return False
    return name.endswith(VALID_EXTS)


def find_class_root(root):
    """Find the directory level that contains class subfolders with image files."""
    entries = [os.path.join(root, e) for e in os.listdir(root)]
    for e in entries:
        if os.path.isdir(e):
            files = os.listdir(e)
            if any(fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')) for fname in files):
                return root
    for e in entries:
        if os.path.isdir(e):
            subentries = [os.path.join(e, s) for s in os.listdir(e)]
            for s in subentries:
                if os.path.isdir(s):
                    files = os.listdir(s)
                    if any(fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')) for fname in files):
                        return e
    return root


def build_color_transforms(resize_to):
    train_transform = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def build_mnist_transforms():
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    return transform


def get_dataset_loaders(name, data_root=None, resize_to=RESIZE_TO, num_workers=DEFAULT_NUM_WORKERS, default_batch=64):
    """Return (train_dataset, val_or_test_dataset, testloader, image_shape, num_classes, num_images, dataset_key).

    Note: We return datasets (not train_loader), because training loop recreates train loader per batch size.
    """
    name_lower = name.lower()
    os.makedirs('./data', exist_ok=True)

    if name_lower == 'mnist':
        transform = build_mnist_transforms()
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        val_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
        sample_img, _ = train_dataset[0]
        image_shape = tuple(sample_img.shape)
        num_classes = 10
        dataset_key = 'MNIST'
    elif name_lower in ('lung-colon-cancer', 'lung_colon_cancer', 'lc25000'):
        train_t, val_t = build_color_transforms(resize_to)
        train_path = './data/lung-colon-cancer/train'
        test_path = './data/lung-colon-cancer/test'
        if not (os.path.isdir(train_path) and os.path.isdir(test_path)):
            raise FileNotFoundError(
                f"Expected train/test folders at {train_path} and {test_path}.")
        train_dataset = ImageFolder(
            root=train_path, transform=train_t, is_valid_file=is_valid_image_file)
        val_dataset = ImageFolder(
            root=test_path, transform=val_t, is_valid_file=is_valid_image_file)
        sample_img, _ = train_dataset[0]
        image_shape = tuple(sample_img.shape)
        num_classes = len(train_dataset.classes)
        dataset_key = 'lung-colon-cancer'
    elif name_lower in ('skin-cancer', 'skin_cancer', 'ham10000'):
        train_t, val_t = build_color_transforms(resize_to)
        train_path = './data/skin-cancer/organized/train'
        test_path = './data/skin-cancer/organized/test'
        if not (os.path.isdir(train_path) and os.path.isdir(test_path)):
            raise FileNotFoundError(
                f"Expected organized train/test at {train_path} and {test_path}. Run preprocessing if missing.")
        train_dataset = ImageFolder(
            root=train_path, transform=train_t, is_valid_file=is_valid_image_file)
        val_dataset = ImageFolder(
            root=test_path, transform=val_t, is_valid_file=is_valid_image_file)
        sample_img, _ = train_dataset[0]
        image_shape = tuple(sample_img.shape)
        num_classes = len(train_dataset.classes)
        dataset_key = 'skin-cancer'
    elif name_lower == 'yelp':
        train_t, val_t = build_color_transforms(resize_to)
        # Adjust this path if your Yelp data lives elsewhere
        train_path = '/N/slate/gsaraswa/yelp-dataset/organized/train'
        if not os.path.isdir(train_path):
            raise FileNotFoundError(f"Yelp dataset not found at {train_path}")
        train_dataset = ImageFolder(
            root=train_path, transform=train_t, is_valid_file=is_valid_image_file)
        val_dataset = ImageFolder(
            root=train_path, transform=val_t, is_valid_file=is_valid_image_file)
        sample_img, _ = train_dataset[0]
        image_shape = tuple(sample_img.shape)
        num_classes = len(train_dataset.classes)
        dataset_key = 'yelp'
    else:
        # Custom: use provided path or interpret 'name' as a path
        custom_root = data_root if data_root is not None else name
        if not os.path.exists(custom_root):
            raise FileNotFoundError(
                f"Custom dataset root not found at {custom_root}")
        train_t, val_t = build_color_transforms(resize_to)
        class_root = find_class_root(custom_root)
        print("Using class root:", class_root)

        full_dataset = ImageFolder(
            root=class_root, transform=val_t, is_valid_file=is_valid_image_file)
        if len(full_dataset) == 0:
            raise RuntimeError(
                "No valid images found by ImageFolder. Check dataset root and file extensions.")

        # Random split 80/20 if no explicit train/test split
        num_images_local = len(full_dataset)
        val_ratio = 0.20
        val_size = int(num_images_local * val_ratio)
        train_size = num_images_local - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        # Apply different transforms (random augmentation for train)
        train_dataset.dataset.transform = train_t
        val_dataset.dataset.transform = val_t

        sample_img, _ = full_dataset[0]
        image_shape = tuple(sample_img.shape) if isinstance(
            sample_img, torch.Tensor) else (3, resize_to[0], resize_to[1])
        num_classes = len(full_dataset.classes)
        dataset_key = os.path.basename(
            os.path.normpath(custom_root)) or 'custom'

    # testloader for eval
    testloader = DataLoader(val_dataset, batch_size=1000, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())
    num_images = len(train_dataset)
    print(f"Using dataset: {name} -> key '{dataset_key}'")
    print(
        f"Image shape: {image_shape}, Classes: {num_classes}, Training images: {num_images}")
    return train_dataset, val_dataset, testloader, image_shape, num_classes, num_images, dataset_key

# ========= HYPERPARAMETER SUGGESTIONS =========


def get_hyperparameter_suggestions(iteration, best_acc, last_config=None):
    suggestions = {
        'learning_rates': [1e-3, 5e-4, 2e-4],
        'batch_sizes': [16, 32, 64],
        'optimizers': ['Adam', 'SGD', 'RMSprop'],
        'dropout_rates': [0.1, 0.2, 0.3],
        'epochs': [3, 4, 5]
    }
    if iteration == 0 or best_acc < 0.5:
        return {'lr': 1e-3, 'batch_size': 32, 'optimizer': 'Adam', 'dropout_rate': 0.2, 'epochs': 3}
    elif best_acc < 0.8:
        return {
            'lr': random.choice([1e-3, 5e-4, 2e-4]),
            'batch_size': random.choice([32, 64]),
            'optimizer': random.choice(['Adam', 'SGD']),
            'dropout_rate': random.choice([0.1, 0.2]),
            'epochs': random.choice([4, 5])
        }
    else:
        return {'lr': random.choice([5e-4, 2e-4]), 'batch_size': random.choice([16, 32]), 'optimizer': 'Adam', 'dropout_rate': random.choice([0.1, 0.15]), 'epochs': 5}

# ========= LLM PROMPTING =========


def extract_python_code(text):
    python_pattern = r'```python\s*(.*?)\s*```'
    match = re.search(python_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    generic_pattern = r'```\s*(.*?)\s*```'
    match = re.search(generic_pattern, text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if 'class GeneratedCNN' in code:
            return code
    lines = text.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        if 'class GeneratedCNN' in line:
            in_code = True
        if in_code:
            code_lines.append(line)
    if code_lines:
        return '\n'.join(code_lines)
    if 'class GeneratedCNN' in text:
        return text.strip()
    raise ValueError("Could not extract Python code from LLM response")


# NOTE: template now uses placeholders for H2,W2,H4,W4 computed below to avoid str.format KeyError
base_system_prompt_template = """
You are a CNN architecture generator for image classification.

Given the input image shape, number of output classes, training images, and suggested hyperparameters,
you should generate a CNN model in PyTorch optimized for the given constraints.

The model must be wrapped in a class named `GeneratedCNN(nn.Module)` with a proper forward method.

Use Conv2D, ReLU, MaxPool2d, Dropout, and Linear layers.
End with a linear output of shape equal to number of classes.

CRITICAL: Calculate tensor shapes correctly!
- Input channels: {input_channels}
- For example, for an input of shape ({H}x{W}) and two Conv+Pool layers:
  - After Conv2d({input_channels},32,3,padding=1) + MaxPool2d(2): 32x{H2}x{W2}
  - After Conv2d(32,64,3,padding=1) + MaxPool2d(2): 64x{H4}x{W4}
  - Flatten to channels * {H4} * {W4} as the input to the first Linear layer.

Architecture Guidelines:
- Use the suggested dropout rate in Dropout layers
- Consider batch normalization for deeper networks
- Add more layers if accuracy is low, optimize existing layers if accuracy is high
- Use appropriate filter progression (e.g., 32->64->128)

IMPORTANT RULES:
1. Output ONLY the model definition: imports + `class GeneratedCNN(nn.Module)`.
2. DO NOT include any training code (no optimizer, no loss function, no model instantiation).
3. The file must not contain variables like `model`, `criterion`, `optimizer`, or training loops.
4. Allowed content:
   - `import torch`
   - `import torch.nn as nn`
   - `import torch.nn.functional as F` (if needed)
   - The `GeneratedCNN` class with `__init__` and `forward` methods
5. The model must end with the `forward` return statement. Nothing after the class.
"""


def get_user_prompt(image_shape, num_classes, num_images, hyperparams, previous_accuracy=None, previous_error=None, previous_config=None):
    channels, H, W = image_shape
    H2 = H // 2
    W2 = W // 2
    H4 = H // 4
    W4 = W // 4
    base_prompt = f"""
Input channels: {channels}
Image size: {H}x{W}
Number of classes: {num_classes}
Training images: {num_images}

HYPERPARAMETERS TO OPTIMIZE FOR:
- Learning Rate: {hyperparams['lr']}
- Batch Size: {hyperparams['batch_size']}
- Optimizer: {hyperparams['optimizer']}
- Dropout Rate: {hyperparams['dropout_rate']}
- Training Epochs: {hyperparams['epochs']}
"""
    if previous_accuracy is None:
        return base_prompt + f"""
Generate a CNN model optimized for these hyperparameters.
Make sure the first Conv2d uses input channels = {channels}.
Example structure:
1. Conv2d({channels}, 32, 3, padding=1) -> ReLU -> MaxPool2d(2) -> Output: 32x{H2}x{W2}
2. Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool2d(2) -> Output: 64x{H4}x{W4}
3. Flatten to 64*{H4}*{W4} features
4. Linear(64*{H4}*{W4}, hidden_size) -> ReLU -> Dropout(rate=SUGGESTED_DROPOUT_RATE)
5. Linear(hidden_size, {num_classes})

Make sure to use the suggested dropout rate in your model!
"""
    else:
        feedback = f"\nPrevious model accuracy: {previous_accuracy:.4f}"
        if previous_config:
            feedback += f"\nPrevious hyperparameters: LR={previous_config['lr']}, Batch={previous_config['batch_size']}, Optimizer={previous_config['optimizer']}"
        if previous_error:
            feedback += f"\nPrevious error: {previous_error}"
            if "mat1 and mat2 shapes cannot be multiplied" in previous_error or "shape" in previous_error:
                feedback += "\n\nShape mismatch error detected! Calculate tensor sizes carefully:"
                feedback += f"\n- Input size: {H}x{W}"
                feedback += "\n- After Conv(kernel=3,pad=1) + MaxPool(2): size = input_size/2"
                feedback += "\n- After 2 such layers: input -> input/2 -> input/4"
                feedback += f"\n- So final conv output: channels * {H4} * {W4}"
                feedback += "\n- Use this exact size for first Linear layer input!"
            feedback += "\nFix the error and generate a corrected model with the new hyperparameters."
        else:
            if previous_accuracy < 0.7:
                feedback += "\nLow accuracy detected. Try adding more convolutional layers or increasing filter sizes."
            elif previous_accuracy < 0.9:
                feedback += "\nModerate accuracy. Fine-tune the architecture with the new hyperparameters."
            else:
                feedback += "\nHigh accuracy! Make minor optimizations while maintaining performance."
        return base_prompt + feedback


def call_llm(system_prompt, user_prompt):
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6,
            max_completion_tokens=1024,
            top_p=1,
            reasoning_effort="medium",
            stream=False
        )
        print("LLM Response received")
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM call failed: {e}")
        raise

# ========= MODEL EXECUTION & AGENT LOOP =========


def load_model_from_file(filename):
    try:
        spec = importlib.util.spec_from_file_location(
            "generated_model", filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.GeneratedCNN
    except Exception as e:
        print(f"Failed to load model: {e}")
        try:
            with open(filename, "r") as f:
                print("Generated model file contents:")
                print("-" * 40)
                print(f.read())
                print("-" * 40)
        except:
            print(f"Could not read {filename} file")
        raise


def train_and_evaluate(model_cls, hyperparams):
    try:
        # instantiate & dummy forward using detected image_shape
        model = model_cls()
        c, h, w = image_shape
        dummy_input = torch.randn(2, c, h, w).to(device)
        model = model.to(device)

        # quick shape validation
        try:
            with torch.no_grad():
                dummy_output = model(dummy_input)
                print(
                    f"Model forward pass successful. Output shape: {dummy_output.shape}")
                if dummy_output.shape[1] != num_classes:
                    raise ValueError(
                        f"Output shape {dummy_output.shape} doesn't match num_classes {num_classes}")
        except Exception as e:
            raise ValueError(f"Model shape error: {e}")

        # configure optimizer
        if hyperparams['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])
        elif hyperparams['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(),
                                  lr=hyperparams['lr'], momentum=0.9)
        elif hyperparams['optimizer'] == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=hyperparams['lr'])
        else:
            optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])

        loss_fn = nn.CrossEntropyLoss()

        print(
            f"Training on {device} with {hyperparams['optimizer']} (lr={hyperparams['lr']}, epochs={hyperparams['epochs']})...")

        # recreate train_loader with chosen batch size
        train_loader = DataLoader(
            train_dataset, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=DATA_NUM_WORKERS, pin_memory=torch.cuda.is_available())

        # training loop (limited iterations per epoch to speed up)
        for epoch in range(hyperparams['epochs']):
            model.train()
            running_loss = 0.0
            batch_count = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                batch_count += 1
                if batch_count > 300:
                    break
            print(
                f"Epoch {epoch + 1}/{hyperparams['epochs']}, Loss: {running_loss / max(1, batch_count):.4f}")

        # evaluation on validation/test set
        correct, total = 0, 0
        model.eval()
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total

    except Exception as e:
        print(f"Training/evaluation failed: {e}")
        raise


def evaluate_with_metrics(model, testloader, device):
    process = psutil.Process()
    cpu_usages = []
    ram_usages = []
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    start_time = time.time()
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            cpu_before = process.cpu_percent(interval=None)
            ram_before = process.memory_info().rss / (1024**3)  # GB
            output = model(x)
            loss = loss_fn(output, y)
            preds = output.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * y.size(0)
            cpu_after = process.cpu_percent(interval=None)
            ram_after = process.memory_info().rss / (1024**3)
            cpu_usages.append((cpu_before + cpu_after) / 2)
            ram_usages.append(max(ram_before, ram_after))
    end_time = time.time()
    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / total if total > 0 else 0
    inference_time = end_time - start_time
    inference_speed = total / inference_time if inference_time > 0 else 0  # images/sec
    cpu_avg = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
    ram_peak = max(ram_usages) if ram_usages else 0
    stability = 100.0  # placeholder for stability percentage
    return {
        "Accuracy%": accuracy * 100,
        "Loss": avg_loss,
        "Inference_Speed": inference_speed,
        "CPU_Usage_Avg_%of_all_cores": cpu_avg,
        "RAM_Peak_GB": ram_peak,
        "Stability%": stability
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='CNN Architecture Generation and Training (multi-dataset)')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='Dataset name or path (MNIST, lung-colon-cancer, skin-cancer, yelp, or a custom folder path)')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Explicit root for custom dataset (overrides --dataset when provided)')
    parser.add_argument('--resize', type=int, nargs=2, metavar=('H', 'W'),
                        default=None, help='Resize images to H W (for color datasets)')
    parser.add_argument('--num-workers', type=int,
                        default=DEFAULT_NUM_WORKERS, help='DataLoader num_workers')

    # Platform integration arguments
    parser.add_argument('--model-id', type=str, default=None,
                        help='Model ID for output file naming (from platform)')
    parser.add_argument('--job-id', type=str, default=None,
                        help='Job ID for progress tracking (from platform)')

    # Hyperparameter overrides (optional - LLM suggestions used if not provided)
    parser.add_argument('--lr', '--learning-rate', type=float, default=None, dest='learning_rate',
                        help='Learning rate override')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size override')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs override')
    parser.add_argument('--optimizer', type=str, default=None, choices=['Adam', 'SGD', 'RMSprop'],
                        help='Optimizer type override')
    parser.add_argument('--dropout', '--dropout-rate', type=float, default=None, dest='dropout_rate',
                        help='Dropout rate override')

    # Training configuration
    parser.add_argument('--max-iterations', type=int, default=None,
                        help='Maximum LLM refinement iterations')
    parser.add_argument('--target-accuracy', type=float, default=None,
                        help='Target accuracy to achieve (0.0-1.0)')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'],
                        help='Training device (cpu, cuda, or mps)')

    return parser.parse_args()


def main():
    global num_classes, num_images, image_shape, dataset_name, train_dataset, testloader, DATA_NUM_WORKERS, max_iterations, target_accuracy, device
    args = parse_args()
    dataset_name = args.dataset
    resize_to = tuple(args.resize) if args.resize else RESIZE_TO
    DATA_NUM_WORKERS = args.num_workers

    # Apply CLI overrides for training configuration
    if args.max_iterations is not None:
        max_iterations = args.max_iterations
    if args.target_accuracy is not None:
        target_accuracy = args.target_accuracy
    if args.device is not None:
        device = args.device

    # Platform integration IDs
    model_id = args.model_id
    job_id = args.job_id

    # Load dataset(s)
    train_dataset, val_dataset, testloader, image_shape, num_classes, num_images, dataset_key = get_dataset_loaders(
        args.dataset, data_root=args.data_root, resize_to=resize_to, num_workers=args.num_workers
    )

    # compute halves/quarters for formatting safely
    H2 = image_shape[1] // 2
    W2 = image_shape[2] // 2
    H4 = image_shape[1] // 4
    W4 = image_shape[2] // 4

    base_system_prompt = base_system_prompt_template.format(
        input_channels=image_shape[0],
        H=image_shape[1],
        W=image_shape[2],
        H2=H2,
        W2=W2,
        H4=H4,
        W4=W4
    )

    best_acc = 0.0
    best_model_code = ""
    best_config = None
    last_error = None
    experiment_history = []

    models_dir = os.path.join('models')
    os.makedirs(models_dir, exist_ok=True)

    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}/{max_iterations}")
        current_config = get_hyperparameter_suggestions(
            iteration, best_acc, best_config)
        print(
            f"Hyperparameters: LR={current_config['lr']}, Batch={current_config['batch_size']}, Optimizer={current_config['optimizer']}, Dropout={current_config['dropout_rate']}, Epochs={current_config['epochs']}")

        try:
            user_prompt = get_user_prompt(image_shape, num_classes, num_images, current_config,
                                          None if iteration == 0 else best_acc, last_error, best_config)
            raw_response = call_llm(base_system_prompt, user_prompt)
            model_code = extract_python_code(raw_response)

            # ensure imports
            if 'import torch' not in model_code:
                model_code = 'import torch\n' + model_code
            if 'import torch.nn as nn' not in model_code:
                model_code = 'import torch.nn as nn\n' + model_code
            if 'import torch.nn.functional as F' not in model_code and 'F.' in model_code:
                model_code = 'import torch.nn.functional as F\n' + model_code

            generated_model_filename = os.path.join(
                models_dir, f"generated_model_{dataset_key}.py")
            with open(generated_model_filename, "w") as f:
                f.write(model_code)
            print(f"Saved {generated_model_filename}")

            try:
                model_cls = load_model_from_file(generated_model_filename)
                acc = train_and_evaluate(model_cls, current_config)
                print(f"Validation accuracy: {acc:.4f}")

                experiment_history.append(
                    {'iteration': iteration + 1, 'accuracy': acc, 'config': current_config.copy(), 'success': True})
                if acc > best_acc:
                    best_acc = acc
                    best_model_code = model_code
                    best_config = current_config.copy()
                    print(
                        f"New best accuracy: {best_acc:.4f} Best config: {best_config}")
                last_error = None

            except Exception as e:
                last_error = str(e)
                print(f"Model execution failed: {e}")
                experiment_history.append({'iteration': iteration + 1, 'accuracy': 0.0,
                                          'config': current_config.copy(), 'success': False, 'error': str(e)})
                continue

        except Exception as e:
            print(f"Iteration failed: {e}")
            last_error = str(e)
            continue

    # Save results
    if best_model_code:
        best_model_filename = os.path.join(
            models_dir, f"best_model_{dataset_key}.py")
        results_dir = os.path.join('results', dataset_key)
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M")
        history_filename = os.path.join(
            results_dir, f"experiment_history_{timestamp}.json")
        with open(best_model_filename, "w") as f:
            f.write(best_model_code)
        with open(history_filename, "w") as f:
            json.dump(experiment_history, f, indent=2)
        print(f"\nBest achieved accuracy: {best_acc:.4f}")
        print(f"Best configuration: {best_config}")
        print(f"Best model saved to {best_model_filename}")
        print(f"Experiment history saved to {history_filename}")

        # Load the best model and compute the metrics, then print them as JSON.
        model_cls = load_model_from_file(best_model_filename)
        model = model_cls().to(device)
        metrics = evaluate_with_metrics(model, testloader, device)
        num_success = sum(
            1 for e in experiment_history if e.get('success', False))
        stability = (num_success / len(experiment_history)) * \
            100 if experiment_history else 100.0
        metrics["Stability%"] = stability
        metrics["Accuracy%"] = best_acc * 100  # Use accuracy from the best run
        print(json.dumps(metrics))
    else:
        print("\nNo successful models generated")


if __name__ == "__main__":
    main()
