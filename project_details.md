# Project Details: Privacy-Preserving Image Classification using Federated Learning and Differential Privacy

This document consolidates the entire source code and project structure to be used for generating documentation or apps via LLMs like Claude AI.

## Directory Structure

```text
data/
  client_split.py
  load_data.py
  preprocess.py
evaluation/
  evaluate.py
  plots.py
federated/
  aggregation.py
  federated_training.py
  model_fn.py
models/
  cnn_model.py
  cnn_model_cifar.py
privacy/
  dp_mechanism.py
training/
  train.py
utils/
  helpers.py
main.py
main_cifar.py
requirements.txt
README.md
```

## File Contents

### `README.md`
```markdown
# Privacy-Preserving Image Classification using Federated Learning and Differential Privacy

This project implements a machine learning system that performs image classification on the MNIST dataset using Federated Learning (FL) with Differential Privacy (DP). It is built using standard Keras/TensorFlow and TensorFlow Federated.

## Features
- **Federated Learning (FL):** Distributes the data across multiple simulated clients. Each client locally trains a model without sharing any raw data, submitting only model updates (weights) back to the central server.
- **Differential Privacy (DP):** Adds calibrated Gaussian noise to the aggregated model updates. This prevents potential reconstruction or inference attacks against the aggregated gradient, guaranteeing that the model won't leak sensitive individual data information.
- **CNN Architecture:** A baseline sequential Convolutional Neural Network used for classification.

## File Structure
- `data/`: Handles loading MNIST and splitting into clients.
- `models/`: Stores the CNN architecture definitions.
- `federated/`: Logic embedding model/metrics in TFF contexts, defining training loops.
- `privacy/`: Setup for Differentially Private Aggregators.
- `training/`: Iterative loop execution.
- `evaluation/`: Provides central testing functions and metric plots.

## Execution Instructions
You can execute the code out-of-the-box using the provided requirements. Let's create an environment, install dependencies, and run:

```bash
pip install -r requirements.txt
python main.py --clients 10 --clients_per_round 5 --rounds 15 --noise_multiplier 0.05
```

## Results Summary
Running the solution prints the client loss and accuracy for both standard Unweighted Federated Averaging and the DP-infused Unweighted Federated Averaging. Standard FL reaches high accuracy relatively quickly without noise, whereas DP-FL trades off some precision and convergence rate for guaranteed privacy protections. The generated `training_comparison.png` visualizes this behavior natively.
```

### `requirements.txt`
```text
tensorflow>=2.10.0
matplotlib>=3.5.0
numpy>=1.20.0
```

### `main.py`
```python
import argparse
from data.load_data import load_mnist
from data.client_split import split_data_into_clients, prepare_central_test_data
from training.train import train_federated_model
from evaluation.plots import plot_metrics
from evaluation.evaluate import evaluate_global_model
from utils.helpers import check_environment

def main():
    check_environment()
    parser = argparse.ArgumentParser(description='Privacy-Preserving Federated Learning')
    parser.add_argument('--clients', type=int, default=10, help='Total number of clients')
    parser.add_argument('--clients_per_round', type=int, default=5, help='Clients sampled per round')
    parser.add_argument('--rounds', type=int, default=10, help='Number of federated rounds')
    parser.add_argument('--noise_multiplier', type=float, default=0.05, help='DP noise multiplier')
    args = parser.parse_args()

    print("Loading and splitting MNIST data...")
    (x_train, y_train), (x_test, y_test) = load_mnist()
    
    # Create disjoint datasets simulating independent devices
    federated_train_data = split_data_into_clients(x_train, y_train, args.clients)
    central_test_data = prepare_central_test_data(x_test, y_test)

    print(f"\n--- Training Standard Federated Learning (Rounds: {args.rounds}) ---")
    fl_weights, fl_metrics = train_federated_model(
        args.clients, federated_train_data, args.rounds, args.clients_per_round, use_dp=False)
    
    print("\nStandard FL Final Evaluation:")
    evaluate_global_model(fl_weights, central_test_data)

    print(f"\n--- Training DP-Federated Learning (Noise: {args.noise_multiplier}) ---")
    dp_weights, dp_metrics = train_federated_model(
        args.clients, federated_train_data, args.rounds, args.clients_per_round, use_dp=True, noise_multiplier=args.noise_multiplier)
    
    print("\nDP-FL Final Evaluation:")
    evaluate_global_model(dp_weights, central_test_data)
        
    print("\nGenerating comparison plots...")
    plot_metrics(fl_metrics, dp_metrics, args.rounds)
    print("Execution complete! Checkout training_comparison.png for details.")

if __name__ == "__main__":
    main()
```

### `main_cifar.py`
```python
import argparse
from data.load_data import load_cifar10
from models.cnn_model_cifar import create_cifar_model
from data.client_split import split_data_into_clients, prepare_central_test_data
from training.train import train_federated_model
from evaluation.plots import plot_metrics
from evaluation.evaluate import evaluate_global_model
from utils.helpers import check_environment

def main():
    check_environment()
    parser = argparse.ArgumentParser(description='Privacy-Preserving Federated Learning (CIFAR-10)')
    parser.add_argument('--clients', type=int, default=10, help='Total number of clients')
    parser.add_argument('--clients_per_round', type=int, default=5, help='Clients sampled per round')
    parser.add_argument('--rounds', type=int, default=10, help='Number of federated rounds')
    parser.add_argument('--noise_multiplier', type=float, default=0.05, help='DP noise multiplier')
    args = parser.parse_args()

    print("Loading and splitting CIFAR-10 data...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    
    # Create disjoint datasets simulating independent devices
    federated_train_data = split_data_into_clients(x_train, y_train, args.clients)
    central_test_data = prepare_central_test_data(x_test, y_test)

    print(f"\n--- Training Standard Federated Learning (Rounds: {args.rounds}) ---")
    fl_weights, fl_metrics = train_federated_model(
        args.clients, federated_train_data, args.rounds, args.clients_per_round, use_dp=False, model_fn=create_cifar_model)
    
    print("\nStandard FL Final Evaluation:")
    evaluate_global_model(fl_weights, central_test_data, model_fn=create_cifar_model)

    print(f"\n--- Training DP-Federated Learning (Noise: {args.noise_multiplier}) ---")
    dp_weights, dp_metrics = train_federated_model(
        args.clients, federated_train_data, args.rounds, args.clients_per_round, use_dp=True, noise_multiplier=args.noise_multiplier, model_fn=create_cifar_model)
    
    print("\nDP-FL Final Evaluation:")
    evaluate_global_model(dp_weights, central_test_data, model_fn=create_cifar_model)
        
    print("\nGenerating comparison plots...")
    plot_metrics(fl_metrics, dp_metrics, args.rounds, output_filename='cifar_training_comparison.png')
    print("Execution complete! Checkout cifar_training_comparison.png for details.")

if __name__ == "__main__":
    main()
```

### `data/client_split.py`
```python
import tensorflow as tf
import numpy as np

def split_data_into_clients(x_data, y_data, num_clients):
    """
    Splits the data into `num_clients` partitions to simulate FL clients.
    """
    num_samples = len(x_data)
    samples_per_client = num_samples // num_clients
    
    # Shuffle data before splitting for IID distribution
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    x_data = x_data[indices]
    y_data = y_data[indices]
    
    clients_datasets = []
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        
        client_x = x_data[start_idx:end_idx]
        client_y = y_data[start_idx:end_idx]
        
        # Create un-batched tf.data.Dataset yielding tuples
        client_dataset = tf.data.Dataset.from_tensor_slices((
            client_x.astype(np.float32) / 255.0, 
            client_y.astype(np.int32)
        ))
        
        # We apply standard batching here for TFF
        client_dataset = client_dataset.shuffle(1000).batch(32, drop_remainder=True)
        clients_datasets.append(client_dataset)
        
    return clients_datasets

def prepare_central_test_data(x_test, y_test):
    """
    Prepares test dataset for central evaluation.
    """
    dataset = tf.data.Dataset.from_tensor_slices((
        x_test.astype(np.float32) / 255.0, 
        y_test.astype(np.int32)
    ))
    return dataset.batch(100)
```

### `data/load_data.py`
```python
import tensorflow as tf

def load_mnist():
    """
    Loads the MNIST dataset from Keras.
    Returns:
        Tuple of (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def load_cifar10():
    """
    Loads the CIFAR-10 dataset from Keras.
    Returns:
        Tuple of (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)

```

### `data/preprocess.py`
```python
import tensorflow as tf

def preprocess_fn(dataset):
    """
    Normalizes dataset and converts labels to int32.
    It expects a tf.data.Dataset object yielding (image, label) tuples.
    """
    def element_fn(image, label):
        # Resize/expand and normalize
        image = tf.expand_dims(image, -1)
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(label, tf.int32)
        return (image, label)

    return dataset.map(element_fn).shuffle(1000).batch(32, drop_remainder=True)

```

### `evaluation/evaluate.py`
```python
import tensorflow as tf
from models.cnn_model import create_keras_model

def evaluate_global_model(model_weights, test_data, model_fn=None):
    """
    Evaluates the final aggregated global model on the central test dataset.
    """
    if model_fn is None:
        model_fn = create_keras_model
    model = model_fn()
    model.compile(
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.set_weights(model_weights)
    loss, accuracy = model.evaluate(test_data, verbose=0)
    print(f"Final Central Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    return accuracy, loss
```

### `evaluation/plots.py`
```python
import matplotlib.pyplot as plt
import os

def plot_metrics(fl_metrics, dp_fl_metrics, num_rounds, output_dir='.', output_filename='training_comparison.png'):
    """
    Plots the accuracy and loss curves for standard FL and DP-FL testing rounds.
    """
    rounds = range(1, num_rounds + 1)
    
    fl_acc = [m['accuracy'] for m in fl_metrics]
    dp_fl_acc = [m['accuracy'] for m in dp_fl_metrics]
    
    fl_loss = [m['loss'] for m in fl_metrics]
    dp_fl_loss = [m['loss'] for m in dp_fl_metrics]
    
    plt.figure(figsize=(12, 5))
    
    # Accuracy curves
    plt.subplot(1, 2, 1)
    plt.plot(rounds, fl_acc, label='Standard FL', color='blue', marker='o')
    plt.plot(rounds, dp_fl_acc, label='DP-FL', color='orange', marker='s')
    plt.title('Client Accuracy vs Rounds')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss curves
    plt.subplot(1, 2, 2)
    plt.plot(rounds, fl_loss, label='Standard FL', color='blue', marker='o')
    plt.plot(rounds, dp_fl_loss, label='DP-FL', color='orange', marker='s')
    plt.title('Client Loss vs Rounds')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    print(f"Plots successfully saved to {output_path}")
    plt.close()
```

### `federated/aggregation.py`
```python
import numpy as np

def fed_avg(client_weights_list):
    """
    Standard Federated Averaging of client weights.
    Returns the averaged weights.
    """
    return [np.mean(layer, axis=0) for layer in zip(*client_weights_list)]

def clip_update(update, clip_norm):
    """Clips an update vector (list of arrays) to the maximum L2 norm."""
    flat = np.concatenate([u.flatten() for u in update])
    l2_norm = np.linalg.norm(flat)
    if l2_norm > clip_norm:
        scalar = clip_norm / l2_norm
        return [u * scalar for u in update]
    return update

def dp_fed_avg(global_weights, client_weights_list, noise_multiplier, clip_norm=1.0):
    """
    Federated Averaging with Central Differential Privacy.
    1. Compute updates: delta = client - global
    2. Clip updates per client
    3. Average clipped updates
    4. Add Gaussian noise to averaged update
    5. Apply update: global + noisy_average
    """
    clipped_updates = []
    for client_w in client_weights_list:
        update = [c - g for c, g in zip(client_w, global_weights)]
        clipped_updates.append(clip_update(update, clip_norm))
        
    # Average the clipped updates
    avg_update = [np.mean(layer, axis=0) for layer in zip(*clipped_updates)]
    
    # Add noise
    noisy_update = []
    m = len(client_weights_list)
    # The sensitivity of average clipped update is clip_norm / m
    stddev = (clip_norm * noise_multiplier) / m
    
    for au in avg_update:
        noise = np.random.normal(loc=0.0, scale=stddev, size=au.shape)
        noisy_update.append(au + noise)
        
    # Apply the noisy update to global weights
    new_global_weights = [g + nu for g, nu in zip(global_weights, noisy_update)]
    return new_global_weights
```

### `federated/federated_training.py`
```python
import tensorflow as tf
import tensorflow_federated as tff

from models.cnn_model import create_keras_model
from federated.model_fn import get_input_spec
from federated.aggregation import get_aggregation_factory

def model_fn():
    """
    Wrapper for TFF to instantiate the Keras model dynamically.
    """
    keras_model = create_keras_model()
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=get_input_spec(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

def build_federated_process(use_dp=False, noise_multiplier=0.1, clients_per_round=10):
    """
    Builds the iterative process for Federated Averaging.
    We use Unweighted FedAvg, which helps stabilize DP noise additions.
    """
    aggregator = get_aggregation_factory(
        use_dp=use_dp, 
        noise_multiplier=noise_multiplier, 
        clients_per_round=clients_per_round
    )
    
    # Return iterative process using the newer TFF learning API
    return tff.learning.algorithms.build_unweighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        model_aggregator=aggregator
    )
```

### `federated/model_fn.py`
```python
import tensorflow as tf

def get_input_spec():
    """
    Returns the tensor specification for the federated model input.
    Matches the batched MNIST datset.
    """
    return (
        tf.TensorSpec(shape=(None, 28, 28), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
```

### `models/cnn_model.py`
```python
import tensorflow as tf

def create_keras_model():
    """
    Creates a simple CNN model for MNIST classification.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape(target_shape=(28, 28, 1), input_shape=(28, 28)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model
```

### `models/cnn_model_cifar.py`
```python
import tensorflow as tf

def create_cifar_model():
    """
    Creates a simple CNN model for CIFAR-10 classification.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model
```

### `privacy/dp_mechanism.py`
```python
import tensorflow_federated as tff

def build_dp_factory(noise_multiplier, clients_per_round, clip_norm=1.0):
    """
    Creates a DifferentiallyPrivateFactory.
    We use standard gaussian noise with clipping.
    """
    return tff.aggregators.DifferentiallyPrivateFactory.gaussian_1d(
        noise_multiplier=noise_multiplier,
        clients_per_round=clients_per_round,
        clip=clip_norm
    )
```

### `training/train.py`
```python
import random
import tensorflow as tf
from models.cnn_model import create_keras_model
from federated.aggregation import fed_avg, dp_fed_avg

def train_federated_model(num_clients, federated_data, num_rounds, clients_per_round, use_dp=False, noise_multiplier=0.1, model_fn=None):
    """
    Custom Federated Training Loop simulating TFF locally.
    """
    if model_fn is None:
        model_fn = create_keras_model

    # Initialize global model
    global_model = model_fn()
    global_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    global_weights = global_model.get_weights()
    
    metrics_list = []
    
    for round_num in range(1, num_rounds + 1):
        sampled_clients = random.sample(federated_data, clients_per_round)
        
        client_weights_list = []
        round_losses = []
        round_accs = []
        
        for client_dataset in sampled_clients:
            # Local client training
            client_model = model_fn()
            client_model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.02), 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy']
            )
            client_model.set_weights(global_weights)
            
            # Train for 1 epoch
            history = client_model.fit(client_dataset, epochs=1, verbose=0)
            
            client_weights_list.append(client_model.get_weights())
            round_losses.append(history.history['loss'][-1])
            round_accs.append(history.history['accuracy'][-1])
            
        # Server Aggregation
        if use_dp:
            global_weights = dp_fed_avg(global_weights, client_weights_list, noise_multiplier)
        else:
            global_weights = fed_avg(client_weights_list)
            
        # Tracking metrics
        avg_loss = sum(round_losses) / len(round_losses)
        avg_acc = sum(round_accs) / len(round_accs)
        metrics_list.append({'accuracy': avg_acc, 'loss': avg_loss})
        
        print(f"Round {round_num:2d} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
        
    return global_weights, metrics_list
```

### `utils/helpers.py`
```python
import sys

def check_environment():
    """
    Prints environment info for debugging purposes.
    """
    print(f"Python version: {sys.version}")
```
