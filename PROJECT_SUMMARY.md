# Complete Project Summary: Privacy-Preserving Federated Learning for Image Classification

## Project Overview
This is a **Python-based machine learning project** implementing **Federated Learning (FL)** with **Differential Privacy (DP)** for privacy-preserving image classification. It supports **MNIST** (handwritten digits) and **CIFAR-10** (colored images). 

**Key Features:**
- Simulates distributed training across multiple clients (devices) without sharing raw data.
- Central server aggregates model updates using standard averaging or DP-noisy averaging.
- Uses TensorFlow/Keras for models and custom simulated FL loop (not full TFF runtime for simplicity).
- Generates comparison plots of standard FL vs DP-FL (accuracy/loss over rounds).
- Command-line args for #clients, rounds, noise level.

**Purpose:** Demonstrate trade-off between model accuracy and privacy guarantees. DP adds Gaussian noise to prevent data leakage from aggregates.

**Outputs:** Console metrics per round, final test acc/loss, PNG plots (`training_comparison.png`, `cifar_training_comparison.png`).

## Tech Stack & Dependencies
- **Python 3.x**
- **TensorFlow/Keras >=2.10.0** (core ML)
- **Matplotlib >=3.5.0** (plots)
- **NumPy >=1.20.0** (data handling)

Install:
```
pip install -r requirements.txt
```

**Environment Check:** `utils/helpers.py` prints Python version.

## Full File Structure
```
d:/coding/RAI/project/
├── asdf.txt (unused?)
├── cifar_training_comparison.png (CIFAR plot output)
├── main_cifar.py (CIFAR-10 entrypoint)
├── main.py (MNIST entrypoint)
├── README.md (this summary source)
├── requirements.txt
├── training_comparison.png (MNIST plot output)
├── data/
│   ├── client_split.py
│   ├── load_data.py
│   └── preprocess.py (likely unused)
├── evaluation/
│   ├── evaluate.py
│   └── plots.py (plot generation)
├── federated/
│   ├── aggregation.py (fed_avg, dp_fed_avg)
│   ├── federated_training.py (TFF process builder)
│   └── model_fn.py (TFF model wrapper)
├── models/
│   ├── cnn_model_cifar.py (CIFAR CNN)
│   └── cnn_model.py (MNIST CNN)
├── privacy/
│   └── dp_mechanism.py (DP aggregator factory)
├── training/
│   └── train.py (custom FL training loop)
├── utils/
│   └── helpers.py (env check)
└── venv_old/ (old virtualenv)
```

## How It Works: High-Level Flow
1. **Data Prep (`data/`):**
   - Load MNIST/CIFAR via `load_data.py`.
   - Split into N client datasets (`client_split.py`): IID shuffle, ~equal samples/client, tf.data batches (32).

2. **Model Definition (`models/`):**
   - MNIST: Simple CNN (Conv2D32-3x3, MaxPool, Flatten, Dense64, Dense10-softmax).
   - CIFAR: Similar but deeper (in `cnn_model_cifar.py`).

3. **Federated Training (`training/train.py`, `federated/`):**
   - Custom loop (not iterative TFF for speed):
     - Init global model.
     - For each round:
       - Sample K clients randomly.
       - Each: Local train 1 epoch (SGD lr=0.02).
       - Aggregate: `fed_avg` (mean weights) or `dp_fed_avg` (noisy mean).
     - Track avg client loss/acc.

4. **Privacy (`privacy/dp_mechanism.py`, `federated/aggregation.py`):**
   - DP: `DifferentiallyPrivateFactory.gaussian_1d` (noise_multiplier, clip_norm=1.0).
   - Applied during server aggregation.

5. **Evaluation (`evaluation/`):**
   - Load final weights into new model, test on central held-out data (batch 100).
   - `plots.py`: Line plots of acc/loss vs rounds (standard vs DP).

6. **Entrypoints:**
   - `main.py` (MNIST): `python main.py --clients 10 --clients_per_round 5 --rounds 15 --noise_multiplier 0.05`
   - `main_cifar.py` (CIFAR): Same args.

**Example Console Output:**
```
Round  1 - Loss: 1.2345, Accuracy: 0.4567
...
Final Central Test Loss: 0.1234, Test Accuracy: 0.9456
```

## Key Code Excerpts

### Data Loading (data/load_data.py)
```python
def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def load_cifar10():  # Similar for CIFAR
    ...
```

### Client Split (data/client_split.py)
```python
def split_data_into_clients(x_data, y_data, num_clients):
    # Shuffle, split equally, normalize /255, tf.data.batch(32)
    ...
```

### Model (models/cnn_model.py)
```python
def create_keras_model():
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

### Training Loop (training/train.py)
```python
for round_num in range(1, num_rounds + 1):
    sampled_clients = random.sample(federated_data, clients_per_round)
    # Local train each, collect weights
    if use_dp:
        global_weights = dp_fed_avg(global_weights, client_weights_list, noise_multiplier)
    else:
        global_weights = fed_avg(client_weights_list)
```

### Main Execution (main.py)
```python
fl_weights, fl_metrics = train_federated_model(..., use_dp=False)
dp_weights, dp_metrics = train_federated_model(..., use_dp=True)
plot_metrics(fl_metrics, dp_metrics)
```

## Potential Improvements/Extensions
- Use full TFF runtime for real simulation.
- Non-IID data splits.
- Advanced DP (DP-SGD per client).
- Hyperparam tuning.
- More datasets/models.

## Generating an App from This
Use this summary in Claude AI/GPT to create:
- Web dashboard (Streamlit/Flask) visualizing FL training.
- Mobile app simulating clients.
- Interactive demo with sliders for noise/clients/rounds.

**Project ready to run - no setup beyond pip install!**

