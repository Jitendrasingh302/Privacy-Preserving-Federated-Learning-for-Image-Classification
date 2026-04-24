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
