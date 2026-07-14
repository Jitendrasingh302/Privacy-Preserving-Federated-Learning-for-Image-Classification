#this main file for traing of cnn model on mnist dataset from tensorflow

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
