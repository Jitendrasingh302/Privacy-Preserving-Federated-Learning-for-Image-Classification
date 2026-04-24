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
