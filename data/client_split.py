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
