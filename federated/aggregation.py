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
