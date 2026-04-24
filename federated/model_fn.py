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
