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
