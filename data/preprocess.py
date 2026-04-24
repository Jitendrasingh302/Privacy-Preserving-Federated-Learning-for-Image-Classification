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
