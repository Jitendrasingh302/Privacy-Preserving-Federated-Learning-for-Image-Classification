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
