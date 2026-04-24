import tensorflow as tf
import tensorflow_federated as tff

from models.cnn_model import create_keras_model
from federated.model_fn import get_input_spec
from federated.aggregation import get_aggregation_factory

def model_fn():
    """
    Wrapper for TFF to instantiate the Keras model dynamically.
    """
    keras_model = create_keras_model()
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=get_input_spec(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

def build_federated_process(use_dp=False, noise_multiplier=0.1, clients_per_round=10):
    """
    Builds the iterative process for Federated Averaging.
    We use Unweighted FedAvg, which helps stabilize DP noise additions.
    """
    aggregator = get_aggregation_factory(
        use_dp=use_dp, 
        noise_multiplier=noise_multiplier, 
        clients_per_round=clients_per_round
    )
    
    # Return iterative process using the newer TFF learning API
    return tff.learning.algorithms.build_unweighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        model_aggregator=aggregator
    )
