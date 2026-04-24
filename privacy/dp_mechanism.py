import tensorflow_federated as tff

def build_dp_factory(noise_multiplier, clients_per_round, clip_norm=1.0):
    """
    Creates a DifferentiallyPrivateFactory.
    We use standard gaussian noise with clipping.
    """
    return tff.aggregators.DifferentiallyPrivateFactory.gaussian_1d(
        noise_multiplier=noise_multiplier,
        clients_per_round=clients_per_round,
        clip=clip_norm
    )
