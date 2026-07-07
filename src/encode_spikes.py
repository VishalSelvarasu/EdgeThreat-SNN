import numpy as np
from config import TIME_STEPS


def rate_encode(features, time_steps=TIME_STEPS):
    features = np.asarray(features, dtype=np.float32)
    spikes = np.random.rand(time_steps, features.shape[0], features.shape[1]) < features[None, :, :]
    return spikes.astype(np.float32)
