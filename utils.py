import numpy as np
import random
from contextlib import contextmanager

@contextmanager
def fixed_seed(seed):
    np_random_state = np.random.get_state()
    random_state = random.getstate()

    np.random.seed(seed)
    random.seed(seed)

    try:
        yield
    finally:
        np.random.set_state(np_random_state)
        random.setstate(random_state)