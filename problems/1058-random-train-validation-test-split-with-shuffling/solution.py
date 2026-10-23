import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    n, d = data.shape
    indices = np.random.default_rng(seed).permutation(n)

    train_end = int(n * train_frac)
    val_end =  train_end +int(n * validation_frac)

    train_indices, val_indices, test_indices = indices[:train_end], indices[train_end:val_end], indices[val_end:]

    return [data[train_indices], data[val_indices], data[test_indices]]