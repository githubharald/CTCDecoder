from itertools import groupby

import numpy as np


def best_path(mat: np.ndarray, chars: str) -> str:
    """Best path (greedy) decoder.

    Take best-scoring character per time-step, then remove repeated characters and CTC blank characters.
    See dissertation of Graves, p63.

    Args:
        mat: Output of neural network of shape TxC.
        chars: The set of characters the neural network can recognize, excluding the CTC-blank.

    Returns:
        The decoded text.
    """

    # get char indices along best path
    best_path_indices = np.argmax(mat, axis=1)

    # collapse best path (using itertools.groupby), map to chars, join char list to string
    blank_idx = len(chars)
    best_chars_collapsed = [chars[k] for k, _ in groupby(best_path_indices) if k != blank_idx]
    res = ''.join(best_chars_collapsed)
    return res
