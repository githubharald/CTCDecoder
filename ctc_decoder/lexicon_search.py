import numpy as np

from ctc_decoder.best_path import best_path
from ctc_decoder.bk_tree import BKTree
from ctc_decoder.loss import probability


def lexicon_search(mat: np.ndarray, chars: str, bk_tree: BKTree, tolerance: int) -> str:
    """Lexicon search decoder.

    The algorithm computes a first approximation using best path decoding. Similar words are queried using the BK tree.
    These word candidates are then scored given the neural network output, and the best one is returned.
    See CRNN paper from Shi, Bai and Yao.

    Args:
        mat: Output of neural network of shape TxC.
        chars: The set of characters the neural network can recognize, excluding the CTC-blank.
        bk_tree: Instance of BKTree which is used to query similar words.
        tolerance: Words to be considered, which are within specified edit distance.

    Returns:
        The decoded text.
    """

    # use best path decoding to get an approximation
    approx = best_path(mat, chars)

    # get similar words from dictionary within given tolerance
    words = bk_tree.query(approx, tolerance)

    # if there are no similar words, return empty string
    if not words:
        return ''

    # else compute probabilities of all similar words and return best scoring one
    word_probs = [(w, probability(mat, w, chars)) for w in words]
    word_probs.sort(key=lambda x: x[1], reverse=True)
    return word_probs[0][0]
