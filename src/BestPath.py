from __future__ import division
from __future__ import print_function
import numpy as np


def ctcBestPath(mat, classes):
    "implements best path decoding as shown by Graves (Dissertation, p63)"

    # get list of char indices along best path
    best_path = np.argmax(mat, axis=1)

    # collapse best path and map char indices to string
    blank_idx = len(classes)
    last_char_idx = blank_idx
    res = ''
    for char_idx in best_path:
        if char_idx != last_char_idx and char_idx != blank_idx:
            res += classes[char_idx]
        last_char_idx = char_idx

    return res


def testBestPath():
    "test decoder"
    classes = 'ab'
    mat = np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])
    print('Test best path decoding')
    expected = ''
    actual = ctcBestPath(mat, classes)
    print('Expected: "' + expected + '"')
    print('Actual: "' + actual + '"')
    print('OK' if expected == actual else 'ERROR')


if __name__ == '__main__':
    testBestPath()
