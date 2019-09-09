from __future__ import division
from __future__ import print_function
from itertools import groupby
import numpy as np


def ctcBestPath(mat, classes):
    "implements best path decoding as shown by Graves (Dissertation, p63)"

    # get char indices along best path
    best_path = np.argmax(mat, axis=1)

    # collapse best path (using itertools.groupby), map to chars, join char list to string
    blank_idx = len(classes)
    best_chars_collapsed = [classes[k] for k, _ in groupby(best_path) if k != blank_idx]
    res = ''.join(best_chars_collapsed)
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
