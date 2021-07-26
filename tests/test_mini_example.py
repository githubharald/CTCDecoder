import numpy as np
import pytest

from ctc_decoder import beam_search
from ctc_decoder import best_path
from ctc_decoder import prefix_search
from ctc_decoder import probability, loss
from ctc_decoder import token_passing


@pytest.fixture
def mat():
    return np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])


@pytest.fixture
def chars():
    return 'ab'


def test_beam_search(mat, chars):
    expected = 'a'
    actual = beam_search(mat, chars)
    assert actual == expected


def test_best_path(mat, chars):
    expected = ''
    actual = best_path(mat, chars)
    assert actual == expected


def test_token_passing(mat, chars):
    expected = 'a'
    actual = token_passing(mat, chars, ['a', 'b', 'ab', 'ba'])
    assert actual == expected


def test_prefix_search(mat, chars):
    expected = 'a'
    actual = prefix_search(mat, chars)
    assert actual == expected


def test_probability(mat, chars):
    expected = 0.64
    actual = probability(mat, 'a', chars)
    assert actual == expected


def test_loss(mat, chars):
    expected = -np.log(0.64)
    actual = loss(mat, 'a', chars)
    assert actual == expected
