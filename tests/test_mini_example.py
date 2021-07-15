import numpy as np
import pytest

from ctc_decoder import LanguageModel
from ctc_decoder import beam_search
from ctc_decoder import best_path
from ctc_decoder import prefix_search
from ctc_decoder import probability, loss
from ctc_decoder import token_passing


@pytest.fixture
def mat():
    return np.array([[0.4, 0, 0.6], [0.4, 0, 0.6]])


@pytest.fixture
def labels():
    return 'ab'


def test_beam_search(mat, labels):
    expected = 'a'
    actual = beam_search(mat, labels)
    assert actual == expected


def test_best_path(mat, labels):
    expected = ''
    actual = best_path(mat, labels)
    assert actual == expected


def test_token_passing(mat, labels):
    expected = 'a'
    lm = LanguageModel('a b ab ba', labels)
    actual = token_passing(mat, labels, lm)
    assert actual == expected


def test_prefix_search(mat, labels):
    expected = 'a'
    actual = prefix_search(mat, labels)
    assert actual == expected


def test_probability(mat, labels):
    expected = 0.64
    actual = probability(mat, 'a', labels)
    assert actual == expected


def test_loss(mat, labels):
    expected = -np.log(0.64)
    actual = loss(mat, 'a', labels)
    assert actual == expected
