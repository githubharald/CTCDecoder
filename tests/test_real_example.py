import re

import numpy as np
import pytest

from ctc_decoder import BKTree
from ctc_decoder import LanguageModel
from ctc_decoder import beam_search
from ctc_decoder import best_path
from ctc_decoder import lexicon_search
from ctc_decoder import loss
from ctc_decoder import prefix_search_heuristic_split
from ctc_decoder import probability
from ctc_decoder import token_passing


def softmax(mat):
    maxT, _ = mat.shape  # dim0=t, dim1=c
    res = np.zeros(mat.shape)
    for t in range(maxT):
        y = mat[t, :]
        e = np.exp(y)
        s = np.sum(e)
        res[t, :] = e / s
    return res


def load_rnn_output(fn):
    return np.genfromtxt(fn, delimiter=';')[:, : -1]


@pytest.fixture
def line_mat():
    return softmax(load_rnn_output('../data/line/rnnOutput.csv'))


@pytest.fixture
def word_mat():
    return softmax(load_rnn_output('../data/word/rnnOutput.csv'))


@pytest.fixture
def labels():
    return ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


@pytest.fixture
def corpus():
    with open('../data/line/corpus.txt') as f:
        txt = f.read()
    return txt


@pytest.fixture
def words():
    with open('../data/word/corpus.txt') as f:
        words = f.read().split()
    return words


def test_line_example_best_path(line_mat, labels):
    mat = line_mat
    assert best_path(mat, labels) == 'the fak friend of the fomly hae tC'


def test_line_example_prefix_search_heuristic_split(line_mat, labels):
    mat = line_mat
    assert prefix_search_heuristic_split(mat, labels) == 'the fak friend of the fomcly hae tC'


def test_line_example_beam_search(line_mat, labels):
    mat = line_mat
    assert beam_search(mat, labels) == 'the fak friend of the fomcly hae tC'


def test_line_example_beam_search_with_language_model(line_mat, labels, corpus):
    mat = line_mat

    # create language model from text corpus
    lm = LanguageModel(corpus, labels)

    assert beam_search(mat, labels, lm=lm) == 'the fake friend of the family, lie th'


def test_line_example_token_passing(line_mat, labels, corpus):
    mat = line_mat

    # create language model from text corpus
    words = re.findall(r'\w+', corpus)

    assert token_passing(mat, labels, words) == 'the fake friend of the family fake the'


def test_line_example_loss_and_probability(line_mat, labels):
    mat = line_mat
    gt = 'the fake friend of the family, like the'

    assert np.isclose(probability(mat, gt, labels), 6.31472642886565e-13)
    assert np.isclose(loss(mat, gt, labels), 28.090721774903226)


def test_word_example_best_path(word_mat, labels, words):
    mat = word_mat
    assert best_path(mat, labels) == 'aircrapt'


def test_word_example_lexicon_search(word_mat, labels, words):
    mat = word_mat

    # create BK tree from list of words
    bk_tree = BKTree(words)

    assert lexicon_search(mat, labels, bk_tree, tolerance=4) == 'aircraft'
