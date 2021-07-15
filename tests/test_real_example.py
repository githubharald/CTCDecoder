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
    "calc softmax such that labels per time-step form probability distribution"
    maxT, _ = mat.shape  # dim0=t, dim1=c
    res = np.zeros(mat.shape)
    for t in range(maxT):
        y = mat[t, :]
        e = np.exp(y)
        s = np.sum(e)
        res[t, :] = e / s
    return res


def load_rnn_output(fn):
    "load RNN output from csv file. Last entry in row terminated by semicolon."
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


def test_line_example(line_mat, labels):
    mat = line_mat
    with open('../data/line/corpus.txt') as f:
        txt = f.read()
    lm = LanguageModel(txt, labels)
    gt = 'the fake friend of the family, like the'

    assert best_path(mat, labels) == 'the fak friend of the fomly hae tC'
    assert prefix_search_heuristic_split(mat, labels) == 'the fak friend of the fomcly hae tC'
    assert beam_search(mat, labels) == 'the fak friend of the fomcly hae tC'
    assert beam_search(mat, labels, lm=lm) == 'the fake friend of the family, lie th'
    assert token_passing(mat, labels, lm) == 'the fake friend of the family fake the'
    assert np.isclose(probability(mat, gt, labels), 6.31472642886565e-13)
    assert np.isclose(loss(mat, gt, labels), 28.090721774903226)


def test_word_example(word_mat, labels):
    mat = word_mat
    with open('../data/word/corpus.txt') as f:
        words = f.read().split()
    tolerance = 4

    assert best_path(mat, labels) == 'aircrapt'
    assert lexicon_search(mat, labels, BKTree(words), tolerance) == 'aircraft'
