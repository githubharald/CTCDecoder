import editdistance as ed

from ctc_decoder import BKTree


def test_bk_tree():
    "test BK tree on words from corpus"
    with open('../data/word/corpus.txt') as f:
        words = f.read().split()

    tolerance = 2
    t = BKTree(words)
    q = 'air'
    actual = sorted(t.query(q, tolerance))
    expected = sorted([w for w in words if ed.eval(q, w) <= tolerance])
    assert actual == expected
