from ctc_decoder import LanguageModel


def test_char_bigram():
    lm = LanguageModel('aab abc', 'ab')
    assert lm.get_char_bigram('a', 'a') == 1 / 3
    assert lm.get_char_bigram('a', 'b') == 2 / 3
    assert lm.get_char_bigram('b', 'a') == 0
