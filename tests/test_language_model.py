from string import ascii_letters

from ctc_decoder import LanguageModel


def test_word_list():
    lm = LanguageModel('This is a random string 12345.', ascii_letters)
    assert lm.get_word_list() == ['This', 'is', 'a', 'random', 'string']


def test_char_bigram():
    lm = LanguageModel('aab abc', 'ab')
    assert lm.get_char_bigram('a', 'a') == 1 / 3
    assert lm.get_char_bigram('a', 'b') == 2 / 3
    assert lm.get_char_bigram('b', 'a') == 0
