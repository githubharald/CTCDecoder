import math
from typing import List

import numpy as np

from ctc_decoder import common


class Token:
    """Token for token passing algorithm. Each token contains a score and a history of visited words."""

    def __init__(self, score=float('-inf'), history=None):
        self.score = score
        self.history = history if history else []

    def __str__(self):
        res = 'class Token: ' + str(self.score) + '; '
        for w in self.history:
            res += w + '; '
        return res


class TokenList:
    """This class simplifies getting/setting tokens."""

    def __init__(self):
        self.tokens = {}

    def set(self, w, s, t, tok):
        self.tokens[(w, s, t)] = tok

    def get(self, w, s, t):
        return self.tokens[(w, s, t)]

    def dump(self, s, t):
        for (k, v) in self.tokens.items():
            if k[1] == s and k[2] == t:
                print(k, v)


def output_indices(toks, words, s, t):
    """argmax_w tok(w,s,t)."""
    res = []
    for (wIdx, _) in enumerate(words):
        res.append(toks.get(wIdx, s, t))

    idx = [i[0] for i in sorted(enumerate(res), key=lambda x: x[1].score)]
    return idx


def log(val):
    """Return -inf for log(0) instead of throwing error like python implementation does it."""
    if val > 0:
        return math.log(val)
    return float('-inf')


def token_passing(mat: np.ndarray, chars: str, words: List[str]) -> str:
    """Token passing algorithm.

    See dissertation of Graves, p67-69.

    Args:
        mat: Output of neural network of shape TxC.
        chars: The set of characters the neural network can recognize, excluding the CTC-blank.
        words: List of words that can be recognized.

    Returns:
        The decoded text.
    """

    blank_idx = len(chars)
    max_T, _ = mat.shape

    # special s index for beginning and end of word
    beg = 0
    end = -1

    # map characters to labels for each word
    label_words = [common.word_to_label_seq(w, chars) for w in words]

    # w' in paper: word with blanks in front, back and between labels: for -> _f_o_r_
    prime_words = [common.extend_by_blanks(w, blank_idx) for w in label_words]

    # data structure holding all tokens
    toks = TokenList()

    # Initialisation: 1-9
    for w_idx, w in enumerate(label_words):
        w = label_words[w_idx]
        w_prime = prime_words[w_idx]

        # set all toks(w,s,t) to init state
        for s in range(len(w_prime)):
            for t in range(max_T):
                toks.set(w_idx, s + 1, t + 1, Token())
                toks.set(w_idx, beg, t, Token())
                toks.set(w_idx, end, t, Token())

        toks.set(w_idx, 1, 1, Token(log(mat[1 - 1, blank_idx]), [w_idx]))
        c_idx = w[1 - 1]
        toks.set(w_idx, 2, 1, Token(log(mat[1 - 1, c_idx]), [w_idx]))

        if len(w) == 1:
            toks.set(w_idx, end, 1, toks.get(w_idx, 2, 1))

    # Algorithm: 11-24
    t = 2
    while t <= max_T:

        sorted_word_idx = output_indices(toks, label_words, end, t - 1)

        for w_idx in sorted_word_idx:
            w_prime = prime_words[w_idx]

            # 15-17
            # if bigrams should be used, these lines have to be adapted
            best_output_tok = toks.get(sorted_word_idx[-1], end, t - 1)
            toks.set(w_idx, beg, t, Token(best_output_tok.score, best_output_tok.history + [w_idx]))

            # 18-24
            s = 1
            while s <= len(w_prime):
                if s == 1:
                    P = [toks.get(w_idx, s, t - 1), toks.get(w_idx, s - 1, t)]
                else:
                    P = [toks.get(w_idx, s, t - 1), toks.get(w_idx, s - 1, t - 1)]

                if w_prime[s - 1] != blank_idx and s > 2 and w_prime[s - 2 - 1] != w_prime[s - 1]:
                    tok = toks.get(w_idx, s - 2, t - 1)
                    P.append(Token(tok.score, tok.history))

                max_tok = sorted(P, key=lambda x: x.score)[-1]
                c_idx = w_prime[s - 1]

                score = max_tok.score + log(mat[t - 1, c_idx])
                history = max_tok.history

                toks.set(w_idx, s, t, Token(score, history))
                s += 1

            max_tok = sorted([toks.get(w_idx, len(w_prime), t), toks.get(w_idx, len(w_prime) - 1, t)],
                             key=lambda x: x.score,
                             reverse=True)[0]
            toks.set(w_idx, end, t, max_tok)

        t += 1

    # Termination: 26-28
    best_w_idx = output_indices(toks, label_words, end, max_T)[-1]
    return str(' ').join([words[i] for i in toks.get(best_w_idx, end, max_T).history])
