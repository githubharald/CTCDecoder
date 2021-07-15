import re
from typing import List


class LanguageModel:
    "Simple language model: word list for token passing, char bigrams for beam search."

    def __init__(self, txt: str, labels: str) -> None:
        """Create language model from text corpus."""
        txt = ' ' + txt + ' '  # ensure first/last characters appear next to whitespace
        self._init_word_list(txt)
        self._init_char_bigrams(txt, labels)

    def _init_word_list(self, txt):
        """Init of word list."""
        words = re.findall(r'\w+', txt)
        self.words = list(filter(lambda x: x.isalpha(), words))

    def _init_char_bigrams(self, txt, labels):
        """Init of character bigrams."""

        # init bigrams with 0 values
        self.bigram = {c: {d: 0 for d in labels} for c in labels}

        # go through text and add each char bigram
        for i in range(len(txt) - 1):
            first = txt[i]
            second = txt[i + 1]

            # ignore unknown chars
            if first not in self.bigram or second not in self.bigram[first]:
                continue

            self.bigram[first][second] += 1

    def get_char_bigram(self, first: str, second: str) -> float:
        """Probability that first character is followed by second one."""
        first = first if first else ' '  # map start to word beginning
        second = second if second else ' '  # map end to word end

        # number of bigrams starting with given char
        num_bigrams = sum(self.bigram[first].values())
        if num_bigrams == 0:
            return 0
        return self.bigram[first][second] / num_bigrams

    def get_word_list(self) -> List[str]:
        """Get list of unique words."""
        return self.words
