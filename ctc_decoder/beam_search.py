from typing import Optional

import numpy as np

from ctc_decoder.language_model import LanguageModel


class BeamEntry:
    """Information about one single beam at specific time-step."""

    def __init__(self):
        self.pr_total = 0  # blank and non-blank
        self.pr_non_blank = 0  # non-blank
        self.pr_blank = 0  # blank
        self.pr_text = 1  # LM score
        self.lm_applied = False  # flag if LM was already applied to this beam
        self.labeling = ()  # beam-labeling


class BeamState:
    """Information about all beams at specific time-step."""

    def __init__(self):
        self.entries = {}

    def norm(self):
        """Length-normalise LM score."""
        for k in self.entries.keys():
            labeling_len = len(self.entries[k].labeling)
            self.entries[k].pr_text = self.entries[k].pr_text ** (1.0 / (labeling_len if labeling_len else 1.0))

    def sort(self):
        """Return beam-labelings, sorted by probability."""
        beams = [v for (_, v) in self.entries.items()]
        sorted_beams = sorted(beams, reverse=True, key=lambda x: x.pr_total * x.pr_text)
        return [x.labeling for x in sorted_beams]


def apply_lm(parent_beam, child_beam, chars, lm):
    """Calculate LM score of child beam by taking score from parent beam and bigram probability of last two chars."""
    if lm and not child_beam.lm_applied:
        c1 = chars[parent_beam.labeling[-1] if parent_beam.labeling else chars.index(' ')]  # first char
        c2 = chars[child_beam.labeling[-1]]  # second char
        lm_factor = 0.01  # influence of language model
        bigram_prob = lm.get_char_bigram(c1, c2) ** lm_factor
        child_beam.pr_text = parent_beam.pr_text * bigram_prob  # probability of char sequence
        child_beam.lm_applied = True  # only apply LM once per beam entry


def add_beam(beam_state, labeling):
    """Add beam if it does not yet exist."""
    if labeling not in beam_state.entries:
        beam_state.entries[labeling] = BeamEntry()


def beam_search(mat: np.ndarray, chars: str, beam_width: int = 25, lm: Optional[LanguageModel] = None) -> str:
    """Beam search decoder.

    See the paper of Hwang et al. and the paper of Graves et al.

    Args:
        mat: Output of neural network of shape TxC.
        chars: The set of characters the neural network can recognize, excluding the CTC-blank.
        beam_width: Number of beams kept per iteration.
        lm: Character level language model if specified.

    Returns:
        The decoded text.
    """

    blank_idx = len(chars)
    max_T, max_C = mat.shape

    # initialise beam state
    last = BeamState()
    labeling = ()
    last.entries[labeling] = BeamEntry()
    last.entries[labeling].pr_blank = 1
    last.entries[labeling].pr_total = 1

    # go over all time-steps
    for t in range(max_T):
        curr = BeamState()

        # get beam-labelings of best beams
        best_labelings = last.sort()[0:beam_width]

        # go over best beams
        for labeling in best_labelings:

            # probability of paths ending with a non-blank
            pr_non_blank = 0
            # in case of non-empty beam
            if labeling:
                # probability of paths with repeated last char at the end
                pr_non_blank = last.entries[labeling].pr_non_blank * mat[t, labeling[-1]]

            # probability of paths ending with a blank
            pr_blank = last.entries[labeling].pr_total * mat[t, blank_idx]

            # add beam at current time-step if needed
            add_beam(curr, labeling)

            # fill in data
            curr.entries[labeling].labeling = labeling
            curr.entries[labeling].pr_non_blank += pr_non_blank
            curr.entries[labeling].pr_blank += pr_blank
            curr.entries[labeling].pr_total += pr_blank + pr_non_blank
            curr.entries[labeling].pr_text = last.entries[labeling].pr_text
            curr.entries[labeling].lm_applied = True  # LM already applied at previous time-step for this beam-labeling

            # extend current beam-labeling
            for c in range(max_C - 1):
                # add new char to current beam-labeling
                new_labeling = labeling + (c,)

                # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                if labeling and labeling[-1] == c:
                    pr_non_blank = mat[t, c] * last.entries[labeling].pr_blank
                else:
                    pr_non_blank = mat[t, c] * last.entries[labeling].pr_total

                # add beam at current time-step if needed
                add_beam(curr, new_labeling)

                # fill in data
                curr.entries[new_labeling].labeling = new_labeling
                curr.entries[new_labeling].pr_non_blank += pr_non_blank
                curr.entries[new_labeling].pr_total += pr_non_blank

                # apply LM
                apply_lm(curr.entries[labeling], curr.entries[new_labeling], chars, lm)

        # set new beam state
        last = curr

    # normalise LM scores according to beam-labeling-length
    last.norm()

    # sort by probability
    best_labeling = last.sort()[0]  # get most probable labeling

    # map label string to char string
    res = ''.join([chars[l] for l in best_labeling])
    return res
