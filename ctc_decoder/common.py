def extend_by_blanks(seq, b):
    """Extend a label seq. by adding blanks at the beginning, end and in between each label."""
    res = [b]
    for s in seq:
        res.append(s)
        res.append(b)
    return res


def word_to_label_seq(w, chars):
    """Map a word (string of characters) to a sequence of labels (indices)."""
    res = [chars.index(c) for c in w]
    return res
