def extend_by_blanks(seq, b):
    "extends a label seq. by adding blanks at the beginning, end and in between each label"
    res = [b]
    for s in seq:
        res.append(s)
        res.append(b)
    return res


def word_to_label_seq(w, labels):
    "map a word to a sequence of labels (indices)"
    res = [labels.index(c) for c in w]
    return res
