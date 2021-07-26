import numpy as np


def prefix_search(mat: np.ndarray, chars: str) -> str:
    """Prefix search decoding.

    See dissertation of Graves, p63-66.

    Args:
        mat: Output of neural network of shape TxC.
        chars: The set of characters the neural network can recognize, excluding the CTC-blank.

    Returns:
        The decoded text.
    """

    blank_idx = len(chars)
    max_T, max_C = mat.shape

    # g_n and g_b: gamma in paper
    g_n = []
    g_b = []

    # p(y|x) and p(y...|x), where y is a prefix (not p as in paper to avoid confusion with probability)
    prob = {}
    prob_ext = {}

    # Init: 1-6
    for t in range(max_T):
        g_n.append({'': 0})
        last = g_b[t - 1][''] if t > 0 else 1
        g_b.append({'': last * mat[t, blank_idx]})

    # init for empty prefix
    prob[''] = g_b[max_T - 1]['']
    prob_ext[''] = 1 - prob['']
    l_star = y_star = ''
    Y = {''}

    # Algorithm: 8-31
    while prob_ext[y_star] > prob[l_star]:
        prob_remaining = prob_ext[y_star]

        # for all chars
        for k in range(max_C - 1):
            y = y_star + chars[k]
            g_n[0][y] = mat[0, k] if len(y_star) == 0 else 0
            g_b[0][y] = 0
            prefix_prob = g_n[0][y]

            # for all time steps
            for t in range(1, max_T):
                new_label_prob = g_b[t - 1][y_star] + (
                    0 if y_star != '' and y_star[-1] == chars[k] else g_n[t - 1][y_star])
                g_n[t][y] = mat[t, k] * (new_label_prob + g_n[t - 1][y])
                g_b[t][y] = mat[t, blank_idx] * (g_b[t - 1][y] + g_n[t - 1][y])
                prefix_prob += mat[t, k] * new_label_prob

            prob[y] = g_n[max_T - 1][y] + g_b[max_T - 1][y]
            prob_ext[y] = prefix_prob - prob[y]
            prob_remaining -= prob_ext[y]

            if prob[y] > prob[l_star]:
                l_star = y
            if prob_ext[y] > prob[l_star]:
                Y.add(y)
            if prob_remaining <= prob[l_star]:
                break

        # 30
        Y.remove(y_star)

        # 31
        best_y = None
        best_prob_ext = 0
        for y in Y:
            if prob_ext[y] > best_prob_ext:
                best_prob_ext = prob_ext[y]
                best_y = y
        y_star = best_y

        # terminate if no more prefix exists
        if best_y is None:
            break

    # Termination: 33-34
    return l_star


def prefix_search_heuristic_split(mat: np.ndarray, chars: str) -> str:
    """Prefix search decoding with heuristic to speed up the algorithm.

    Speed up prefix computation by splitting sequence into subsequences as described by Graves (p66).

    Args:
        mat: Output of neural network of shape TxC.
        chars: The set of characters the neural network can recognize, excluding the CTC-blank.

    Returns:
        The decoded text.
    """

    blank_idx = len(chars)
    max_T, _ = mat.shape

    # split sequence into 3 subsequences, splitting points should be roughly placed at 1/3 and 2/3
    split_targets = [int(max_T * 1 / 3), int(max_T * 2 / 3)]
    best = [{'target': s, 'bestDist': max_T, 'bestIdx': s} for s in split_targets]

    # find good splitting points (blanks above threshold)
    thres = 0.9
    for t in range(max_T):
        for b in best:
            if mat[t, blank_idx] > thres and abs(t - b['target']) < b['bestDist']:
                b['bestDist'] = abs(t - b['target'])
                b['bestIdx'] = t
                break

    # splitting points plus begin and end of sequence
    ranges = [0] + [b['bestIdx'] for b in best] + [max_T]

    # do prefix search for each subsequence and concatenate results
    res = ''
    for i in range(len(ranges) - 1):
        beg = ranges[i]
        end = ranges[i + 1]
        res += prefix_search(mat[beg: end, :], chars)

    return res
