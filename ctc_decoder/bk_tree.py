from typing import List

import editdistance as ed


class BKTree:
    """Burkhard Keller tree: used to find strings within tolerance (w.r.t. edit distance metric)
    to given query string."""

    def __init__(self, txt_list: List[str]) -> None:
        """Pass list of texts (words) which are inserted into the tree."""
        self.root = None
        for txt in txt_list:
            self._insert(self.root, txt)

    def query(self, txt: str, tolerance: int) -> List[str]:
        """Query strings within given tolerance (w.r.t. edit distance metric)."""
        return self._query(self.root, txt, tolerance)

    def _insert(self, node, txt):
        # insert root node
        if node is None:
            self.root = (txt, {})
            return

        # insert all other nodes
        d = ed.eval(node[0], txt)
        if d in node[1]:
            self._insert(node[1][d], txt)
        else:
            node[1][d] = (txt, {})

    def _query(self, node, txt, tolerance):
        # handle empty root node
        if node is None:
            return []

        # distance between query and current node
        d = ed.eval(node[0], txt)

        # add current node to result if within tolerance
        res = []
        if d <= tolerance:
            res.append(node[0])

        # iterate over children
        for (edge, child) in node[1].items():
            if d - tolerance <= edge <= d + tolerance:
                res += self._query(child, txt, tolerance)

        return res
