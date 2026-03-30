"""
tree.py  –  Token tree data structures for SpecInfer.

Terminology
-----------
virtual root   : depth-0 sentinel node (token_id=-1) that represents
                 "end of prompt".  It holds no real token.
draft node     : any node at depth ≥ 1; each represents one candidate
                 token predicted by the draft model.
draft_ancestors: the chain of draft nodes from depth-1 down to (and
                 including) a given node.  The virtual root is excluded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TreeNode:
    """
    One node in the speculative token tree.

    Attributes
    ----------
    token_id   : candidate token (-1 for the virtual root)
    parent     : parent node (None only for the virtual root)
    children   : child nodes; populated by TokenTree.expand()
    depth      : 0 = virtual root | 1 = first draft token | …
    draft_prob : P(token_id | ancestor context, draft model)
                 Used in the MSS acceptance criterion.
    """

    token_id: int
    parent: Optional[TreeNode] = None
    children: List[TreeNode] = field(default_factory=list)
    depth: int = 0
    draft_prob: float = 1.0

    # ------------------------------------------------------------------ helpers

    def is_virtual_root(self) -> bool:
        return self.depth == 0

    def draft_ancestors(self) -> List[TreeNode]:
        """
        Return all real (non-virtual) nodes on the root→self path, inclusive.
        Ordered shallowest → deepest.
        """
        path: List[TreeNode] = []
        node: Optional[TreeNode] = self
        while node is not None and not node.is_virtual_root():
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def draft_token_sequence(self) -> List[int]:
        """Token IDs along the root→self path (the draft prefix up to this node)."""
        return [n.token_id for n in self.draft_ancestors()]


class TokenTree:
    """
    Container for a complete speculative token tree.

    Layout
    ------
    depth 0 : virtual root (no real token; represents end of prompt)
    depth 1 : branching_factor candidates for position P
    depth 2 : branching_factor^2 candidates for position P+1
    …

    Usage
    -----
    tree = TokenTree()
    children = tree.expand(tree.root, [tok_a, tok_b], [0.6, 0.4])
    grand    = tree.expand(children[0], [tok_c, tok_d], [0.7, 0.3])
    """

    def __init__(self) -> None:
        self.root = TreeNode(token_id=-1, depth=0)   # virtual root

    # ------------------------------------------------------------------ mutate

    def expand(
        self,
        parent: TreeNode,
        token_ids: List[int],
        probs: List[float],
    ) -> List[TreeNode]:
        """
        Attach len(token_ids) child nodes to `parent` and return them.
        Called by DraftModel.build_tree() level-by-level.
        """
        children: List[TreeNode] = []
        for tid, p in zip(token_ids, probs):
            child = TreeNode(
                token_id=tid,
                parent=parent,
                depth=parent.depth + 1,
                draft_prob=p,
            )
            parent.children.append(child)
            children.append(child)
        return children

    # ------------------------------------------------------------------ query

    def bfs_nodes(self) -> List[TreeNode]:
        """All draft nodes (virtual root excluded) in breadth-first order."""
        result: List[TreeNode] = []
        queue: List[TreeNode] = list(self.root.children)
        while queue:
            node = queue.pop(0)
            result.append(node)
            queue.extend(node.children)
        return result

    def leaves(self) -> List[TreeNode]:
        """Leaf nodes (draft nodes without children)."""
        return [n for n in self.bfs_nodes() if not n.children]

    def max_depth(self) -> int:
        """Maximum depth of any draft node (0 if tree is empty)."""
        nodes = self.bfs_nodes()
        return max((n.depth for n in nodes), default=0)

    def size(self) -> int:
        """Total number of draft nodes."""
        return len(self.bfs_nodes())
