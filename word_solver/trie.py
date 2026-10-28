"""Prefix tree (Trie) used to prune the board search.

Each node represents a prefix. Walking from the root through the children for
"C", "A", "T" reaches the node for the prefix "CAT"; if that node has
``is_word`` set, "CAT" is a dictionary word. If a child is missing, no
dictionary word starts with that prefix, so the search can stop immediately.
"""

from __future__ import annotations

from typing import Iterable


class TrieNode:
    """A single prefix in the Trie."""

    __slots__ = ("children", "is_word")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_word: bool = False

    def child(self, letter: str) -> TrieNode | None:
        """Return the node for this prefix + ``letter``, or None if no word continues that way."""
        return self.children.get(letter)


class Trie:
    """A set of words stored as a prefix tree."""

    def __init__(self, words: Iterable[str] = ()) -> None:
        self.root = TrieNode()
        self._size = 0
        for word in words:
            self.insert(word)

    def __len__(self) -> int:
        return self._size

    def __contains__(self, word: str) -> bool:
        return self.contains(word)

    def insert(self, word: str) -> None:
        """Add ``word`` to the Trie. Inserting an existing word is a no-op."""
        node = self.root
        for letter in word:
            nxt = node.children.get(letter)
            if nxt is None:
                nxt = TrieNode()
                node.children[letter] = nxt
            node = nxt
        if not node.is_word:
            node.is_word = True
            self._size += 1

    def find_node(self, prefix: str) -> TrieNode | None:
        """Return the node reached by ``prefix``, or None if no word has that prefix."""
        node: TrieNode | None = self.root
        for letter in prefix:
            node = node.children.get(letter)
            if node is None:
                return None
        return node

    def contains(self, word: str) -> bool:
        """True if ``word`` is a complete word in the Trie."""
        node = self.find_node(word)
        return node is not None and node.is_word

    def has_prefix(self, prefix: str) -> bool:
        """True if at least one word in the Trie starts with ``prefix``."""
        return self.find_node(prefix) is not None
