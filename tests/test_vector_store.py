import os
import tempfile
import unittest

import numpy as np

from src.rag.vector_store import VectorStore


class VectorStoreContainmentTests(unittest.TestCase):
    def make_store(self, terms):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        store = VectorStore(
            vector_path=os.path.join(temp_dir.name, "vector_index.npy"),
            terms_path=os.path.join(temp_dir.name, "terms_index.json"),
            embed_dim=2,
        )
        store.terms = list(terms)
        store._rebuild_lexical_index()
        return store

    def test_containment_matches_normalized_punctuation(self):
        store = self.make_store(["Blue-Palace Key", "Temple of Miraak"])

        hits = store.search_containment("blue palace", top_k=5)

        self.assertEqual(hits, [(0, "Blue-Palace Key")])

    def test_containment_keeps_single_token_substring_recall(self):
        store = self.make_store(["Scorched Dragonbone", "Dragon"])
        similarities = np.array([0.9, 0.1], dtype=np.float32)

        hits = store.search_containment("dragon", top_k=5, similarities=similarities)

        self.assertEqual(hits, [(0, "Scorched Dragonbone"), (1, "Dragon")])


if __name__ == "__main__":
    unittest.main()