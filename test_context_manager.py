import sqlite3
import unittest

from context_manager import compact, load_stored_entries, persist_entries


class ContextCompactionTests(unittest.TestCase):
    def setUp(self):
        self.messages = [{"role": "system", "content": "Stable Studio system prompt"}]
        for index in range(18):
            self.messages.append({"role": "user", "content": f"Task {index} " + "detail " * 500})
            self.messages.append({"role": "assistant", "content": f"Result {index} " + "output " * 500})
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("""CREATE TABLE context_summaries (
            session_id TEXT, turn_key TEXT, turn_index INTEGER, summary TEXT,
            tokens INTEGER, created_at REAL, PRIMARY KEY(session_id, turn_key))""")
        self.conn.row_factory = sqlite3.Row

    def tearDown(self):
        self.conn.close()

    def test_compacts_at_threshold_and_persists_reusable_digests(self):
        report = compact(self.messages, self.messages[0]["content"], None,
                         12000, 78, 1024, window_source="test",
                         digest_role="system", min_recent_messages=2)
        self.assertTrue(report["compacted"])
        self.assertGreater(report["turns_folded"], 0)
        self.assertLess(len(report["messages"]), len(self.messages))
        # The active exchange is preserved verbatim while older turns are summarized.
        self.assertEqual(report["messages"][-2:], self.messages[-2:])
        self.assertGreater(persist_entries(self.conn, "session", report["entries"]), 0)

        stored = load_stored_entries(self.conn, "session")
        replay = compact(self.messages, self.messages[0]["content"], None,
                         12000, 78, 1024, window_source="test",
                         digest_role="system", min_recent_messages=2,
                         stored_entries=stored)
        self.assertEqual(replay["stored_reused"], len(stored))
        self.assertFalse(replay["compacted"])
        self.assertLess(len(replay["messages"]), len(self.messages))


if __name__ == "__main__":
    unittest.main()
