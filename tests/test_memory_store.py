import sqlite3
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from nanbeige_memory import MAX_ENTRY_CHARS, MAX_NOTES, MAX_SAVED_TURNS, MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def test_private_file_reopens_and_forgetting_persists(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "new folder" / "memory #1.sqlite"
            with MemoryStore(path) as store:
                note = store.remember("My favorite color is aubergine.")
                store.record("The server is named Puffin.", "Puffin.")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

            with MemoryStore(path) as store:
                self.assertEqual(store.notes(), [(note, "My favorite color is aubergine.")])
                self.assertIn("aubergine", store.recall("What is my favorite color?")[0])
                self.assertIn("Puffin", store.recall("What is the server's name?")[0])
                self.assertTrue(store.forget(note))
                self.assertFalse(store.forget(note))
                self.assertEqual(store.history(), [("The server is named Puffin.", "Puffin.")])
                store.clear()
            with MemoryStore(path) as store:
                self.assertEqual(store.recall("server color"), [])

    def test_latest_corrections_and_relevance_outrank_unrelated_turns(self):
        with MemoryStore(None) as store:
            store.record("The project codename is Alder.", "Okay.")
            store.record("The project codename is Juniper.", "Okay.")
            store.record("Explain how an ocarina works.", "An ocarina is a wind instrument.")
            recall = store.recall("What is the project codename?", limit=2)
            self.assertEqual(len(recall), 2)
            self.assertIn("Juniper", recall[0])
            self.assertIn("Alder", recall[1])
            self.assertIn("Alder", store.recall("project codename", exclude=[
                ("The project codename is Juniper.", "Okay."),
            ])[0])
            store.remember("The project codename is now Sequoia.")
            self.assertIn("Sequoia", store.recall("project codename")[0])

    def test_chinese_queries_can_match_parts_of_a_sentence(self):
        with MemoryStore(None) as store:
            store.record("我最喜欢的水果是芒果。", "明白了。")
            store.record("这是一条无关的信息。", "好的。")
            self.assertIn("芒果", store.recall("我的水果偏好是什么？")[0])

    def test_recall_can_quote_a_match_from_middle_of_long_note(self):
        with MemoryStore(None) as store:
            store.remember("Nondescript " * 60 + "The project codename is Herringbone. " + "More filler " * 60)
            excerpt = store.recall("What is the project codename?")[0]
            self.assertIn("The project codename is Herringbone.", excerpt)
            self.assertLess(len(excerpt), 400)

    def test_bounds_automatic_history_without_evicting_explicit_notes(self):
        with MemoryStore(None) as store:
            note = store.remember("Keep this note.")
            for index in range(MAX_SAVED_TURNS):
                store.record(f"Turn {index}", "Okay.")
            prompt = "start " + "x" * MAX_ENTRY_CHARS + " end"
            store.record(prompt, "also kept")
            history = store.history()
            self.assertEqual(len(history), MAX_SAVED_TURNS)
            self.assertEqual(history[0], ("Turn 1", "Okay."))
            self.assertLessEqual(len(history[-1][0]), MAX_ENTRY_CHARS)
            self.assertTrue(history[-1][0].endswith(" end"))
            self.assertEqual(store.notes(), [(note, "Keep this note.")])
            self.assertFalse(any("start" in entry for entry in store.recall(
                "start end", exclude=[(prompt, "also kept")],
            )))
        with MemoryStore(None) as store:
            self.assertEqual(store.history(), [])
            self.assertEqual(store.notes(), [])

    def test_full_explicit_notes_require_a_deletion_before_adding_more(self):
        with MemoryStore(None) as store:
            first = store.remember("Original note")
            for index in range(MAX_NOTES - 1):
                store.remember(f"Note {index}")
            with self.assertRaisesRegex(ValueError, "Forget a note"):
                store.remember("New note")
            self.assertEqual(store.notes()[0], (first, "Original note"))
            self.assertTrue(store.forget(first))
            last = store.remember("New note")
            self.assertEqual(store.notes()[-1], (last, "New note"))

    def test_opening_an_unrelated_database_leaves_it_intact(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "other.sqlite"
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE notes (text)")
                connection.execute("INSERT INTO notes VALUES ('original data')")
                path.chmod(0o600)
            try:
                with self.assertRaisesRegex(ValueError, "Nanbeige memory database"):
                    MemoryStore(path)
                self.assertEqual(connection.execute("SELECT * FROM notes").fetchall(), [("original data",)])
            finally:
                connection.close()
