from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401

from threadsieve.archive import archive_thread
from threadsieve.extractor import extract_items
from threadsieve.importers import import_text
from threadsieve.index import index_object, index_thread, search
from threadsieve.writer import write_item


class PipelineTests(unittest.TestCase):
    def test_offline_extract_writes_and_indexes(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            thread = import_text(
                "User: Can this preserve source links?\nAssistant: Yes, cite message spans in every extracted object.",
                title="Source links",
            )
            source_dir = archive_thread(tmp_path, thread)
            index_thread(tmp_path, thread, source_dir)

            items = extract_items(thread, {"provider": "offline"}, threshold=0.5)

            self.assertTrue(items)
            note_path = write_item(tmp_path, items[0], thread, source_dir)
            index_object(tmp_path, items[0], thread, note_path)

            matches = search(tmp_path, "source")
            self.assertTrue(matches)
            self.assertEqual(matches[0]["local_path"], str(note_path))


if __name__ == "__main__":
    unittest.main()
