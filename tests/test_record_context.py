"""Real read-only SQLite tests for richer model context; no model required."""
import json
from pathlib import Path
import sqlite3
import tempfile
import types
import unittest
from rag.record_context import fetch_public_rows, record_excerpt, field_samples
from rag.domain_config import RAGUnavailableError


class RecordContextTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/'records.db'
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE records (id TEXT, topic TEXT, description TEXT, fee TEXT, email TEXT, _raw_json TEXT)')
            db.execute('INSERT INTO records VALUES (?,?,?,?,?,?)',('one','Library','3 books, 14 days','0','private@example.com','PRIVATE RAW'))
        self.config=types.SimpleNamespace(name='demo',db_path=self.path,table='records',id_column='id',
            primary_entity='topic',evidence_fields=['topic','description','fee'],
            schema={'topic':'Topic','description':'Content','fee':'Fee'})

    def test_read_returns_only_public_columns_and_preserves_database(self):
        before=self.path.read_bytes()
        rows=fetch_public_rows(['one','one'],self.config)
        self.assertEqual(rows,{'one':{'topic':'Library','description':'3 books, 14 days','fee':'0'}})
        self.assertEqual(self.path.read_bytes(),before)
        self.assertNotIn('email',json.dumps(rows))
        self.assertNotIn('PRIVATE',json.dumps(rows))

    def test_missing_database_is_not_created(self):
        self.config.db_path=self.path.with_name('missing.db')
        with self.assertRaises(RAGUnavailableError):fetch_public_rows(['one'],self.config)
        self.assertFalse(self.config.db_path.exists())

    def test_invalid_identifiers_rejected(self):
        self.config.table='records; DROP TABLE records'
        with self.assertRaises(RAGUnavailableError):fetch_public_rows(['one'],self.config)
        with sqlite3.connect(self.path) as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM records').fetchone()[0],1)

    def test_excerpt_and_samples_are_bounded_and_exclude_private_values(self):
        row={'topic':'Library','description':'x'*5000,'fee':'0','email':'PRIVATE'}
        text=record_excerpt(row,self.config,max_chars=600)
        self.assertLessEqual(len(text),600)
        self.assertIn('Topic: Library',text)
        self.assertNotIn('PRIVATE',text)
        samples=field_samples([row]*10,self.config)
        self.assertLessEqual(len(samples['description']),2)
        self.assertLessEqual(len(samples['description'][0]),180)
        self.assertNotIn('email',samples)

    def test_unknown_record_is_absent_not_substituted(self):
        self.assertEqual(fetch_public_rows(['unknown'],self.config),{})


if __name__=='__main__':unittest.main()
