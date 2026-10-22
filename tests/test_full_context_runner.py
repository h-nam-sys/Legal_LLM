import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import testing.full_context_check as runner


class RunnerVerificationTests(unittest.TestCase):
    def archive(self,root,mode='full',field_vote=False,bad_hash=False):
        hashes={name.replace('/','\\'):value for name,value in runner.BASELINE.items()}
        if bad_hash:hashes['backend\\services\\domain_router.py']='different'
        events=[{'task':'answer.cache','context_mode':mode,'cache_hit':False},
                {'task':'rag.evidence.diagnostic','event':'complete','context_mode':mode}]
        if field_vote:events.append({'task':'rag.evidence.diagnostic','event':'field_selection','selected':['fee']})
        path=Path(root)/'run.zip'
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('run.json',json.dumps({'runtime_sha256':hashes}))
            z.writestr('logs/timing/test.log','\n'.join('date TIMING '+json.dumps(e) for e in events))
        return path

    def test_verified_full_context_archive_is_labeled(self):
        with tempfile.TemporaryDirectory() as root:
            path=self.archive(root)
            self.assertTrue(runner.annotate(path))
            with zipfile.ZipFile(path) as z:
                data=json.loads(z.read('experiment.json'))
                self.assertTrue(data['configuration_verified'])
                self.assertEqual(data['field_selection_calls'],0)

    def test_selected_mode_or_field_vote_cannot_masquerade_as_full_context(self):
        for kwargs in ({'mode':'selected'},{'field_vote':True}):
            with self.subTest(kwargs=kwargs),tempfile.TemporaryDirectory() as root:
                self.assertFalse(runner.annotate(self.archive(root,**kwargs)))

    def test_changed_routing_invalidates_the_isolated_experiment(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertFalse(runner.annotate(self.archive(root,bad_hash=True)))


if __name__=='__main__':unittest.main()
