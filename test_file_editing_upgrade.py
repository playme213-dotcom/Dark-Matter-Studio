import os
import shutil
import tempfile
import unittest
from pathlib import Path

import agent_tools


class FileEditingUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.original = agent_tools.get_workspace_dir()
        self.root = tempfile.mkdtemp(prefix='colibri-file-edit-')
        self.assertEqual(agent_tools.set_workspace_dir(self.root)['status'], 'success')

    def tearDown(self):
        agent_tools.set_workspace_dir(self.original)
        shutil.rmtree(self.root, ignore_errors=True)

    def test_read_hash_and_stale_patch_conflict(self):
        Path(self.root, 'a.py').write_text('x = 1\n', encoding='utf-8')
        read = agent_tools.read_workspace_file('a.py')
        self.assertEqual(len(read['sha256']), 64)
        Path(self.root, 'a.py').write_text('x = 2\n', encoding='utf-8')
        result = agent_tools.patch_workspace_file('a.py', 'x = 1', 'x = 3', expected_sha256=read['sha256'])
        self.assertEqual(result['status'], 'conflict')
        self.assertEqual(Path(self.root, 'a.py').read_text(), 'x = 2\n')

    def test_ambiguous_patch_rejected_then_replace_all(self):
        Path(self.root, 'a.txt').write_text('a\na\n', encoding='utf-8')
        result = agent_tools.patch_workspace_file('a.txt', 'a', 'b')
        self.assertEqual(result['status'], 'ambiguous')
        result = agent_tools.patch_workspace_file('a.txt', 'a', 'b', replace_all=True)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(Path(self.root, 'a.txt').read_text(), 'b\nb\n')
        self.assertTrue(result['backup'] and os.path.isfile(result['backup']))
        self.assertIn('--- a/a.txt', result['diff'])

    def test_create_only_prevents_overwrite(self):
        Path(self.root, 'existing.txt').write_text('keep', encoding='utf-8')
        result = agent_tools.write_workspace_file('existing.txt', 'replace', create_only=True)
        self.assertEqual(result['status'], 'error')
        self.assertEqual(Path(self.root, 'existing.txt').read_text(), 'keep')

    def test_line_edit_is_atomic_and_checks_range(self):
        Path(self.root, 'lines.txt').write_text('one\ntwo\nthree\n', encoding='utf-8')
        read = agent_tools.read_workspace_file('lines.txt')
        bad = agent_tools.edit_workspace_lines('lines.txt', 2, 9, 'bad', expected_sha256=read['sha256'])
        self.assertEqual(bad['status'], 'error')
        self.assertEqual(Path(self.root, 'lines.txt').read_text(), 'one\ntwo\nthree\n')
        good = agent_tools.edit_workspace_lines('lines.txt', 2, 2, 'TWO', expected_sha256=read['sha256'])
        self.assertEqual(good['status'], 'success')
        self.assertEqual(Path(self.root, 'lines.txt').read_text(), 'one\nTWO\nthree\n')

    def test_dispatcher_passes_edit_safety_arguments(self):
        Path(self.root, 'a.txt').write_text('before\n', encoding='utf-8')
        digest = agent_tools.read_workspace_file('a.txt')['sha256']
        _, _, result = agent_tools.dispatch_tool_call('patch_workspace_file', {
            'path': 'a.txt', 'search_pattern': 'before', 'replacement': 'after',
            'expected_sha256': digest
        })
        self.assertEqual(result['status'], 'success')


if __name__ == '__main__':
    unittest.main()
