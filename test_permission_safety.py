import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import agent_tools


class PermissionSafetyTests(unittest.TestCase):
    def setUp(self):
        self.original = agent_tools.get_workspace_dir()
        self.root = tempfile.mkdtemp(prefix='colibri-permission-')
        agent_tools.set_workspace_dir(self.root)

    def tearDown(self):
        agent_tools.set_workspace_dir(self.original)
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def test_inspection_reports_owner_group_mode_and_access(self):
        target = Path(self.root, 'sample.txt')
        target.write_text('safe', encoding='utf-8')
        target.chmod(0o640)
        result = agent_tools.inspect_workspace_permissions('sample.txt')
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['mode'], '0640')
        self.assertIn('owner', result)
        self.assertIn('group', result)
        self.assertEqual(target.stat().st_mode & 0o777, 0o640)

    def test_repair_requires_explicit_approval(self):
        target = Path(self.root, 'sample.txt')
        target.write_text('safe', encoding='utf-8')
        target.chmod(0o600)
        result = agent_tools.repair_workspace_metadata('sample.txt', 'chmod', mode='0644')
        self.assertEqual(result['status'], 'confirmation_required')
        self.assertEqual(target.stat().st_mode & 0o777, 0o600)

    def test_repair_rejects_recursive_or_special_modes_and_other_actions(self):
        target = Path(self.root, 'sample.txt')
        target.write_text('safe', encoding='utf-8')
        for mode in ('07777', '8888', '7777'):
            result = agent_tools.repair_workspace_metadata('sample.txt', 'chmod', approved=True, mode=mode)
            self.assertEqual(result['status'], 'error')
        result = agent_tools.repair_workspace_metadata('sample.txt', 'delete', approved=True)
        self.assertEqual(result['status'], 'error')

    def test_compatibility_normalizer_is_noop(self):
        target = Path(self.root, 'sample.txt')
        target.write_text('safe', encoding='utf-8')
        target.chmod(0o640)
        agent_tools._normalize_permissions(str(target))
        self.assertEqual(target.stat().st_mode & 0o777, 0o640)

    def test_repair_uses_pkexec_only_for_metadata(self):
        target = Path(self.root, 'sample.txt')
        target.write_text('safe', encoding='utf-8')
        with mock.patch.object(agent_tools.shutil, 'which', return_value='/usr/bin/pkexec'), \
             mock.patch.object(agent_tools.subprocess, 'run') as run:
            run.return_value.returncode = 0
            run.return_value.stdout = ''
            run.return_value.stderr = ''
            result = agent_tools.repair_workspace_metadata('sample.txt', 'chmod', approved=True, mode='0640')
        self.assertEqual(result['status'], 'success')
        args = run.call_args.args[0]
        self.assertEqual(args[0], 'pkexec')
        self.assertIn('/usr/bin/chmod', args)
        self.assertFalse(run.call_args.kwargs.get('shell', True))


if __name__ == '__main__':
    unittest.main()
