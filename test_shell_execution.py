import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agent_tools


class ShellExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "subdir").mkdir()
        self.workspace_patch = patch.object(agent_tools, "WORKSPACE_DIR", str(self.root))
        self.workspace_patch.start()

    def tearDown(self):
        self.workspace_patch.stop()
        self.tmp.cleanup()

    def test_runs_command_in_selected_workspace_subdirectory(self):
        result = agent_tools.execute_terminal("pwd", cwd="subdir")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["cwd"], str(self.root / "subdir"))
        self.assertIn(str(self.root / "subdir"), result["output"])

    def test_rejects_working_directory_escape(self):
        result = agent_tools.execute_terminal("pwd", cwd="..")
        self.assertEqual(result["status"], "error")
        self.assertIn("Invalid working directory", result["output"])

    def test_destructive_command_requires_confirmation(self):
        result = agent_tools.execute_terminal("rm -rf definitely-not-run")
        self.assertEqual(result["status"], "confirmation_required")
        self.assertTrue(result["requires_confirmation"])

    def test_confirmed_deletion_is_scoped_to_test_workspace(self):
        target = self.root / "temporary.txt"
        target.write_text("test", encoding="utf-8")
        result = agent_tools.execute_terminal("rm temporary.txt", confirm=True)
        self.assertEqual(result["status"], "success")
        self.assertFalse(target.exists())

    def test_timeout_returns_timeout_status(self):
        result = agent_tools.execute_terminal("sleep 3", timeout=1)
        self.assertEqual(result["status"], "timeout")
        self.assertTrue(result["timed_out"])

    def test_output_is_bounded(self):
        result = agent_tools.execute_terminal("python -c 'print(\"x\" * 5000)'", max_output_chars=1000)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["output_truncated"])
        self.assertLess(len(result["output"]), 1200)


if __name__ == "__main__":
    unittest.main()
