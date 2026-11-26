"""
Comprehensive Smoke & Unit Test Suite for Colibri Autonomous Studio (test_agent_tools.py)
Run with:
    python3 -m unittest -v test_agent_tools.py
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path

# Allow tests to run even in minimal CI/CLI environments without GUI/browser packages installed
for _mod_name in ("mss", "duckduckgo_search", "playwright", "playwright.sync_api"):
    if _mod_name not in sys.modules:
        try:
            __import__(_mod_name)
        except ImportError:
            _mock_mod = types.ModuleType(_mod_name)
            if _mod_name == "duckduckgo_search":
                _mock_mod.DDGS = None
            elif _mod_name == "playwright.sync_api":
                _mock_mod.sync_playwright = None
            sys.modules[_mod_name] = _mock_mod

import agent_tools


class CodingToolsSmokeTests(unittest.TestCase):
    def setUp(self):
        self.original_workspace = agent_tools.get_workspace_dir()
        self.temp_dir = tempfile.mkdtemp(prefix="colibri-tool-smoke-")
        result = agent_tools.set_workspace_dir(self.temp_dir)
        self.assertEqual(result["status"], "success")

    def tearDown(self):
        agent_tools.set_workspace_dir(self.original_workspace)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # =================================================================
    # 1. WORKSPACE ESCAPE & PATH TRAVERSAL PROTECTION
    # =================================================================
    def test_workspace_escape_is_rejected(self):
        res_abs = agent_tools.read_workspace_file("/etc/passwd")
        self.assertEqual(res_abs["status"], "error")
        self.assertIn("Access denied", res_abs["output"])

        res_rel = agent_tools.write_workspace_file("../../outside.txt", "blocked")
        self.assertEqual(res_rel["status"], "error")
        self.assertIn("Access denied", res_rel["output"])

    # =================================================================
    # 2. FILE WRITE, LINE-RANGED READ, PATCH & SURGICAL LINE EDIT
    # =================================================================
    def test_write_read_patch_and_edit_lines(self):
        initial = "def calc(a, b):\n    return a - b\n\ndef scale(x):\n    return x * 2\n"
        w_res = agent_tools.write_workspace_file("calc.py", initial)
        self.assertEqual(w_res["status"], "success")

        # Line-ranged read
        r_res = agent_tools.read_workspace_file("calc.py", line_start=1, line_count=2)
        self.assertEqual(r_res["status"], "success")
        self.assertIn("return a - b", r_res["output"])
        self.assertNotIn("def scale", r_res["output"])

        # Surgical patch
        p_res = agent_tools.patch_workspace_file(
            "calc.py",
            search_pattern="def calc(a, b):\n    return a - b",
            replacement="def calc(a, b):\n    return a + b"
        )
        self.assertEqual(p_res["status"], "success")
        self.assertIn("return a + b", agent_tools.read_workspace_file("calc.py")["output"])

        # Line-range replacement
        if hasattr(agent_tools, "edit_workspace_lines"):
            e_res = agent_tools.edit_workspace_lines(
                "calc.py",
                start_line=4,
                end_line=5,
                new_content="def scale(x):\n    return int(x) * 10"
            )
            self.assertEqual(e_res["status"], "success")
            self.assertIn("int(x) * 10", agent_tools.read_workspace_file("calc.py")["output"])

    # =================================================================
    # 3. PYTHON AST SYNTAX CHECKS & PROJECT SCAN
    # =================================================================
    def test_python_syntax_and_project_scan(self):
        Path(self.temp_dir, "good.py").write_text("def greet(name):\n    return name\n", encoding="utf-8")
        Path(self.temp_dir, "bad.py").write_text("def broken(:\n    pass\n", encoding="utf-8")

        checker = getattr(agent_tools, "check_python_syntax", None) or getattr(agent_tools, "check_python_file", None)
        self.assertIsNotNone(checker, "Expected check_python_syntax or check_python_file in agent_tools")

        self.assertEqual(checker("good.py")["status"], "success")
        bad_res = checker("bad.py")
        self.assertEqual(bad_res["status"], "error")
        self.assertIn("SyntaxError", bad_res["output"])

        if hasattr(agent_tools, "run_project_checks"):
            self.assertEqual(agent_tools.run_project_checks(".")["status"], "error")

    # =================================================================
    # 4. CODE SEARCH (GREP / LITERAL) & GLOB FILE DISCOVERY
    # =================================================================
    def test_code_search_and_file_discovery(self):
        sub = Path(self.temp_dir, "pkg")
        sub.mkdir(parents=True, exist_ok=True)
        Path(sub, "module.py").write_text("API_VERSION = 'v1'\n\ndef call_api():\n    pass\n", encoding="utf-8")

        if hasattr(agent_tools, "grep_workspace"):
            g_res = agent_tools.grep_workspace("API_VERSION", path=".")
            self.assertEqual(g_res["status"], "success")
            self.assertIn("module.py:1", g_res["output"])

        if hasattr(agent_tools, "search_workspace_code"):
            s_res = agent_tools.search_workspace_code("API_VERSION")
            self.assertEqual(s_res["status"], "success")
            self.assertEqual(s_res["count"], 1)
            self.assertIn("module.py:1", s_res["output"])

        if hasattr(agent_tools, "find_workspace_files"):
            f_res = agent_tools.find_workspace_files("*.py")
            self.assertEqual(f_res["status"], "success")
            self.assertIn("pkg/module.py", f_res["output"])

    # =================================================================
    # 5. MULTI-FORMAT TOOL CALL PARSER & <think> STRIPPING
    # =================================================================
    def test_extract_all_tool_calls_formats(self):
        if not hasattr(agent_tools, "extract_all_tool_calls"):
            self.skipTest("extract_all_tool_calls not available in this agent_tools build")

        # 1) Qwen3-Coder Native XML (<function=...><parameter=...>) + <think> guard
        xml_reply = """<think>
Drafting thought: do not execute <function=execute_terminal><parameter=command>rm -rf /</parameter></function>
</think>
Reading target file:
<tool_call>
<function=read_workspace_file>
<parameter=path>
pkg/module.py
</parameter>
<parameter=line_start>
1
</parameter>
<parameter=line_count>
25
</parameter>
</function>
</tool_call>"""
        xml_calls = agent_tools.extract_all_tool_calls(xml_reply)
        self.assertEqual(len(xml_calls), 1)
        self.assertEqual(xml_calls[0][0], "read_workspace_file")
        self.assertEqual(xml_calls[0][1]["path"], "pkg/module.py")
        self.assertEqual(xml_calls[0][1]["line_count"], 25)

        # 2) Qwen JSON inside <tool_call>
        json_reply = '<tool_call>{"name": "grep_workspace", "arguments": {"pattern": "API_VERSION", "path": "."}}</tool_call>'
        json_calls = agent_tools.extract_all_tool_calls(json_reply)
        self.assertEqual(len(json_calls), 1)
        self.assertEqual(json_calls[0][0], "grep_workspace")
        self.assertEqual(json_calls[0][1]["pattern"], "API_VERSION")

        # 3) OpenAI streamed delta.tool_calls
        streamed_tcs = [{
            "id": "call_1",
            "function": {
                "name": "write_workspace_file",
                "arguments": json.dumps({"path": "notes.txt", "content": "ok"})
            }
        }]
        oai_calls = agent_tools.extract_all_tool_calls("", streamed_tool_calls=streamed_tcs)
        self.assertEqual(len(oai_calls), 1)
        self.assertEqual(oai_calls[0][0], "write_workspace_file")
        self.assertEqual(oai_calls[0][1]["path"], "notes.txt")

        # 4) Exact screenshot format: XML inside a tool_call wrapper, numeric coercion.
        screenshot_reply = """<tool_call>
<function=list_workspace_files>
<parameter=relative_path>
.
</parameter>
<parameter=max_depth>
3
</parameter>
</function>
</tool_call>"""
        screenshot_calls = agent_tools.extract_all_tool_calls(screenshot_reply)
        self.assertEqual(len(screenshot_calls), 1)
        self.assertEqual(screenshot_calls[0][0], "list_workspace_files")
        self.assertEqual(screenshot_calls[0][1], {"relative_path": ".", "max_depth": 3})

        # Duplicate textual + native calls must execute only once.
        duplicate_calls = agent_tools.extract_all_tool_calls(
            screenshot_reply,
            [{"function": {"name": "list_workspace_files", "arguments": '{"relative_path":".","max_depth":3}'}}],
        )
        self.assertEqual(len(duplicate_calls), 1)

        # 5) Markdown Fenced Action Blocks
        md_reply = (
            "```read_file:pkg/module.py:1:10```\n"
            "```patch:pkg/module.py\n<<<<<<< SEARCH\nAPI_VERSION = 'v1'\n=======\nAPI_VERSION = 'v2'\n>>>>>>>```\n"
            "```bash\npython3 -V\n```"
        )
        md_calls = agent_tools.extract_all_tool_calls(md_reply)
        self.assertEqual(len(md_calls), 3)
        self.assertEqual(md_calls[0][0], "read_workspace_file")
        self.assertEqual(md_calls[1][0], "patch_workspace_file")
        self.assertEqual(md_calls[2][0], "execute_terminal")

    # =================================================================
    # 6. UNIFIED SKILL DISPATCHER & ALIASES (INCLUDING 'ggrep')
    # =================================================================
    def test_dispatch_tool_call_and_aliases(self):
        if not hasattr(agent_tools, "dispatch_tool_call"):
            self.skipTest("dispatch_tool_call not available in this agent_tools build")

        ui_name, target, w_res = agent_tools.dispatch_tool_call(
            "create_file",
            {"path": "alias_test.py", "content": "MAGIC_CONST = 999\n"}
        )
        self.assertEqual(ui_name, "write_file")
        self.assertEqual(w_res["status"], "success")

        # Test Qwen3-Coder 30B 'ggrep' quirk alias
        ui_name, target, g_res = agent_tools.dispatch_tool_call(
            "ggrep",
            {"pattern": "MAGIC_CONST", "path": "."}
        )
        self.assertEqual(ui_name, "grep_search")
        self.assertEqual(g_res["status"], "success")
        self.assertIn("alias_test.py:1", g_res["output"])

        # Test syntax check dispatch
        ui_name, target, c_res = agent_tools.dispatch_tool_call(
            "check_syntax",
            {"path": "alias_test.py"}
        )
        self.assertEqual(ui_name, "check_syntax")
        self.assertEqual(c_res["status"], "success")

        # On-demand Studio skill reader is allow-listed and reads bundled docs.
        ui_name, target, skill_res = agent_tools.dispatch_tool_call(
            "read_skill", {"name": "tools_skills.md"}
        )
        self.assertEqual(ui_name, "read_skill")
        self.assertEqual(skill_res["status"], "success")
        self.assertIn("tools_skills.md", skill_res["skill"])
        self.assertTrue(skill_res["output"].strip())
        _, _, blocked_skill = agent_tools.dispatch_tool_call(
            "read_skill", {"name": "../../app.py"}
        )
        self.assertEqual(blocked_skill["status"], "error")

    # =================================================================
    # 7. COLIBRI /v1/chat/completions MESSAGE FORMAT SANITIZER
    # =================================================================
    def test_colibri_chat_format_and_role_alternation(self):
        try:
            import app as studio_app
        except ImportError:
            self.skipTest("Flask/requests or app.py not available in current test runner")

        if not hasattr(studio_app, "build_colibri_messages"):
            self.skipTest("build_colibri_messages not present in app.py")

        sid = "unit_test_session"
        with sqlite3.connect(studio_app.DB_PATH) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
            # Simulate two consecutive user messages (e.g. after user clicked Stop) + assistant with <think>
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, 'user', ?, 1.0)",
                (sid, "Aborted prompt 1")
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, 'user', ?, 2.0)",
                (sid, "Retry prompt 2")
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, 'assistant', ?, 3.0)",
                (sid, "<think>hidden chain of thought</think>Final verified response.")
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, 'user', ?, 4.0)",
                (sid, "Follow-up prompt 3")
            )
            conn.commit()

        try:
            messages = studio_app.build_colibri_messages(sid, "Follow-up prompt 3")
            self.assertEqual(messages[0]["role"], "system")
            roles = [m["role"] for m in messages[1:]]
            self.assertEqual(roles, ["user", "assistant", "user"])
            self.assertIn("Aborted prompt 1", messages[1]["content"])
            self.assertIn("Retry prompt 2", messages[1]["content"])
            self.assertNotIn("<think>", messages[2]["content"])
            self.assertEqual(messages[2]["content"], "Final verified response.")
        finally:
            with sqlite3.connect(studio_app.DB_PATH) as conn:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                conn.commit()

    # =================================================================
    # 8. OPTIONAL CODING TOOLKIT TESTS (API PARITY, SKILLS & MEMORY)
    # =================================================================
    def test_dependency_check_and_pip_install_guard(self):
        req = Path(self.temp_dir, "requirements.txt")
        req.write_text("\n# comment\nrequests>=2.0\nPillow\n", encoding="utf-8")
        result = agent_tools.check_python_dependencies(".")
        self.assertIn(result["status"], ("success", "error"))
        self.assertEqual(result["checked_count"], 2)
        self.assertIn("No packages were installed", result["output"])

        bad = agent_tools.install_python_package("--index-url=https://example.invalid")
        self.assertEqual(bad["status"], "error")
        self.assertIn("Rejected package spec", bad["output"])
        bad_url = agent_tools.install_python_package("https://example.invalid/pkg.whl")
        self.assertEqual(bad_url["status"], "error")

        _, _, checked = agent_tools.dispatch_tool_call("check_dependencies", {"path": "."})
        self.assertIn(checked["status"], ("success", "error"))

    def test_optional_api_parity_and_memory_tools(self):
        if hasattr(agent_tools, "compare_python_api"):
            source = "def public_api(value):\n    return value\n\nclass PublicThing:\n    pass\n"
            Path(self.temp_dir, "old.py").write_text(source, encoding="utf-8")
            Path(self.temp_dir, "same.py").write_text(source, encoding="utf-8")
            Path(self.temp_dir, "changed.py").write_text("def public_api(value, extra=None):\n    return value\n", encoding="utf-8")
            self.assertEqual(agent_tools.compare_python_api("old.py", "same.py")["status"], "success")
            self.assertEqual(agent_tools.compare_python_api("old.py", "changed.py")["status"], "error")

        if hasattr(agent_tools, "write_project_memory") and hasattr(agent_tools, "read_project_memory"):
            saved = agent_tools.write_project_memory("api_contract", "Keep public_api(value) stable.")
            self.assertEqual(saved["status"], "success")
            loaded = agent_tools.read_project_memory("api_contract")
            self.assertEqual(loaded["status"], "success")
            self.assertIn("Keep public_api(value) stable", loaded["output"])

        if hasattr(agent_tools, "list_coding_skills") and hasattr(agent_tools, "load_coding_skill"):
            custom = Path(self.temp_dir, ".agent_skills", "review")
            custom.mkdir(parents=True, exist_ok=True)
            (custom / "SKILL.md").write_text("Review changed lines and require regression tests.", encoding="utf-8")
            self.assertIn("review", agent_tools.list_coding_skills()["skills"])
            self.assertIn("regression tests", agent_tools.load_coding_skill("review")["output"])



    def test_colibri_only_routing_and_speed_metric(self):
        from pathlib import Path
        app_source = Path(__file__).with_name("app.py").read_text(encoding="utf-8")
        self.assertNotIn("OLLAMA_ROUTER_MODEL", app_source)
        self.assertNotIn("ENABLE_OLLAMA_REFLEX", app_source)
        self.assertIn("tok/s", app_source)
        self.assertIn("TARGET_COLIBRI", app_source)

if __name__ == "__main__":
    unittest.main(verbosity=2)