import ast
import pathlib
import unittest

APP_SOURCE = pathlib.Path(__file__).with_name("app.py").read_text(encoding="utf-8")
TREE = ast.parse(APP_SOURCE)
HELPER_NODE = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "build_shared_agent_loop_instructions")
MODULE = ast.Module(body=[HELPER_NODE], type_ignores=[])
namespace = {}
exec(compile(MODULE, "app.py", "exec"), namespace)
build_shared_agent_loop_instructions = namespace["build_shared_agent_loop_instructions"]

class SharedAgentLoopTests(unittest.TestCase):
    def test_shared_loop_policy_is_enabled_for_every_loader(self):
        for loader, model in (("Colibri", "native-model"), ("Ollama", "qwen3.5:4b"), ("Web Extension", "gemini")):
            with self.subTest(loader=loader):
                policy = build_shared_agent_loop_instructions(loader, model)
                self.assertIn("read_studio_skill", policy)
                self.assertIn("initialization", policy)
                self.assertIn("plain-text file", policy)
                self.assertIn("continues until verified completion", policy)
                self.assertIn("exact repeated-response detection", policy)
                self.assertIn(model, policy)

    def test_execution_loop_has_no_fixed_turn_cap_and_exact_duplicate_guard(self):
        self.assertNotIn("MAX_AGENT_STEPS", APP_SOURCE)
        self.assertIn("while True:", APP_SOURCE)
        self.assertIn("is_exact_duplicate_completed_response(last_completed_response, clean_reply)", APP_SOURCE)
        self.assertIn("Loop stopped: identical response detected.", APP_SOURCE)
        self.assertIn("ACTIVE_RUNS", APP_SOURCE)

    def test_streaming_iteration_does_not_clear_accumulated_output(self):
        self.assertIn("data.type === 'iteration_start'", APP_SOURCE)
        self.assertIn("contentElem.dataset.rawContent = (contentElem.dataset.rawContent || '') + data.content;", APP_SOURCE)
        self.assertIn("// Preserve the accumulated assistant output across tool cycles.", APP_SOURCE)
        self.assertNotIn("contentElem.dataset.rawContent = preservedThinking", APP_SOURCE)

    def test_loader_specific_tool_call_formats(self):
        for loader in ("Ollama", "Colibri", "Web Extension"):
            with self.subTest(loader=loader):
                policy = build_shared_agent_loop_instructions(loader, "m")
                self.assertIn("read_studio_skill", policy)
                self.assertLess(len(policy), 1000)
        self.assertIn("<function=read_studio_skill>", build_shared_agent_loop_instructions("Colibri", "m"))
        self.assertIn('"arguments"', build_shared_agent_loop_instructions("Ollama", "m"))

    def test_initialization_is_queued_as_a_text_file_not_a_large_prompt(self):
        self.assertIn('"kind": "initialization_file"', APP_SOURCE)
        self.assertIn('"status":"file"', APP_SOURCE)
        self.assertIn("colibri_initialization.txt", APP_SOURCE)
        self.assertIn("Please read the attached colibri_initialization.txt file", APP_SOURCE)
        self.assertIn("task.status === \"file\"", (pathlib.Path(__file__).parent / "web_extension" / "content.js").read_text(encoding="utf-8"))

    def test_blocked_status_is_recognized_by_stream_parser(self):
        self.assertIn(r'<task_status>\s*(complete|continue|blocked)\s*</task_status>', APP_SOURCE)
        self.assertIn('if task_status == "blocked":', APP_SOURCE)

    def test_duplicate_comparison_is_exact_and_ignores_empty(self):
        fn = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "is_exact_duplicate_completed_response")
        ns = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "app.py", "exec"), ns)
        compare = ns["is_exact_duplicate_completed_response"]
        self.assertTrue(compare("same", "same"))
        self.assertFalse(compare("similar", "similar but different"))
        self.assertFalse(compare("same", ""))
        self.assertFalse(compare(None, "same"))

    def test_strata_usage_panel_is_bottom_left_and_receives_metrics(self):
        self.assertIn('id="strata-usage-status"', APP_SOURCE)
        self.assertIn("Prefill", APP_SOURCE)
        self.assertIn("Generation", APP_SOURCE)
        self.assertIn("Max context", APP_SOURCE)
        self.assertIn('payload["stream_options"] = {"include_usage": True}', APP_SOURCE)

if __name__ == "__main__":
    unittest.main()
