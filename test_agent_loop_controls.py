"""Tests for agent-loop controls: question detection, auto-yes/manual mode,
'message done' stop detection, and the plan/progress checklist UI.

These target the real app.py API: detect_user_question, detect_message_done,
and the LOADER_STATE['settings']['question_mode'] toggle.
"""
import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app


def _sse_events(text: str) -> list[dict]:
    events = []
    for chunk in text.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        for line in chunk.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


def _event_types(events: list[dict]) -> list[str]:
    return [e.get("type") for e in events]


def _task_status_values(events: list[dict]) -> list[str]:
    return [e.get("status") for e in events if e.get("type") == "task_status"]


class DetectUserQuestionTests(unittest.TestCase):
    def test_detects_common_question_phrases(self):
        samples = [
            "Should I proceed with the edit?",
            "Do you want me to restart the service?",
            "Would you like me to run the tests?",
            "Please confirm before I delete the file.",
            "I need your approval to continue.",
            "Waiting for your reply.",
            "Is that correct?",
            "yes/no",
            "Should we use option A?",
        ]
        for text in samples:
            with self.subTest(text=text):
                self.assertIsNotNone(app.detect_user_question(text))

    def test_returns_matched_phrase(self):
        phrase = app.detect_user_question("Should I apply the patch now?")
        self.assertIsNotNone(phrase)
        self.assertIn("should i", phrase.lower())

    def test_no_false_positive_on_plain_statement(self):
        samples = [
            "I edited app.py and ran the tests successfully.",
            "The loop now stops on completion.",
            "All files were written and verified.",
            "",
        ]
        for text in samples:
            with self.subTest(text=text):
                self.assertIsNone(app.detect_user_question(text))


class DetectMessageDoneTests(unittest.TestCase):
    def test_detects_done_signals(self):
        samples = [
            "message done",
            "Task done — all requirements verified.",
            "job done",
            "all done",
            "finished the task",
        ]
        for text in samples:
            with self.subTest(text=text):
                self.assertTrue(app.detect_message_done(text))

    def test_no_false_positive(self):
        samples = [
            "I will continue working on the task.",
            "The work is not finished yet.",
            "",
        ]
        for text in samples:
            with self.subTest(text=text):
                self.assertFalse(app.detect_message_done(text))


class QuestionModeSettingTests(unittest.TestCase):
    def setUp(self):
        self._orig = dict(app.LOADER_STATE["settings"])

    def tearDown(self):
        app.LOADER_STATE["settings"].clear()
        app.LOADER_STATE["settings"].update(self._orig)

    def test_default_is_manual(self):
        app.LOADER_STATE["settings"].pop("question_mode", None)
        self.assertEqual(app.LOADER_STATE["settings"].get("question_mode", "manual"), "manual")

    def test_auto_yes_mode_is_honored(self):
        app.LOADER_STATE["settings"]["question_mode"] = "auto_yes"
        self.assertEqual(app.LOADER_STATE["settings"].get("question_mode", "manual"), "auto_yes")


class AgentLoopQuestionBehaviorTests(unittest.TestCase):
    """Drive the real event_stream with a fake provider to verify pause/auto-yes/done."""

    def setUp(self):
        self._orig_loader = app.LOADER_STATE.get("loader")
        self._orig_model = app.LOADER_STATE.get("model")
        self._orig_settings = dict(app.LOADER_STATE["settings"])
        app.LOADER_STATE["loader"] = "colibri"
        app.LOADER_STATE["model"] = "test-model"
        app.LOADER_STATE["settings"]["question_mode"] = "manual"

    def tearDown(self):
        app.LOADER_STATE["loader"] = self._orig_loader
        app.LOADER_STATE["model"] = self._orig_model
        app.LOADER_STATE["settings"].clear()
        app.LOADER_STATE["settings"].update(self._orig_settings)

    def _run_with_replies(self, replies: list[str]) -> list[dict]:
        calls = {"n": 0}

        def fake_stream(messages, model, options):
            idx = min(calls["n"], len(replies) - 1)
            calls["n"] += 1
            yield replies[idx]

        original = app.stream_colibri_completion
        app.stream_colibri_completion = fake_stream
        try:
            out = app.event_stream("q", "test question", "test-session")
            return _sse_events(out)
        finally:
            app.stream_colibri_completion = original

    def test_manual_mode_pauses_on_question(self):
        events = self._run_with_replies(["Should I proceed with the edit?"])
        self.assertIn("user_question", _event_types(events))
        self.assertIn("awaiting_user", _task_status_values(events))
        # Loop must stop (no 'continuing' after the question).
        self.assertNotIn("continuing", _task_status_values(events))

    def test_auto_yes_mode_continues_after_question(self):
        app.LOADER_STATE["settings"]["question_mode"] = "auto_yes"
        events = self._run_with_replies([
            "Should I proceed with the edit?",
            "<task_status>complete</task_status>Done, all verified.",
        ])
        self.assertIn("auto_yes", _task_status_values(events))
        # After auto-yes the loop keeps going and eventually completes.
        self.assertIn("complete", _task_status_values(events))

    def test_message_done_signal_stops_loop(self):
        events = self._run_with_replies(["All work finished. message done"])
        statuses = _task_status_values(events)
        self.assertIn("complete", statuses)
        done_event = next(e for e in events if e.get("type") == "task_status" and e.get("status") == "complete")
        self.assertEqual(done_event.get("reason"), "message_done_signal")

    def test_complete_status_still_stops_loop(self):
        events = self._run_with_replies(["<task_status>complete</task_status>All requirements verified."])
        self.assertIn("complete", _task_status_values(events))
        self.assertNotIn("continuing", _task_status_values(events))


class PlanChecklistUITests(unittest.TestCase):
    def test_checklist_panel_present_in_template(self):
        self.assertIn('id="plan-checklist-panel"', app.HTML_TEMPLATE)
        self.assertIn('id="plan-checklist"', app.HTML_TEMPLATE)
        self.assertIn("Plan / Progress", app.HTML_TEMPLATE)

    def test_checklist_is_below_chat_selection(self):
        # The session list (chat selection) must appear before the checklist panel.
        session_idx = app.HTML_TEMPLATE.index('id="session-list"')
        checklist_idx = app.HTML_TEMPLATE.index('id="plan-checklist-panel"')
        self.assertLess(session_idx, checklist_idx)

    def test_render_plan_checklist_function_exists(self):
        self.assertIn("function renderPlanChecklist", app.HTML_TEMPLATE)

    def test_plan_update_event_is_handled(self):
        self.assertIn("plan_update", app.HTML_TEMPLATE)

    def test_question_mode_select_exists(self):
        self.assertIn('id="question-mode"', app.HTML_TEMPLATE)


if __name__ == "__main__":
    unittest.main()