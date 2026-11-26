"""Tests for the agent-loop question pause, auto-yes, and message-done features."""
import unittest

import app


class QuestionDetectionTests(unittest.TestCase):
    def test_detects_yes_no_question(self):
        text = "I found the bug. Should I apply the fix now? Reply yes or no."
        self.assertTrue(app.looks_like_user_question(text))

    def test_detects_confirm_phrase(self):
        self.assertTrue(app.looks_like_user_question("Do you want me to continue?"))

    def test_does_not_flag_statement(self):
        self.assertFalse(app.looks_like_user_question("I applied the fix and ran the tests."))

    def test_does_not_flag_tool_call(self):
        text = "Let me read it. " + '<tool_call>' + '{"name":"read_workspace_file"}' + '</tool_call>'
        self.assertFalse(app.looks_like_user_question(text))

    def test_does_not_flag_status_tag(self):
        self.assertFalse(app.looks_like_user_question("<task_status>continue</task_status>"))

    def test_does_not_flag_long_text(self):
        long_text = "x " * 200 + "Should I proceed?"
        self.assertFalse(app.looks_like_user_question(long_text))


class MessageDoneTests(unittest.TestCase):
    def test_detects_message_done_tag(self):
        self.assertTrue(app.looks_like_message_done("<task_status>done</task_status>"))

    def test_detects_message_done_phrase(self):
        self.assertTrue(app.looks_like_message_done("All steps verified. Message done."))

    def test_does_not_flag_normal_text(self):
        self.assertFalse(app.looks_like_message_done("I still need to run the tests."))


class AutoYesTests(unittest.TestCase):
    def test_auto_yes_replies_yes(self):
        self.assertEqual(app.auto_yes_reply("Should I apply the fix?"), "yes")

    def test_auto_yes_replies_continue_for_open_question(self):
        self.assertEqual(app.auto_yes_reply("What should I name the file?"), "continue")

    def test_auto_yes_replies_continue_for_empty(self):
        self.assertEqual(app.auto_yes_reply(""), "continue")


class QuestionModeNormalisationTests(unittest.TestCase):
    def test_normalises_valid_values(self):
        self.assertEqual(app.normalise_question_mode("AUTO"), "auto")
        self.assertEqual(app.normalise_question_mode("manual"), "manual")

    def test_defaults_to_manual(self):
        self.assertEqual(app.normalise_question_mode("nonsense"), "manual")
        self.assertEqual(app.normalise_question_mode(None), "manual")


if __name__ == "__main__":
    unittest.main()
