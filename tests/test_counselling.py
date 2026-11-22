from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from uuid import uuid4

from fastapi import HTTPException

from backend import main
from backend.catalog import evaluate_requirement, get_catalog
from backend.dialogue import add_turn, comparison_answer, new_session
from backend.nlp import get_extractor
from backend.ollama import fallback_question, make_question
from backend.store import Conflict, Sessions


def render(intent, language, latest, observations):
    return fallback_question(intent, language, latest, observations), "test"


class CounsellingTests(TestCase):
    def test_spans_and_attribution_in_hinglish(self):
        text = "Papa chahte hain main engineering karun, but mujhe data analyse karna pasand hai."
        family, student = get_extractor().extract(text)
        self.assertEqual((family.kind, family.tags), ("external_preference", ()))
        self.assertEqual((student.kind, student.tags), ("interest", ("I",)))
        self.assertEqual(student.text, text[student.start:student.end])

    def test_lack_of_exposure_and_dislike_are_distinct(self):
        observations = get_extractor().extract("I have never tried coding, but I don't like writing poems.")
        self.assertEqual((observations[0].kind, observations[0].polarity), ("limited_exposure", 0))
        self.assertEqual((observations[1].kind, observations[1].polarity), ("interest", -1))

    def test_vague_turn_has_no_recommendation(self):
        state = add_turn(new_session(10, "english", []), "Science is nice.", False, render)
        self.assertEqual(state["report"]["status"], "needs_evidence")
        self.assertEqual(state["last_intent"]["kind"], "clarify")

    def test_contextual_turns_and_reports_follow_stage(self):
        investigations = "I like analysing survey data and finding patterns."
        building = "I like fixing broken equipment and building things by hand."
        left = add_turn(new_session(10, "english", []), investigations, False, render)
        right = add_turn(new_session(10, "english", []), building, False, render)
        self.assertNotEqual(left["messages"][-1]["text"], right["messages"][-1]["text"])
        for stage in (10, 12):
            state = add_turn(new_session(stage, "english", []), investigations, False, render)
            self.assertTrue(state["report"]["options"])
            for option in state["report"]["options"]:
                self.assertEqual(get_catalog().by_id[option["id"]].stage, stage)
                self.assertTrue(option["evidence"])

    def test_skip_and_short_comparison(self):
        state = add_turn(new_session(12, "english", []), "I enjoy writing short stories.", False, render)
        intent = state["last_intent"]
        self.assertEqual(comparison_answer("the second one", intent)[0].tags, (intent["codes"][1],))
        before = len(state["ledger"])
        add_turn(state, "", True, render)
        self.assertEqual(len(state["ledger"]), before)

    def test_offline_fallback_and_comparison_order(self):
        intent = {"kind": "compare", "activities": ["building a bridge", "analysing data"]}
        with patch("backend.ollama.local_chat", return_value={"question": "Would you prefer analysing data or building a bridge?"}):
            question, engine = make_question(intent, "english", "", [])
        self.assertEqual(engine, "local-fallback")
        self.assertLess(question.index("building a bridge"), question.index("analysing data"))

    def test_hinglish_uses_local_wording_when_model_replies_in_english(self):
        intent = {"kind": "compare", "activities": ["building a bridge", "analysing data"]}
        with patch("backend.ollama.local_chat", return_value={"question": "Do you prefer building a bridge or analysing data?"}):
            question, engine = make_question(intent, "hinglish", "", [])
        self.assertEqual(engine, "local-fallback")
        self.assertIn("Aap", question)

    def test_three_valued_and_scoped_requirements(self):
        combined = {"all_of": [{"fact": "a"}, {"any_of": [{"fact": "b"}, {"fact": "c"}]}]}
        self.assertIsNone(evaluate_requirement(combined, {"a": True, "b": False}))
        self.assertFalse(evaluate_requirement(combined, {"a": False}))
        self.assertTrue(evaluate_requirement(combined, {"a": True, "c": True}))
        catalog = get_catalog()
        economics = catalog.by_id["economics"]
        rule = "du-economics-2026"
        self.assertEqual(catalog.checks(economics, {rule: {"cuet_mathematics_2026": False}})[0]["status"], "verification_needed")
        self.assertEqual(catalog.checks(economics, {rule: {"class_xii_mathematics": True, "cuet_mathematics_2026": False}})[0]["status"], "recorded_condition_unmet")
        self.assertEqual(catalog.checks(economics, {rule: {"class_xii_mathematics": True, "cuet_mathematics_2026": True}})[0]["status"], "recorded_conditions_met")

    def test_session_retry_correction_and_deletion(self):
        with TemporaryDirectory() as directory:
            store = Sessions(Path(directory) / "test.sqlite3")
            with patch.object(main, "sessions", return_value=store), patch.object(main, "make_question", side_effect=render):
                state = main.start_session(main.SessionInput(stage=10))
                data = main.TurnInput(request_id=uuid4(), revision=0, text="I enjoy writing short stories.")
                first = main.turn(state["id"], data)
                called_twice = main.turn(state["id"], data)
                self.assertEqual(first, called_twice)
                with self.assertRaises(HTTPException) as stale:
                    main.turn(state["id"], data.model_copy(update={"request_id": uuid4()}))
                self.assertEqual(stale.exception.status_code, 409)
                entry = first["ledger"][0]
                corrected = main.retract_evidence(state["id"], entry["id"], main.RevisionInput(revision=1))
                self.assertEqual(corrected["report"]["status"], "needs_evidence")
                with self.assertRaises(Conflict):
                    store.update(first, 1)
                main.delete(state["id"])
                self.assertIsNone(store.get(state["id"]))

    def test_distress_pauses_normal_response_without_retaining_text(self):
        with TemporaryDirectory() as directory:
            store = Sessions(Path(directory) / "test.sqlite3")
            with patch.object(main, "sessions", return_value=store), patch.object(main, "make_question") as generate:
                state = main.start_session(main.SessionInput(stage=12))
                reply = "I want to hurt myself."
                result = main.turn(state["id"], main.TurnInput(request_id=uuid4(), revision=0, text=reply))
                self.assertNotIn(reply, str(result))
                self.assertIn("trusted adult", result["messages"][-1]["text"])
                generate.assert_not_called()
