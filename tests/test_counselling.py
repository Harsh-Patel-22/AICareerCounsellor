import json
from io import BytesIO
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
from backend.ollama import QUESTION_SCHEMA, fallback_question, local_chat, make_question
from backend.store import Conflict, Sessions


def render(intent, language, latest, observations):
    return fallback_question(intent, language, latest, observations), "test"


SHOP_REPLY = "I tracked which items sold more after a discount at my uncle's shop."
FAIR_REPLY = "I am not sure. I used to love the science fairs that happened in my school"


class CounsellingTests(TestCase):
    def test_spans_and_attribution_in_hinglish(self):
        text = "Papa chahte hain main engineering karun, but mujhe data analyse karna pasand hai."
        family, student = get_extractor().extract(text)
        self.assertEqual((family.kind, family.tags), ("external_preference", ()))
        self.assertEqual((student.kind, student.tags), ("interest", ("I",)))
        self.assertEqual(student.text, text[student.start:student.end])

    def test_observed_work_is_not_the_students_demonstrated_action(self):
        extractor = get_extractor()
        observed = extractor.extract("One team built a collapsible moon prop from cardboard.")
        performed = extractor.extract("I built a cardboard bridge.")
        self.assertFalse(any(item.kind == "demonstrated_action" for item in observed))
        self.assertTrue(any(item.kind == "demonstrated_action" for item in performed))

    def test_known_activity_survives_an_unfamiliar_joined_activity(self):
        reply = "I enjoy building circuits and testing how sensors work."
        entry = get_extractor().extract(reply)[0]
        self.assertEqual((entry.kind, entry.tags), ("interest", ("R",)))
        self.assertEqual(reply[entry.start:entry.end], "I enjoy building circuits")

    def test_lack_of_exposure_and_dislike_are_distinct(self):
        observations = get_extractor().extract("I have never tried coding, but I don't like writing poems.")
        self.assertEqual((observations[0].kind, observations[0].polarity), ("limited_exposure", 0))
        self.assertEqual((observations[1].kind, observations[1].polarity), ("interest", -1))

    def test_vague_turn_has_no_recommendation(self):
        state = add_turn(new_session(10, "english", []), "Science is nice.", False, render)
        self.assertEqual(state["report"]["status"], "needs_evidence")
        self.assertEqual(state["last_intent"]["kind"], "clarify")

    def test_science_fair_thread_grounds_first_question_and_repairs_corrections(self):
        state = new_session(10, "english", [])
        hallucination = {"question": "Which activity do you prefer between science fairs and other activities you mentioned?"}
        with patch("backend.ollama.local_chat", return_value=hallucination) as model:
            add_turn(state, FAIR_REPLY, False, make_question)
            self.assertEqual(state["last_intent"]["kind"], "probe_topic")
            self.assertEqual(state["messages"][-1]["text"], "What did you enjoy about the science fairs?")
            self.assertEqual(state["messages"][-1]["engine"], "local-fallback")
            for correction, expected in (
                ("What other activities did I mention?", "You mentioned the science fairs. What stood out to you?"),
                ("I mentioned no other activity", "You're right. We can focus on the science fairs. What do you remember enjoying?"),
            ):
                add_turn(state, correction, False, make_question)
                self.assertEqual(state["last_intent"]["kind"], "repair_reference")
                self.assertEqual(state["messages"][-1]["text"], expected)
                self.assertEqual(state["messages"][-1]["engine"], "local-repair")
            model.assert_called_once()

    def test_vague_clarification_cannot_invent_a_comparison(self):
        with patch("backend.ollama.local_chat", return_value={"question": "Which activity do you prefer between the ones you mentioned?"}):
            state = add_turn(new_session(10, "english", []), "Science is nice.", False, make_question)
        self.assertEqual(state["last_intent"]["kind"], "clarify")
        self.assertEqual(state["messages"][-1]["engine"], "local-fallback")
        self.assertIn("Science is nice", state["messages"][-1]["text"])

    def test_topic_followup_can_use_grounded_ollama_output(self):
        answer = "What did you find most interesting about the science fairs?"
        with patch("backend.ollama.local_chat", return_value={"question": answer}):
            state = add_turn(new_session(10, "english", []), FAIR_REPLY, False, make_question)
        self.assertEqual(state["messages"][-1]["engine"], "ollama")
        self.assertEqual(state["messages"][-1]["text"], answer)

    def test_topic_followup_cannot_assume_the_student_participated(self):
        answer = "What did you enjoy most about participating in the science fairs?"
        with patch("backend.ollama.local_chat", return_value={"question": answer}):
            state = add_turn(new_session(10, "english", []), FAIR_REPLY, False, make_question)
        self.assertEqual(state["messages"][-1]["engine"], "local-fallback")
        self.assertEqual(state["messages"][-1]["text"], "What did you enjoy about the science fairs?")

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

    def test_shop_activity_gets_an_interest_probe_and_grounds_short_reply(self):
        state = add_turn(new_session(10, "english", []), SHOP_REPLY, False, render)
        self.assertEqual(state["last_intent"]["kind"], "probe_interest")
        self.assertIn("discount at your uncle's shop", state["messages"][-1]["text"])
        self.assertFalse(any(item["kind"] == "interest" for item in state["ledger"]))
        add_turn(state, "Finding patterns in which items sold.", False, render)
        self.assertTrue(any(item["kind"] == "interest" and "I" in item["tags"] for item in state["ledger"]))
        self.assertNotEqual(state["last_intent"]["kind"], "probe_interest")

    def test_shop_probe_rejects_generic_factual_or_repeated_output(self):
        state = add_turn(new_session(10, "english", []), SHOP_REPLY, False, render)
        intent = state["last_intent"]
        cases = [
            ("english", "What did you find interesting about this experience?"),
            ("english", "Which items sold more after the discount at your uncle's shop?"),
            ("hinglish", "Discount ke baad uncle ki shop mein kaunse items zyada bike?"),
            ("hinglish", "Bike ke items mein discount ke baad aapko kaunsa part interesting laga?"),
            ("hinglish", SHOP_REPLY),
        ]
        for language, answer in cases:
            with self.subTest(answer=answer), patch("backend.ollama.local_chat", return_value={"question": answer}):
                question, engine = make_question(intent, language, SHOP_REPLY, [])
                self.assertEqual(engine, "local-fallback")
                self.assertIn("discount", question)
                self.assertIn("interesting", question)
                self.assertEqual(question.count("?"), 1)

    def test_shop_probe_accepts_a_grounded_question_in_either_language(self):
        state = add_turn(new_session(10, "english", []), SHOP_REPLY, False, render)
        for language, answer in [
            ("english", "What part of tracking sales after the discount at your uncle's shop interested you?"),
            ("hinglish", "Uncle ki shop mein discount ke baad sales track karne mein aapko kya interesting laga?"),
        ]:
            with self.subTest(language=language), patch("backend.ollama.local_chat", return_value={"question": answer}):
                self.assertEqual(make_question(state["last_intent"], language, SHOP_REPLY, []), (answer, "ollama"))

    def test_shop_probe_uses_students_perspective(self):
        state = add_turn(new_session(10, "english", []), SHOP_REPLY, False, render)
        answer = "What part of tracking sales after the discount at my uncle's shop interested you?"
        with patch("backend.ollama.local_chat", return_value={"question": answer}):
            question, engine = make_question(state["last_intent"], "english", SHOP_REPLY, [])
        self.assertEqual(engine, "ollama")
        self.assertIn("your uncle's shop", question)

    def test_probe_does_not_treat_obligation_or_dislike_as_an_interest(self):
        for answer in ("Nothing.", "Checking stock was boring.", "Finding the totals, but I just had to."):
            state = add_turn(new_session(10, "english", []), SHOP_REPLY, False, render)
            add_turn(state, answer, False, render)
            self.assertFalse(any(item["kind"] == "interest" and item["polarity"] > 0 for item in state["ledger"]))

    def test_truncated_ollama_response_is_rejected_with_thinking_disabled(self):
        body = {"done_reason": "length", "message": {"content": "{\"question\":\"What did you enjoy?\"}"}}
        with patch("backend.ollama.urlopen", return_value=BytesIO(json.dumps(body).encode())) as request:
            with self.assertRaisesRegex(ValueError, "output limit"):
                local_chat("Ask one question.", {}, QUESTION_SCHEMA, max_tokens=72)
        data = json.loads(request.call_args.args[0].data)
        self.assertFalse(data["think"])
        self.assertEqual(data["options"]["num_predict"], 72)

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
