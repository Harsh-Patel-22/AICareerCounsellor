"""Exercise the locally served API with real HTTP and disposable authored sessions."""

import argparse
import json
import os
import re
import socket
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic, sleep
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
PATHWAYS = {row["id"]: row["stage"] for row in json.loads((ROOT / "catalog/pathways.json").read_text(encoding="utf-8"))["pathways"]}
FAIR = "I enjoyed science fairs in my school. They have so many innovative plus crazy experiments"
DETAIL = "It's just how interesting the experiments are. Drones flying around, rocket launchers, hydraulics and so on"
SHOP = "I tracked which items sold more after a discount at my uncle's shop."


class Client:
    def __init__(self, base):
        self.base = base
        self.requests = 0
        self.engines = Counter()
        self.latencies = []
        self.transcript = []

    def request(self, method, path, body=None, expected=(200,)):
        self.requests += 1
        request = Request(self.base + path, method=method, headers={"Content-Type": "application/json"},
                          data=json.dumps(body, ensure_ascii=False).encode() if body is not None else None)
        try:
            response = urlopen(request, timeout=85)
        except HTTPError as exc:
            response = exc
        with response:
            status = response.code
            raw = response.read()
        try:
            payload = json.loads(raw) if raw else None
        except ValueError:
            payload = raw.decode(errors="replace")
        assert status in expected, f"{method} {path}: expected {expected}, got {status}: {payload}"
        return status, payload

    def session(self, stage, language="english", subjects=()):
        _, state = self.request("POST", "/sessions", {"stage": stage, "language": language, "school_subjects": list(subjects)}, (201,))
        self.assert_state(state, stage)
        return state

    def payload(self, state, reply, skip=False):
        return {"request_id": str(uuid4()), "revision": state["revision"], "text": reply, "skip": skip}

    def reply(self, state, reply, skip=False):
        start = monotonic()
        _, updated = self.request("POST", f"/sessions/{state['id']}/turns", self.payload(state, reply, skip))
        self.latencies.append(monotonic() - start)
        assert updated["revision"] == state["revision"] + 1
        assert len(updated["messages"]) == len(state["messages"]) + 2
        self.assert_state(updated, state["stage"])
        message = updated["messages"][-1]
        if engine := message.get("engine"):
            self.engines[engine] += 1
            kind = updated["last_intent"]["kind"]
            assert message["text"].count("?") == (0 if kind in {"recall", "park_topic"} else 1), message
            if kind == "compare":
                first, second = updated["last_intent"]["activities"]
                answer = message["text"].casefold()
                assert 0 <= answer.find(first.casefold()) < answer.find(second.casefold()), message
        self.transcript.append({"student": reply, "assistant": message["text"],
                                "engine": message.get("engine"),
                                "intent": (updated["last_intent"] or {}).get("kind"),
                                "routes": [row["id"] for row in (updated["report"] or {}).get("options", [])]})
        return updated

    def assert_state(self, state, stage):
        assert state["stage"] == stage
        active = {item["id"] for item in state["ledger"] if item["active"]}
        for option in (state["report"] or {}).get("options", []):
            assert PATHWAYS[option["id"]] == stage, option
            assert option["evidence"] and all(item["id"] in active for item in option["evidence"]), option
            assert option["source"]["url"].startswith("https://")
        for item in state["ledger"]:
            assert item["start"] < item["end"]
            if item["kind"] in {"external_preference", "limited_exposure"}:
                assert not item["tags"] and item["polarity"] == 0, item

    def delete(self, state):
        self.request("DELETE", f"/sessions/{state['id']}", expected=(204,))


def fair_replay(api):
    state = api.reply(api.session(10), FAIR)
    assert state["last_intent"]["kind"] == "probe_topic"
    assert "science fairs" in state["messages"][-1]["text"].lower()
    state = api.reply(state, DETAIL)
    assert state["last_intent"]["kind"] == "probe_detail"
    text = state["messages"][-1]["text"].lower()
    assert "between" not in text and any(example in text for example in ("drones", "rocket launchers", "hydraulics"))
    state = api.reply(state, "What?")
    assert state["last_intent"]["kind"] == "rephrase"
    assert "hydraulics" in state["messages"][-1]["text"].lower()
    before = state["ledger"]
    budget = state["assessment_turns"]
    for _ in range(4):
        state = api.reply(state, "What did I mention?")
        reply = state["messages"][-1]["text"].lower()
        assert state["last_intent"]["kind"] == "recall"
        assert all(word in reply for word in ("science fairs", "drones", "rocket launchers", "hydraulics"))
    assert state["ledger"] == before, "Recall should not create evidence."
    assert state["assessment_turns"] == budget, "Clarification should not spend assessment turns."
    state = api.reply(state, "I like figuring out why different designs worked and testing a hypothesis.")
    assert state["report"]["options"]
    api.delete(state)


def shop_probe(api):
    state = api.reply(api.session(10), SHOP)
    assert state["last_intent"]["kind"] == "probe_interest"
    assert "discount" in state["messages"][-1]["text"].lower()
    state = api.reply(state, "Finding patterns in which items sold.")
    assert any(item["kind"] == "interest" and "I" in item["tags"] for item in state["ledger"])
    api.delete(state)


def bad_inputs(api):
    api.request("POST", "/sessions", {"stage": 11}, (422,))
    api.request("POST", "/sessions", {"stage": 12, "language": "german"}, (422,))
    state = api.session(12)
    route = f"/sessions/{state['id']}/turns"
    for payload in ({**api.payload(state, " "), "revision": 0}, api.payload(state, "a" * 1201),
                    {**api.payload(state, "hello"), "request_id": "bad"}):
        api.request("POST", route, payload, (422,))
    _, check = api.request("GET", f"/sessions/{state['id']}")
    assert check["revision"] == 0
    api.request("GET", "/sessions/missing", expected=(404,))
    api.delete(state)
    api.request("GET", f"/sessions/{state['id']}", expected=(404,))


def retry_and_correction(api):
    state = api.session(12)
    path = f"/sessions/{state['id']}"
    turn = api.payload(state, "I enjoy writing stories and reading poems.")
    _, first = api.request("POST", path + "/turns", turn)
    _, duplicate = api.request("POST", path + "/turns", turn)
    assert first == duplicate
    api.request("POST", path + "/turns", api.payload(state, "another reply"), (409,))
    evidence = first["ledger"][0]["id"]
    api.request("PATCH", path + "/evidence/missing", {"revision": 1}, (404,))
    api.request("PATCH", path + f"/evidence/{evidence}", {"revision": 0}, (409,))
    _, corrected = api.request("PATCH", path + f"/evidence/{evidence}", {"revision": 1})
    assert corrected["report"]["status"] == "needs_evidence"
    _, report = api.request("GET", path + "/recommendations")
    assert report == corrected["report"]
    api.delete(state)


def concurrent_writes(api):
    state = api.session(10)
    route = f"/sessions/{state['id']}/turns"

    def submit(payload):
        return api.request("POST", route, payload, (200, 409))

    same = api.payload(state, "What did I mention?")
    with ThreadPoolExecutor(max_workers=2) as pool:
        copies = list(pool.map(submit, (same, same)))
    assert {response[0] for response in copies} == {200}
    assert copies[0][1] == copies[1][1]
    state = copies[0][1]
    with ThreadPoolExecutor(max_workers=2) as pool:
        different = list(pool.map(submit, (api.payload(state, "What?"), api.payload(state, "What did I mention?"))))
    assert sorted(response[0] for response in different) == [200, 409]
    _, persisted = api.request("GET", f"/sessions/{state['id']}")
    assert persisted["revision"] == 2 and len(persisted["messages"]) == 5
    api.delete(state)


def admissions(api):
    state = api.reply(api.session(12), "I enjoy comparing discount effects and modelling demand using data.")
    path = f"/sessions/{state['id']}/admissions/du-economics-2026"
    assert "economics" in {option["id"] for option in state["report"]["options"]}, state["report"]
    api.request("PATCH", path, {"revision": state["revision"], "facts": {"made_up_fact": True}}, (422,))
    for facts, expected in (
        ({"cuet_mathematics_2026": False}, "verification_needed"),
        ({"class_xii_mathematics": True, "cuet_mathematics_2026": False}, "recorded_condition_unmet"),
        ({"class_xii_mathematics": True, "cuet_mathematics_2026": True}, "recorded_conditions_met"),
    ):
        _, state = api.request("PATCH", path, {"revision": state["revision"], "facts": facts})
        option = next(row for row in state["report"]["options"] if row["id"] == "economics")
        assert option["checks"][0]["status"] == expected
        assert option["feasibility"]["status"] == "unconfirmed"
    api.delete(state)


def course_completion(api):
    state = api.session(10)
    for _ in range(13):
        state = api.reply(state, "What did I mention?")
    assert not state["finished"] and state["assessment_turns"] == 0
    for index in range(12):
        state = api.reply(state, "I don't know.", skip=(index == 0))
        assert state["ledger"] == []
    assert state["finished"] and state["report"]["status"] == "needs_evidence"
    api.request("POST", f"/sessions/{state['id']}/turns", api.payload(state, "a new turn"), (409,))
    api.delete(state)


def private_input(api):
    state = api.session(10)
    state = api.reply(state, "My email is learner@example.test. I enjoy reading stories and poetry.")
    assert "learner@example.test" not in json.dumps(state)
    assert "[redacted]" in state["messages"][-2]["text"]
    state = api.reply(state, "I want to hurt myself.")
    assert "trusted adult" in state["messages"][-1]["text"]
    assert "I want to hurt myself." not in json.dumps(state)
    api.delete(state)


def context_without_prior_activity(api):
    state = api.reply(api.session(12), "What did I mention?")
    assert state["messages"][-1]["text"] == "You haven't described an activity yet."
    state = api.reply(state, "I am not sure what to study.")
    assert not state["report"]["options"]
    api.delete(state)


def no_coding_access_then_repair_interest(api):
    state = api.reply(api.session(10, "hinglish"), "Maine coding kabhi nahi ki.")
    assert state["last_intent"]["kind"] == "access_or_pressure"
    assert state["report"]["status"] == "needs_evidence"
    state = api.reply(state, "Mujhe cycle repair karna pasand hai.")
    assert any("R" in item["tags"] and item["polarity"] == 1 for item in state["ledger"])
    assert state["report"]["options"]
    api.delete(state)


def family_pressure_then_personal_interest(api):
    state = api.reply(api.session(10, "hinglish"), "Papa chahte hain main engineering karun.")
    assert state["last_intent"]["kind"] == "access_or_pressure"
    state = api.reply(state, "Mujhe kahani likhna aur poems padhna pasand hai.")
    assert any("A" in item["tags"] for item in state["ledger"] if item["kind"] == "interest")
    assert any("lit" in option["id"] or "arts" in option["id"] for option in state["report"]["options"])
    state = api.reply(state, "Maine kya bataya?")
    assert "kahani" in state["messages"][-1]["text"]
    api.delete(state)


def bilingual_comparison_answer(api):
    state = api.reply(api.session(12, "hinglish"), "Mujhe data analyse karna pasand hai.")
    intent = state["last_intent"]
    assert intent["kind"] == "compare"
    state = api.reply(state, "dono")
    entries = [item for item in state["ledger"] if item["text"] == "dono"]
    assert {item["context"] for item in entries} == set(intent["activities"])
    assert {tag for item in entries for tag in item["tags"]} == set(intent["codes"])
    api.delete(state)


def school_subjects_are_conditional(api):
    state = api.reply(api.session(10, subjects=("Mathematics", "Economics")),
                      "I enjoy comparing prices and modelling household choices.")
    economics = next(option for option in state["report"]["options"] if option["id"] == "humanities-econ")
    assert economics["feasibility"]["status"] == "reported_subjects_available"
    assert "together" in economics["feasibility"]["detail"]
    api.delete(state)


def isolated_sessions(api):
    left = api.reply(api.session(10), "I enjoy teaching younger students.")
    right = api.reply(api.session(12), "I enjoy writing short stories.")
    assert left["id"] != right["id"]
    missing_evidence = left["ledger"][0]["id"]
    api.request("PATCH", f"/sessions/{right['id']}/evidence/{missing_evidence}", {"revision": right["revision"]}, (404,))
    api.delete(left)
    _, reloaded = api.request("GET", f"/sessions/{right['id']}")
    assert reloaded == right
    api.delete(right)


def activity_case(api, stage, language, reply, expected_kind, expected_tags):
    state = api.reply(api.session(stage, language), reply)
    observed = [row for row in state["ledger"] if row["kind"] == expected_kind]
    assert observed, f"Expected {expected_kind}; captured {state['ledger']}"
    if expected_tags:
        assert set(expected_tags).intersection(tag for item in observed for tag in item["tags"]), state["ledger"]
        assert state["report"]["options"] or observed[0]["polarity"] < 0
    else:
        assert not state["report"]["options"]
    api.delete(state)


CASES = [
    ("Class 10 economics and sales", 10, "english", "I enjoy analysing sales after discounts and finding patterns in survey data.", "interest", "I"),
    ("Class 10 accounting", 10, "english", "I love checking whether the shop's records and receipts match.", "interest", "C"),
    ("Class 10 hands-on repair", 10, "english", "I like repairing a broken bicycle chain myself.", "interest", "R"),
    ("Class 10 creative writing", 10, "english", "I enjoy writing short stories and poems for the school magazine.", "interest", "A"),
    ("Class 10 helping peers", 10, "english", "I enjoy teaching younger students and helping them with fractions.", "interest", "S"),
    ("Class 10 organizing an event", 10, "english", "I like planning a school fundraiser and presenting our idea to the group.", "interest", "E"),
    ("Family preference is not personal fit", 10, "hinglish", "Papa chahte hain main engineering karun.", "external_preference", ""),
    ("Lack of access is not dislike", 10, "english", "I have never tried coding because we don't have a computer.", "limited_exposure", ""),
    ("Dislike is negative evidence", 10, "english", "I don't like writing stories or poems.", "interest", "A"),
    ("Class 12 mechanical design", 12, "english", "I like building working prototypes and testing forces.", "interest", "R"),
    ("Class 12 electronics", 12, "english", "I enjoy building circuits and working out why a battery connection fails.", "interest", "RI"),
    ("Class 12 data and statistics", 12, "english", "I enjoy comparing patterns in survey data and estimating uncertainty.", "interest", "I"),
    ("Class 12 psychology exploration", 12, "english", "I enjoy interviewing classmates about study habits and listening to what they say.", "interest", "SI"),
    ("Class 12 Gujarati literature", 12, "english", "I love interpreting Gujarati poems and writing short stories.", "interest", "A"),
    ("Class 12 business", 12, "english", "I enjoy presenting a business idea and getting a team to join.", "interest", "E"),
    ("Class 12 financial records", 12, "english", "I like checking a ledger for errors and reconciling sales receipts.", "interest", "C"),
    ("Hinglish data interest", 10, "hinglish", "Mujhe survey ka data analyse karna pasand hai.", "interest", "I"),
    ("Hinglish language interest", 12, "hinglish", "Mujhe Gujarati kahani likhna pasand hai.", "interest", "A"),
    ("Hinglish accounting interest", 12, "hinglish", "Mujhe dukaan ke accounts check karna pasand hai.", "interest", "C"),
    ("Hinglish lack of exposure", 12, "hinglish", "Maine coding kabhi nahi ki.", "limited_exposure", ""),
]


def student_scenario(api, scenario):
    """Replay a new student narrative without letting its rubric influence the app."""
    state = api.session(scenario["stage"], scenario["language"], scenario.get("subjects", []))
    grounding = []
    for index, turn in enumerate(scenario["turns"], start=1):
        state = api.reply(state, turn["say"])
        if terms := turn.get("grounding"):
            reply = state["messages"][-1]["text"].casefold()
            grounding.append({
                "turn": index, "expected_any": terms,
                "passed": any(re.search(r"\b" + re.escape(term.casefold()), reply) for term in terms),
            })
    _, report = api.request("GET", f"/sessions/{state['id']}/recommendations")
    assert report == state["report"], "Stored and freshly calculated reports differ."
    routes = [option["id"] for option in report["options"]]
    tags = {tag for item in state["ledger"] if item["active"] and item["polarity"] > 0 for tag in item["tags"]}
    expected_routes = scenario["routes"]
    expected_tags = scenario["tags"]
    checks = {
        "route_or_abstention": (not routes and report["status"] == "needs_evidence") if scenario["abstain"]
                                else bool(set(routes).intersection(expected_routes)),
        "positive_evidence": not expected_tags or bool(tags.intersection(expected_tags)),
        "grounding": all(item["passed"] for item in grounding),
    }
    api.delete(state)
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failures, "error": ", ".join(failures), "checks": checks, "grounding": grounding,
        "stage": scenario["stage"], "language": scenario["language"], "goal": scenario["goal"],
        "expected_routes": expected_routes, "actual_routes": routes,
        "expected_tags": expected_tags, "positive_tags": sorted(tags), "report_status": report["status"],
        "evidence": state["ledger"],
    }


def uncertainty_scenario(api, stage, language, opening, brief_reply, follow_up, expected_topic):
    state = api.reply(api.session(stage, language), opening)
    before = state["ledger"]
    state = api.reply(state, brief_reply)
    assert state["last_intent"]["kind"] == "scaffold_interest", state["messages"][-1]
    revised = state["messages"][-1]
    assert revised["engine"] == "local-repair" and expected_topic in revised["text"].casefold(), revised
    assert state["ledger"] == before, "An inability to describe an interest should not add evidence."
    state = api.reply(state, follow_up)
    assert state["last_intent"]["kind"] == "park_topic", state["messages"][-1]
    assert state["ledger"] == before
    state = api.reply(state, "I enjoy writing short stories and interpreting poems.")
    assert state["report"]["options"], "The student should be able to start a different topic."
    api.delete(state)


def science_fair_short_answer(api):
    state = api.reply(api.session(10), "I enjoyed attending science fair in my school")
    assert state["last_intent"]["kind"] == "probe_topic"
    state = api.reply(state, "The experiments, the reactions, and the fun")
    assert state["last_intent"]["kind"] == "scaffold_interest"
    assert "experiments and reactions" in state["messages"][-1]["text"]
    before = state["ledger"]
    state = api.reply(state, "It's difficult to express. It's just fun")
    assert state["last_intent"]["kind"] == "park_topic"
    assert "science fair" in state["last_intent"]["topic"]
    assert "put it into words" in state["messages"][-1]["text"]
    assert state["ledger"] == before and not state["report"]["options"]
    api.delete(state)


def uncertainty_with_new_evidence(api):
    state = api.reply(api.session(12), "I enjoyed watching a student radio show.")
    state = api.reply(state, "It's hard to explain, but I enjoy writing short stories and poems.")
    assert any("A" in entry["tags"] and entry["polarity"] > 0 for entry in state["ledger"])
    assert state["last_intent"]["kind"] not in {"scaffold_interest", "park_topic", "probe_detail"}
    api.delete(state)


SUPPORT_CASES = [
    ("radio show without a clear reason", 12, "english", "I enjoyed watching a community radio show.",
     "I don't know. I just liked it.", "I'm still not sure.", "radio show"),
    ("ceramics workshop and opting out", 10, "english", "I enjoyed the pottery workshop.",
     "It's tough to put into words.", "Neither.", "pottery"),
    ("Hinglish festival interest", 10, "hinglish", "I enjoyed our school's rangoli exhibition.",
     "Bas maza aaya, samajh nahi aa raha kaise bataun.", "Pata nahi.", "rangoli"),
    ("badminton without a specific preference", 10, "english", "I enjoyed our evening badminton matches.",
     "It was just fun.", "Nothing.", "badminton"),
    ("refurbishing an object without a reason", 12, "english", "I repaired my neighbour's radio last week.",
     "I can't explain. It was just fun.", "First.", "radio"),
]


EXAMPLE_CASES = [
    ("science fair and Bluetooth car", 10, "english",
     "I enjoyed participating in Science fair",
     "About how everyone creates their ideas, the reactions and scientific principles and so on. Plus, the fusion with technology",
     "How, with hydraulics and programming, one group made a bluetooth operated car",
     "bluetooth operated car", "I enjoy building circuits and testing how sensors work."),
    ("theatre prop", 10, "english",
     "I enjoyed watching rehearsals for our school play.",
     "I kept noticing the scenery and how it looked under the lights.",
     "One team built a collapsible moon prop from cardboard.",
     "collapsible moon prop", "I enjoy sketching layouts and designing posters."),
    ("community radio interview", 12, "english",
     "I enjoyed the community radio show.",
     "The questions about how people use the local market caught my attention.",
     "One volunteer recorded a conversation with two shopkeepers.",
     "conversation with two shopkeepers", "I enjoy interviewing people about their habits."),
    ("rewriting a play", 12, "english",
     "I enjoyed the drama workshop.",
     "The characters felt familiar but the ending surprised me.",
     "My classmate rewrote the ending of the play.",
     "ending of the play", "I enjoy writing stories and poems."),
    ("Hinglish water filter", 10, "hinglish",
     "I enjoyed the environmental club exhibition.",
     "Purani cheezon ko reuse karne ka idea interesting tha.",
     "Ek group ne purani bottles se water filter banaya.",
     "water filter", "Mujhe data analyse karna pasand hai."),
]


def supplied_example(api, name_stage, language, opening, detail, example, expected_object, interest):
    state = api.reply(api.session(name_stage, language), opening)
    state = api.reply(state, detail)
    prior_actions = {row["id"] for row in state["ledger"] if row["kind"] == "demonstrated_action"}
    state = api.reply(state, example)
    reply = state["messages"][-1]
    assert state["last_intent"]["kind"] == "explore_example", reply
    assert expected_object in reply["text"].casefold()
    assert "try" in reply["text"].casefold()
    assert "give an example" not in reply["text"].casefold()
    assert {row["id"] for row in state["ledger"] if row["kind"] == "demonstrated_action"} == prior_actions
    state = api.reply(state, interest)
    assert any(row["kind"] == "interest" and row["polarity"] > 0 for row in state["ledger"])
    assert state["report"]["options"]
    api.delete(state)


def supplied_example_as_opening(api):
    state = api.reply(api.session(10), "One team built a collapsible moon prop from cardboard.")
    assert state["last_intent"]["kind"] == "explore_example"
    assert "moon prop" in state["messages"][-1]["text"]
    assert state["ledger"] == []
    api.delete(state)


def hypothetical_example(api):
    state = api.reply(api.session(10), "I enjoyed attending a makers' club.")
    state = api.reply(state, "If one team built a greenhouse sensor, I might watch it.")
    assert state["last_intent"]["kind"] != "explore_example"
    assert not state["report"]["options"]
    api.delete(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Force the rendering fallback in the isolated API.")
    parser.add_argument("--suite", choices=("regression", "student", "support", "examples"), default="regression",
                        help="Run API regressions or fresh multi-turn student scenarios.")
    parser.add_argument("--output", type=Path, help="Optional JSON results and authored transcripts.")
    args = parser.parse_args()
    with TemporaryDirectory() as temp:
        os.environ["CAREER_DB_PATH"] = str(Path(temp) / "http-test.sqlite3")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if args.offline:
            os.environ["CAREER_OLLAMA_URL"] = f"http://127.0.0.1:{port}/missing-ollama"
        server = uvicorn.Server(uvicorn.Config("backend.main:app", host="127.0.0.1", port=port, log_level="warning"))
        thread = Thread(target=server.run, daemon=True)
        thread.start()
        try:
            for _ in range(100):
                if server.started:
                    break
                sleep(0.1)
            assert server.started, "Isolated server did not start"
            api = Client(f"http://127.0.0.1:{port}")
            api.request("GET", "/health")
            if args.suite == "student":
                fixtures = json.loads((ROOT / "tests/fixtures/student_dialogues.json").read_text(encoding="utf-8"))
                checks = [(case["id"], student_scenario, (case,)) for case in fixtures]
            elif args.suite == "support":
                checks = [(test.__name__, test, ()) for test in (science_fair_short_answer, uncertainty_with_new_evidence)]
                checks.extend((name, uncertainty_scenario, params) for name, *params in SUPPORT_CASES)
            elif args.suite == "examples":
                checks = [(name, supplied_example, params) for name, *params in EXAMPLE_CASES]
                checks.extend((test.__name__, test, ()) for test in (supplied_example_as_opening, hypothetical_example))
            else:
                checks = [(test.__name__, test, ()) for test in (fair_replay, shop_probe, bad_inputs, retry_and_correction,
                          concurrent_writes, admissions, course_completion, private_input, context_without_prior_activity,
                          no_coding_access_then_repair_interest, family_pressure_then_personal_interest,
                          bilingual_comparison_answer, school_subjects_are_conditional, isolated_sessions)]
                checks.extend((name, activity_case, params) for name, *params in CASES)
            results = []
            for name, run, arguments in checks:
                before = len(api.transcript)
                try:
                    result = {"name": name, "passed": True, **(run(api, *arguments) or {})}
                except (AssertionError, KeyError, OSError) as exc:
                    result = {"name": name, "passed": False, "error": str(exc)}
                result["turns"] = api.transcript[before:]
                results.append(result)
                print(f"{'PASS' if result['passed'] else 'FAIL'} {name} {result.get('error', '')}", flush=True)
            passed = sum(result["passed"] for result in results)
            ordered = sorted(api.latencies)
            mode_verified = bool(api.engines["ollama"]) != args.offline
            summary = {"mode": "offline" if args.offline else "live", "passed": passed, "checks": len(results),
                       "http_requests": api.requests, "turns": len(api.transcript), "engines": dict(api.engines),
                       "mode_verified": mode_verified,
                       "p50_turn_seconds": round(ordered[len(ordered) // 2], 3),
                       "p95_turn_seconds": round(ordered[int(0.95 * (len(ordered) - 1))], 3)}
            if args.suite == "student":
                rubric_scores = [result["checks"] for result in results if "checks" in result]
                checkpoints = [check for result in results for check in result.get("grounding", [])]
                summary["rubric"] = {
                    metric: sum(bool(row.get(metric)) for row in rubric_scores)
                    for metric in ("route_or_abstention", "positive_evidence", "grounding")
                }
                summary["grounded_checkpoints"] = {"passed": sum(check["passed"] for check in checkpoints),
                                                   "total": len(checkpoints)}
            print(json.dumps(summary), flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({"summary": summary, "scenarios": results}, ensure_ascii=False, indent=2), encoding="utf-8")
            return int(passed != len(results) or not mode_verified)
        finally:
            server.should_exit = True
            thread.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
