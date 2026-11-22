import re
from collections import defaultdict
from dataclasses import asdict
from uuid import uuid4

import numpy as np

from backend.catalog import get_catalog
from backend.nlp import CODES, DIMENSIONS, STYLES, Observation, activity_to_probe, described_example, get_extractor, normalize, span_candidates, stated_topic

TURN_CAP = 12
REFERENCE_CORRECTION = re.compile(
    r"\b(?:what|which)\s+(?:other\s+)?(?:activit(?:y|ies)|things?|options?)\s+did i (?:mention|say|list)\b"
    r"|\bwhat else did i (?:mention|say|list)\b"
    r"|\bi\s+(?:(?:mentioned|said|listed)\s+(?:no other|only one)"
    r"|(?:never|didn't|did not)\s+(?:mention|say|list)\s+(?:any\s+)?(?:other|another|anything else))\b",
    re.I,
)
RECALL = re.compile(r"^(?:what did i (?:mention|say|tell you)|what have i (?:mentioned|said)|maine kya (?:bola|kaha|bataya))$")
CONFUSION = re.compile(
    r"^(?:what|huh|what do you mean|i (?:don't understand|didn't get that)|can you rephrase(?: that)?"
    r"|kya matlab|samajh nahi aaya)$"
)
NO_DETAIL = {"i don't know", "i am not sure", "i'm not sure", "i'm still not sure", "i am still not sure",
             "nothing", "not sure", "pata nahi"}
HARD_TO_DESCRIBE = re.compile(
    r"\b(?:(?:hard|difficult|tough) to (?:explain|express|describe|put into words)"
    r"|(?:can't|cannot|don't know how to) (?:explain|describe|put (?:it|that) into words)"
    r"|samajh nahi aa raha (?:kaise|kya) bata|kaise (?:bataun|samjhaun))\b"
)
GENERAL_ENJOYMENT = re.compile(
    r"^(?:(?:it(?:'s| was| is)|i) )?(?:just )?(?:fun|nice|interesting|liked it|enjoyed it|"
    r"the fun|bas maza aaya|maza aaya)$"
)
BRIEF_CHOICE = {"first", "second", "third", "none", "neither", "pehla", "dusra", "teesra", "koi nahi"}


def is_conversation_repair(text: str) -> bool:
    return bool(RECALL.fullmatch(text) or CONFUSION.fullmatch(text) or REFERENCE_CORRECTION.search(text))


def lacks_detail(text: str) -> bool:
    clauses = [segment.strip() for clause, _, _ in span_candidates(text)
               for segment in normalize(clause).split(",") if segment.strip()]
    return bool(clauses) and all(
        clause in NO_DETAIL or GENERAL_ENJOYMENT.fullmatch(clause) or HARD_TO_DESCRIBE.search(clause)
        for clause in clauses
    )


def brief_examples(text: str) -> str:
    """Use the concrete parts of a short mixed list, leaving out feelings such as 'the fun'."""
    parts = re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+", text, flags=re.I)
    if not 1 < len(parts) <= 5 or len(text.split()) > 12:
        return ""
    if not any(GENERAL_ENJOYMENT.fullmatch(normalize(part).strip(" .!?")) for part in parts):
        return ""
    examples = [re.sub(r"^the\s+", "", part.strip(" .!?"), flags=re.I) for part in parts
                if part.strip(" .!?") and not GENERAL_ENJOYMENT.fullmatch(normalize(part).strip(" .!?"))]
    if not examples:
        return ""
    return ", ".join(examples[:-1]) + " and " + examples[-1] if len(examples) > 1 else examples[0]


def detail_excerpt(text: str) -> str:
    """Prefer the latest concrete clause over a long recap of the whole reply."""
    return next((clause for clause, _, _ in reversed(list(span_candidates(text)))
                 if 3 <= len(clause.split()) <= 22 and len(clause) <= 150), "")


def recent_context(messages: list[dict]) -> dict:
    detail = ""
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        text = message["text"]
        short = normalize(text).strip(" .!?")
        if (is_conversation_repair(short) or lacks_detail(text) or short == "skipped this question"
                or text.startswith("[")):
            continue
        if topic := stated_topic(text):
            return {"topic": topic, "detail": detail}
        if not detail:
            detail = detail_excerpt(text)
    return {"topic": "", "detail": detail}


def new_session(stage: int, language: str, school_subjects: list[str]) -> dict:
    if language == "hinglish":
        greeting = f"Class {stage} ke baad aap kya explore kar rahe hain? Kisi activity ya subject ka example share kar sakte hain."
    else:
        greeting = f"What are you considering after Class {stage}? You can start with something you enjoyed doing recently."
    return {
        "id": uuid4().hex, "revision": 0, "stage": stage, "language": language,
        "school_subjects": school_subjects, "messages": [{"role": "assistant", "text": greeting}],
        "ledger": [], "turns": 0, "assessment_turns": 0, "substantive_turns": 0, "last_intent": None,
        "asked": [], "report": None, "finished": False,
        "admission_facts": {},
    }


def active_evidence(state: dict) -> list[dict]:
    return [entry for entry in state["ledger"] if entry["active"]]


def profile(state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = np.full(len(CODES), 2.0)
    negative = np.full(len(CODES), 2.0)
    strength = np.zeros(len(CODES))
    seen = set()
    for entry in active_evidence(state):
        key = (normalize(entry["text"]), entry["kind"], entry.get("context", ""))
        if key in seen or not entry["tags"]:
            continue
        seen.add(key)
        weight = (0.7 if entry.get("context") else 1.0) / len(entry["tags"])
        for code in entry["tags"]:
            index = CODES.index(code)
            if entry["kind"] == "interest":
                (positive if entry["polarity"] > 0 else negative)[index] += weight
            if entry["kind"] in {"demonstrated_action", "self_reported_strength"} and entry["polarity"] > 0:
                strength[index] += weight * (1.0 if entry["kind"] == "demonstrated_action" else 0.35)
    return positive, negative, strength


def rank_paths(state: dict, draws: int = 256):
    catalog = get_catalog()
    paths = [path for path in catalog.pathways if path.stage == state["stage"]]
    positive, negative, strength = profile(state)
    interest = np.random.default_rng(423).beta(positive, negative, (draws, len(CODES)))
    weights = np.zeros((len(CODES), len(paths)))
    for index, path in enumerate(paths):
        for position, code in enumerate(path.codes):
            weights[CODES.index(code), index] = (len(path.codes) - position)
    weights /= weights.sum(axis=0)
    texts = [entry.get("context") or entry["text"] for entry in active_evidence(state)
             if entry["polarity"] > 0 and entry["kind"] in {"interest", "demonstrated_action"}]
    matches = defaultdict(float)
    for path_id, _, score in catalog.retrieve(texts, state["stage"], limit=15):
        matches[path_id] = max(matches[path_id], score)
    fit = (
        0.60 * (interest @ weights)
        + 0.22 * ((0.5 + 0.5 * (1 - np.exp(-0.6 * strength))) @ weights)
        + 0.18 * np.array([matches[path.id] for path in paths])
    )
    order = np.argsort(-fit.mean(axis=0), kind="stable")
    return paths, interest, fit, order


def decision_risk(fit: np.ndarray) -> np.ndarray:
    """Per-draw regret for choosing each lead, plus an action to offer two routes."""
    best = fit.max(axis=1, keepdims=True)
    leaders = np.argsort(-fit.mean(axis=0), kind="stable")[:2]
    shortlist = best[:, 0] - fit[:, leaders].max(axis=1) + 0.025
    return np.column_stack((best - fit, shortlist))


def comparison_value(interest: np.ndarray, fit: np.ndarray, first: str, second: str) -> float:
    gap = interest[:, CODES.index(first)] - interest[:, CODES.index(second)]
    choose_first = 0.75 / (1 + np.exp(-6.0 * gap))
    outcome_weights = (choose_first, 0.75 - choose_first, np.full(len(gap), 0.25))
    before = decision_risk(fit).mean(axis=0).min()
    after = 0.0
    for likelihood in outcome_weights:
        indices = np.argsort(-(likelihood @ fit), kind="stable")[:2]
        regret = fit.max(axis=1, keepdims=True) - fit
        shortlist = fit.max(axis=1) - fit[:, indices].max(axis=1) + 0.025
        after += min((likelihood @ regret).min() / len(gap), (likelihood @ shortlist) / len(gap))
    return float(before - after)


def select_intent(state: dict, current: list[Observation], latest: str, policy: str = "decision") -> dict:
    short = normalize(latest).strip(" .!?")
    if RECALL.fullmatch(short):
        return {"kind": "recall", "key": f"recall-{state['turns']}", **recent_context(state["messages"][:-1])}
    if CONFUSION.fullmatch(short):
        return {"kind": "rephrase", "key": f"rephrase-{state['turns']}", **recent_context(state["messages"][:-1])}
    if REFERENCE_CORRECTION.search(short):
        return {"kind": "repair_reference", "key": f"repair-{state['turns']}", **recent_context(state["messages"][:-1])}
    previous = state["last_intent"] or {}
    if previous.get("kind") in {"scaffold_interest", "explore_example", "park_topic"} and (lacks_detail(latest) or short in BRIEF_CHOICE):
        return {"kind": "park_topic", "key": f"park-{state['turns']}", "topic": previous.get("topic", "")}
    if lacks_detail(latest):
        context = recent_context(state["messages"][:-1])
        topic = previous.get("topic") or context["topic"]
        activity = previous.get("anchor", "") if previous.get("kind") == "probe_interest" else ""
        detail = previous.get("detail") or activity or context["detail"]
        if topic or detail:
            return {"kind": "scaffold_interest", "key": f"scaffold-{state['turns']}", "topic": topic, "detail": detail,
                    "quote_detail": bool(detail)}
    if current and not any(item.tags for item in current):
        if any(item.kind in {"limited_exposure", "external_preference"} for item in current):
            return {"kind": "access_or_pressure", "key": f"access-{state['turns']}"}
    named_topic = stated_topic(latest)
    if (not named_topic and (example := described_example(latest))
            and (previous.get("kind") in {"probe_topic", "probe_detail", "clarify", "scaffold_interest"}
                 or not previous and not activity_to_probe(latest))):
        return {"kind": "explore_example", "key": f"example-{state['turns']}",
                "topic": previous.get("topic", ""), "example": example}
    if previous.get("kind") == "probe_topic" and not named_topic and (detail := detail_excerpt(latest)):
        if examples := brief_examples(detail):
            return {"kind": "scaffold_interest", "key": f"scaffold-{state['turns']}",
                    "topic": previous["topic"], "detail": examples}
        if normalize(detail) != normalize(previous["topic"]) and short not in NO_DETAIL:
            detail = re.sub(r"^(?:plus|also)\s*[, ]+\s*", "", detail, flags=re.I) or detail
            return {"kind": "probe_detail", "key": f"detail-{state['turns']}", "topic": previous["topic"], "detail": detail}
    if state["last_intent"] is None or state["last_intent"]["kind"] != "probe_interest":
        activity = activity_to_probe(latest)
        if activity and not any(item.kind == "interest" for item in current):
            key = f"interest:{normalize(activity['anchor'])}"
            if key not in state["asked"]:
                return {"kind": "probe_interest", "key": key, **activity}
    if not any(item.tags for item in current):
        if named_topic:
            key = f"topic:{normalize(named_topic)}"
            if key not in state["asked"]:
                return {"kind": "probe_topic", "key": key, "topic": named_topic}
    if not state["substantive_turns"] or (latest and normalize(latest).strip(" .!?") not in {"neither", "koi nahi", "koi bhi nahi"}
                                          and not any(item.tags for item in current)):
        return {"kind": "clarify", "key": f"clarify-{state['turns']}", "anchor": latest[:170]}
    paths, interest, fit, order = rank_paths(state)
    counts = defaultdict(int)
    recent = {tag for item in current if item.polarity > 0 for tag in item.tags}
    for entry in active_evidence(state):
        for code in entry["tags"]:
            counts[code] += 1
    choices = []
    # Use distinct activity pairs from the leading options, then explore broader Holland dimensions.
    for i, left_index in enumerate(order[:5]):
        left = paths[left_index]
        for right_index in order[i + 1:i + 6]:
            right = paths[right_index]
            second = next((code for code in right.codes if code != left.codes[0] and code not in left.codes), right.codes[0])
            if left.codes[0] == second:
                continue
            first = left.codes[0]
            key = "".join(sorted((first, second)))
            if key in state["asked"]:
                continue
            coverage = 0.005 * (1 / (1 + counts[first]) + 1 / (1 + counts[second]))
            value = 0 if policy == "coverage" else comparison_value(interest, fit, first, second)
            grounding = 0.004 if recent.intersection((first, second)) else 0.0
            right_activity = right.activities[0] if second == right.codes[0] else STYLES[second]
            choices.append((value + coverage + grounding, {"kind": "compare", "key": key, "codes": [first, second],
                                                "activities": [left.activities[0], right_activity]}))
    for i, first in enumerate(CODES):
        for second in CODES[i + 1:]:
            key = "".join(sorted((first, second)))
            if key in state["asked"]:
                continue
            coverage = 0.005 * (1 / (1 + counts[first]) + 1 / (1 + counts[second]))
            value = 0 if policy == "coverage" else comparison_value(interest, fit, first, second)
            grounding = 0.004 if recent.intersection((first, second)) else 0.0
            choices.append((value + coverage + grounding - 0.002, {"kind": "compare", "key": key, "codes": [first, second],
                                                         "activities": [STYLES[first], STYLES[second]]}))
    if not choices:
        return {"kind": "clarify", "key": f"clarify-{state['turns']}", "anchor": latest[:170]}
    if recent:
        grounded = [choice for choice in choices if recent.intersection(choice[1]["codes"])]
        if grounded:
            choices = grounded
    return max(choices, key=lambda item: item[0])[1]


def comparison_answer(text: str, intent: dict | None) -> list[Observation]:
    if not intent or intent["kind"] != "compare":
        return []
    reply = normalize(text).strip(" .!?")
    label = {
        "first": [0], "the first": [0], "the first one": [0], "option 1": [0], "1": [0], "pehla": [0],
        "second": [1], "the second": [1], "the second one": [1], "option 2": [1], "2": [1], "dusra": [1],
        "both": [0, 1], "dono": [0, 1],
    }.get(reply, [])
    return [Observation(text, 0, len(text), "interest", (intent["codes"][index],), 1) for index in label]


def report(state: dict) -> dict:
    catalog = get_catalog()
    paths, _, fit, order = rank_paths(state)
    signals = [entry for entry in active_evidence(state) if entry["tags"]]
    positive_signals = [entry for entry in signals if entry["polarity"] > 0]
    if not positive_signals:
        return {"status": "needs_evidence", "options": [], "note": "Share a specific activity you enjoyed or tried to get a personalised shortlist."}
    supported_codes = {code for entry in positive_signals for code in entry["tags"]}
    supported_order = [index for index in order if supported_codes.intersection(paths[index].codes)]
    if not supported_order:
        return {"status": "needs_evidence", "options": [], "note": "Share another activity to help us match a pathway."}
    leader = supported_order[0]
    stability = float((np.argmax(fit, axis=1) == leader).mean())
    independent = {normalize(entry.get("context") or entry["text"]) for entry in positive_signals}
    explicit = [entry for entry in positive_signals if entry["kind"] == "interest"]
    has_conflict = (
        any(entry["polarity"] < 0 and paths[leader].codes[0] in entry["tags"] for entry in signals)
        and any(paths[leader].codes[0] in entry["tags"] for entry in explicit)
    )
    has_lead = len(independent) >= 2 and bool(explicit) and stability >= 0.70 and not has_conflict
    ranked = []
    for index in supported_order:
        candidate = paths[index]
        linked = [entry for entry in positive_signals if set(candidate.codes).intersection(entry["tags"])]
        label, url = catalog.sources[candidate.source]
        ranked.append({
            "id": candidate.id, "stream": candidate.stream, "title": candidate.title, "focus": candidate.focus,
            "evidence": [{"text": entry["text"], "id": entry["id"], "context": entry.get("context", ""),
                          "kind": entry["kind"], "tags": [DIMENSIONS[code] for code in entry["tags"]]}
                         for entry in {entry["text"] + entry.get("context", ""): entry for entry in linked}.values()][:3],
            "feasibility": catalog.feasibility(candidate, state["school_subjects"]),
            "checks": catalog.checks(candidate, state.get("admission_facts", {})),
            "rationale": f"This route explores {candidate.activities[0]} and {candidate.activities[1]}. Your answers below point to related interests or strengths.",
            "next_step": f"Try {candidate.activities[0]} for 15 minutes with a small example you can access. Note which part you want to keep exploring.",
            "source": {"label": label, "url": url},
        })
        if len(ranked) == 2:
            break
    if len(ranked) == 2:
        first, second = (catalog.by_id[item["id"]] for item in ranked)
        tradeoff = f"A useful difference to explore is whether you prefer {first.activities[0]} or {second.activities[0]}."
    else:
        tradeoff = "Try one of the study activities and share which part held your interest."
    return {"status": "lead" if has_lead else "provisional", "options": ranked, "note": tradeoff}


def add_turn(state: dict, text: str, skip: bool, render, policy: str = "decision") -> dict:
    repair = not skip and is_conversation_repair(normalize(text).strip(" .!?"))
    state["assessment_turns"] = state.get("assessment_turns", state["turns"]) + (not repair)
    state["turns"] += 1
    state["messages"].append({"role": "user", "text": "Skipped this question" if skip else text})
    latest = "" if skip else text
    comparison = [] if skip else comparison_answer(latest, state["last_intent"])
    observations = comparison or ([] if skip or lacks_detail(latest) else get_extractor().extract(latest))
    if not skip and not observations and state["last_intent"] and state["last_intent"]["kind"] in {"probe_interest", "probe_topic", "probe_detail", "explore_example"}:
        observations = get_extractor().answer_to_interest_probe(latest)
    if any(item.tags for item in observations):
        state["substantive_turns"] += 1
    for item in observations:
        entry = asdict(item)
        entry["tags"] = list(item.tags)
        entry.update(id=uuid4().hex, turn=state["turns"], active=True)
        if comparison:
            i = state["last_intent"]["codes"].index(item.tags[0])
            entry["context"] = state["last_intent"]["activities"][i]
        state["ledger"].append(entry)
    state["report"] = report(state)
    done = (not repair and (state["assessment_turns"] >= TURN_CAP
                            or (state["substantive_turns"] >= 6 and state["report"]["status"] == "lead")))
    if done:
        state["finished"] = True
        state["last_intent"] = None
        state["messages"].append({"role": "assistant", "text": "Here is your current report. You can review or correct the evidence in the sidebar."})
    else:
        intent = select_intent(state, observations, latest, policy)
        question, engine = render(intent, state["language"], latest, observations)
        state["last_intent"] = intent
        state["asked"].append(intent["key"])
        state["messages"].append({"role": "assistant", "text": question, "engine": engine})
    return state
