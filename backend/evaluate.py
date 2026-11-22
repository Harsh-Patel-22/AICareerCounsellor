import argparse
import json
from collections import Counter

from backend.catalog import get_catalog
from backend.dialogue import add_turn, new_session
from backend.nlp import CODES, ROOT, get_extractor
from backend.ollama import fallback_question, local_chat


def render(intent, language, latest, observations):
    return fallback_question(intent, language, latest, observations), "evaluation"


def counts(expected, predicted):
    expected, predicted = Counter(expected), Counter(predicted)
    correct = sum((predicted & expected).values())
    return correct, sum(predicted.values()), sum(expected.values())


def metrics(totals):
    tp, predicted, expected = totals
    return {
        "precision": round(tp / max(1, predicted), 3),
        "recall": round(tp / max(1, expected), 3),
        "f1": round(2 * tp / max(1, predicted + expected), 3),
    }


def evaluate_evidence(examples):
    output = {}
    for mode in ("rules", "word", "character", "combined"):
        extractor = get_extractor(mode)
        totals = {"all": [0, 0, 0], "english": [0, 0, 0], "hinglish": [0, 0, 0]}
        for example in examples:
            predicted = [(item.kind, "".join(sorted(item.tags)), item.polarity) for item in extractor.extract(example["text"])]
            expected = [(kind, "".join(sorted(tags)), polarity) for kind, tags, polarity in example["labels"]]
            result = counts(expected, predicted)
            for key in ("all", example["language"]):
                totals[key] = [a + b for a, b in zip(totals[key], result)]
        output[mode] = {key: metrics(row) for key, row in totals.items()}
    return output


def evaluate_retrieval(profiles):
    ranks = []
    for case in profiles:
        matches = get_catalog().retrieve([case["text"]], case["stage"], limit=30)
        ordered = list(dict.fromkeys(path for path, _, _ in matches))[:5]
        ranks.append(next((i for i, path in enumerate(ordered, start=1) if path in case["acceptable"]), 0))
    return {"queries": len(ranks), "recall_at_5": round(sum(bool(rank) for rank in ranks) / len(ranks), 3),
            "mrr_at_5": round(sum(1 / rank for rank in ranks if rank) / len(ranks), 3)}


def evaluate_ollama(examples):
    labels = ["interest", "demonstrated_action", "self_reported_strength", "external_preference", "limited_exposure"]
    schema = {"type": "object", "properties": {"evidence": {"type": "array", "items": {
        "type": "object", "properties": {"quote": {"type": "string"}, "kind": {"enum": labels},
        "tags": {"type": "array", "items": {"enum": list(CODES)}}, "polarity": {"enum": [-1, 0, 1]}},
        "required": ["quote", "kind", "tags", "polarity"]}}}, "required": ["evidence"]}
    instruction = (
        "Extract vocational evidence from the English or Hinglish reply. Quote original spans verbatim. "
        "Holland codes are R Realistic, I Investigative, A Artistic, S Social, E Enterprising, C Conventional. "
        "Use polarity 1 for support, -1 for explicit dislike, and 0 with no tags for external preferences "
        "or limited exposure. Use no evidence for a vague subject name or an unrelated fact. Return JSON."
    )
    totals = [0, 0, 0]
    failures = 0
    for case in examples:
        try:
            raw = local_chat(instruction, {"reply": case["text"]}, schema, timeout=45)
            predicted = [(entry["kind"], "".join(sorted(entry["tags"])), entry["polarity"])
                         for entry in raw["evidence"] if entry["quote"] and entry["quote"] in case["text"]]
        except (OSError, ValueError, KeyError, TypeError):
            failures += 1
            predicted = []
        expected = [(kind, "".join(sorted(tags)), polarity) for kind, tags, polarity in case["labels"]]
        totals = [a + b for a, b in zip(totals, counts(expected, predicted))]
    return {**metrics(totals), "examples": len(examples), "failed_requests": failures}


def evaluate_policy(profiles, policy, turns):
    results = []
    for case in profiles:
        state = new_session(case["stage"], "english", [])
        text = case["text"]
        first_hit = None
        for turn in range(1, turns + 1):
            state = add_turn(state, text, False, render, policy)
            lead = [item["id"] for item in state["report"]["options"]]
            if first_hit is None and set(lead).intersection(case["acceptable"]):
                first_hit = turn
            intent = state["last_intent"]
            if state["finished"]:
                break
            if not intent or intent["kind"] != "compare":
                text = "I'm still unsure."
                continue
            ranking = {code: index for index, code in enumerate(case["priority"])}
            first, second = (ranking.get(code, 10) for code in intent["codes"])
            text = "neither" if first == second else "first" if first < second else "second"
        results.append((lead, case["acceptable"], first_hit))
    return {
        "profiles": len(results), "turn_budget": turns,
        "top_two_hit_rate": round(sum(bool(set(lead).intersection(acceptable)) for lead, acceptable, _ in results) / len(results), 3),
        "first_acceptable_turn": [turn for _, _, turn in results],
        "lead_hit_rate": round(sum(bool(lead and lead[0] in acceptable) for lead, acceptable, _ in results) / len(results), 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the authored career counsellor development fixtures.")
    parser.add_argument("--turns", type=int, choices=range(1, 13), default=8)
    parser.add_argument("--ollama", action="store_true", help="Include the slow prompt-only extraction baseline on the same cases.")
    args = parser.parse_args()
    examples = [json.loads(row) for row in (ROOT / "data/evidence_eval.jsonl").read_text(encoding="utf-8").splitlines()]
    profiles = json.loads((ROOT / "data/policy_profiles.json").read_text(encoding="utf-8"))
    results = {
        "scope": "Small authored development benchmark. Neither a human study nor a sealed test set.",
        "evidence_examples": len(examples), "evidence": evaluate_evidence(examples),
        "retrieval": evaluate_retrieval(profiles),
        "policy": {policy: evaluate_policy(profiles, policy, args.turns) for policy in ("coverage", "disagreement", "decision")},
    }
    if args.ollama:
        results["ollama_extraction"] = evaluate_ollama(examples)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
