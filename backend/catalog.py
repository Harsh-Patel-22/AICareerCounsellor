import json
from dataclasses import dataclass
from functools import lru_cache

from sklearn.metrics.pairwise import cosine_similarity

from backend.nlp import ROOT, normalize, vectorizers


def evaluate_requirement(expression: dict, facts: dict[str, bool]) -> bool | None:
    if "fact" in expression:
        return facts.get(expression["fact"])
    if "all_of" in expression:
        statuses = [evaluate_requirement(item, facts) for item in expression["all_of"]]
        return False if False in statuses else None if None in statuses else True
    statuses = [evaluate_requirement(item, facts) for item in expression["any_of"]]
    return True if True in statuses else None if None in statuses else False


@dataclass(frozen=True)
class Pathway:
    id: str
    stage: int
    stream: str
    title: str
    focus: str
    codes: str
    activities: list[str]
    subjects: list[str]
    source: str


class Catalog:
    def __init__(self):
        data = json.loads((ROOT / "catalog/pathways.json").read_text(encoding="utf-8"))
        self.pathways = tuple(Pathway(**record) for record in data["pathways"])
        self.sources = data["sources"]
        self.rules = data["rules"]
        self.version = data["version"]
        self.by_id = {pathway.id: pathway for pathway in self.pathways}
        self.activities = [(pathway.id, activity) for pathway in self.pathways for activity in pathway.activities]
        self.features = vectorizers()
        self.matrix = self.features.fit_transform([activity for _, activity in self.activities])

    def retrieve(self, evidence: list[str], stage: int, limit: int = 5) -> list[tuple[str, str, float]]:
        if not evidence:
            return []
        similarity = cosine_similarity(self.features.transform(evidence), self.matrix).max(axis=0)
        ordered = sorted(range(len(self.activities)), key=lambda index: (-similarity[index], index))
        return [
            (self.activities[index][0], self.activities[index][1], float(similarity[index]))
            for index in ordered
            if similarity[index] > 0 and self.by_id[self.activities[index][0]].stage == stage
        ][:limit]

    def feasibility(self, pathway: Pathway, school_subjects: list[str]) -> dict:
        if pathway.stage == 10 and pathway.subjects and school_subjects:
            offered = {normalize(subject).strip() for subject in school_subjects}
            if "maths" in offered or "math" in offered:
                offered.add("mathematics")
            missing = [subject for subject in pathway.subjects if subject not in offered]
            if missing:
                return {"status": "unconfirmed", "detail": "Check whether your school offers " + ", ".join(missing) + " in this combination."}
            return {"status": "reported_subjects_available", "detail": "The subjects are on your list. Ask your school whether it offers them together."}
        if pathway.stage == 10:
            return {"status": "unconfirmed", "detail": "Check this subject combination with your school and board."}
        return {"status": "unconfirmed", "detail": "Check program-specific prerequisites for the institution and intake year."}

    def checks(self, pathway: Pathway, facts: dict[str, dict[str, bool]]) -> list[dict]:
        flags = {True: "recorded_conditions_met", False: "recorded_condition_unmet", None: "verification_needed"}
        checks = []
        for rule in self.rules:
            if rule["pathway"] != pathway.id:
                continue
            supplied = facts.get(rule["id"], {})
            applies = "applies_when" not in rule or evaluate_requirement(rule["applies_when"], supplied) is True
            result = evaluate_requirement(rule["requires"], supplied) if applies else None
            checks.append({**rule, "status": flags[result]})
        return checks


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    return Catalog()
