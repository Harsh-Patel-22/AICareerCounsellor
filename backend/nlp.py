import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).resolve().parents[1]
CODES = "RIASEC"
DIMENSIONS = dict(zip(CODES, ("Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional")))
STYLES = {
    "R": "building or troubleshooting something hands-on",
    "I": "investigating a question and finding a pattern",
    "A": "creating or interpreting an original work",
    "S": "listening to or helping someone",
    "E": "organising people around an idea",
    "C": "checking and organising detailed records",
}
GLOSSARY = {"analyse": "analyze", "analysing": "analyzing", "organise": "organize",
            "organising": "organizing", "accha": "achha", "acha": "achha",
            "nahin": "nahi", "nhi": "nahi", "mujko": "mujhe",
            "padana": "padhana", "samjana": "samjhana"}
SPLIT = re.compile(r"[.!?;\n]+|\b(?:but|lekin|magar|however)\b", re.I)
JOINED_ACTIVITIES = re.compile(r"\s+(?:and|aur)\s+", re.I)
EXTERNAL = re.compile(
    r"\b(?:parents?|family|mom|dad|mum|mother|father|papa|mummy|everyone at home|ghar\s*(?:wale|waale))\b"
    r"[^.!?;]{0,35}\b(?:wants?|says?|expects?|insists?|thinks?|would like|pressure|"
    r"chahte|chahti|chahata|bolte|bolti|kehte|kehti)\b",
    re.I,
)
EXPOSURE = re.compile(r"\b(?:never tried|never done|never coded|no access|no chance|"
                      r"not had a chance|don't have|do not have|"
                      r"haven't tried|have not tried|kabhi.{0,25}(?:nahi|nahin|nhi))\b", re.I)
NEGATIVE = re.compile(r"\b(?:dislike|hate|not enjoy|do not like|don't like|"
                      r"don't enjoy|not interested|pasand nahi|nahi pasand|"
                      r"not my thing)\b", re.I)
INTEREST = re.compile(
    r"\b(?:enjoy(?:ed|s)?|likes?|liked|love(?:d|s)?|prefer(?:red|s)?|would prefer|would rather|"
    r"want to\s+(?:use|work|explore|study|report|change|know|learn|do|try)|"
    r"hate|dislike|pasand|maz[ae]|achha lag|curious|curiosity|interesting|interest(?:ed)? in|dilchaspi)\b",
    re.I,
)
ACTION = re.compile(
    r"\b(?:built|tested|wrote|made|taught|volunteered|assembled|repaired|tracked|helped|"
    r"debugged|recorded|reconciled|compared|analysed|analyzed|"
    r"interviewed|explained|designed|sketched|coordinated|"
    r"organised|organized|pitched|checked|timed|drew|shot|counted|examined|mapped|sampled|edited|observed|"
    r"kiye|kiya|banaya|banayi|banaye|banana|badla|badli|badal|badalke|dekha|dekhi|seekha|seekhi|padhaya|padhai)\b",
    re.I,
)
EXAMPLE_RESULT = re.compile(
    r"\b(?:made|built|created|designed|rewrote|wrote|programmed|wired|played|recorded|tested|"
    r"demonstrated|mixed|changed|repaired|assembled|sketched|painted|compared|mapped|measured)\s+(?P<result>[^,]+)",
    re.I,
)
EXAMPLE_OBSERVER = re.compile(
    r"\b(?:i (?:saw|watched|noticed)|(?:one|another|the|a) (?:group|team|classmate|volunteer)|"
    r"(?:my|our) (?:friend|teacher|classmate)|they)\b",
    re.I,
)
HINGLISH_EXAMPLE = re.compile(
    r"\b(?:ek|dusre|kisi)\s+(?:group|team|classmate|student)\s+ne\s+(?P<result>[^,]+?)\s+"
    r"(?:banaya|banayi|badla|badli|likha|likhi|dikhaya|test kiya)\b",
    re.I,
)
STRENGTH = re.compile(r"\b(?:good at|comfortable with|do well in|comes easily|achhi aati|achchi aati)\b", re.I)
FIRST_PERSON_ACTION = re.compile(r"^(?P<speaker>i|maine)\s+", re.I)
STATED_PREFERENCE = re.compile(r"^i\s+(?:used to\s+)?(?:really\s+)?(?:enjoy(?:ed)?|lov(?:e|ed)|lik(?:e|ed))\s+(.+)", re.I)
NO_ACTIVITY = re.compile(r"\b(?:never|didn't|did not|don't|do not|would|might|could|want to|nahi)\b", re.I)
INTEREST_FRAGMENT = re.compile(
    r"^(?:finding|figuring|seeing|spotting|comparing|checking|working|noticing|building|making|writing|"
    r"designing|organizing|interviewing|testing|analyzing|reading|listening|helping|teaching)\b"
    r"|\b(?:dekhna|samajhna|dhundhna|compare karna|check karna)\s*$",
    re.I,
)
NO_INTEREST_FRAGMENT = re.compile(r"\b(?:nothing|none|no|not|nahi|bas|had to|was told|boring)\b", re.I)


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return re.sub(r"\b\w+\b", lambda match: GLOSSARY.get(match[0], match[0]), value)


def span_candidates(text: str):
    """Segment the raw input so evidence offsets still index the original text."""
    start = 0
    boundaries = [(match.start(), match.end()) for match in SPLIT.finditer(text)]
    for stop, next_start in [*boundaries, (len(text), len(text))]:
        raw = text[start:stop]
        trimmed = raw.strip(" \t\r\n,:–—")
        if trimmed:
            offset = start + raw.index(trimmed)
            yield trimmed, offset, offset + len(trimmed)
        start = next_start


def activity_to_probe(text: str) -> dict | None:
    """Find an activity worth discussing even if its Holland tag is uncertain."""
    for clause, _, _ in span_candidates(text):
        speaker = FIRST_PERSON_ACTION.match(clause)
        plain = normalize(clause)
        if not speaker or EXTERNAL.search(plain) or INTEREST.search(plain) or NO_ACTIVITY.search(plain):
            continue
        fragment = clause[speaker.end():]
        if len(fragment.split()) < 4 or not ACTION.search(plain):
            continue
        if speaker["speaker"].casefold() == "i" and not ACTION.match(normalize(fragment)):
            continue
        return {"anchor": " ".join(fragment.split()[:16]), "roman_hindi": speaker["speaker"].casefold() == "maine"}
    return None


def described_example(text: str) -> str:
    """Identify an outcome in a described event without assigning the student a role."""
    for clause, _, _ in span_candidates(text):
        roman_hindi = HINGLISH_EXAMPLE.search(clause)
        if not (event := roman_hindi or EXAMPLE_RESULT.search(clause)):
            continue
        if (not roman_hindi and not EXAMPLE_OBSERVER.search(clause)
                and not FIRST_PERSON_ACTION.match(clause) and not normalize(clause).startswith("we ")):
            continue
        if re.search(r"\b(?:if|never|didn't|did not|would|might|could)\b", normalize(clause[:event.start()])):
            continue
        words = event["result"].strip(" ,:–—\"“”").split()
        if 2 <= len(words) <= 13 and " ".join(words).casefold() not in {"it work", "it happen", "something similar"}:
            return " ".join(words)
    return ""


def describes_someone_elses_action(clause: str) -> bool:
    if HINGLISH_EXAMPLE.search(clause):
        return True
    event = EXAMPLE_RESULT.search(clause)
    return bool(event and EXAMPLE_OBSERVER.search(clause[:event.start()]))


def stated_topic(text: str) -> str:
    """Return the student's own activity phrase, without its trailing narration."""
    for clause, _, _ in span_candidates(text):
        match = STATED_PREFERENCE.match(clause)
        if not match:
            continue
        topic = re.split(r"\s+(?:that|which|because|when|where|at my|in my)\b", match[1], maxsplit=1, flags=re.I)[0]
        topic = topic.strip(" ,:–—\"“”")
        words = topic.split()
        if not words or len(words) > 10:
            continue
        if topic.casefold() not in {"it", "that", "this", "them", "those"}:
            return topic
    return ""


@dataclass(frozen=True)
class Observation:
    text: str
    start: int
    end: int
    kind: str
    tags: tuple[str, ...] = ()
    polarity: int = 0


def vectorizers(mode: str = "combined") -> FeatureUnion:
    features = [
        ("word", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2))),
        ("character", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(3, 5))),
    ]
    return FeatureUnion(features if mode == "combined" else [(key, estimator) for key, estimator in features if key == mode])


class EvidenceExtractor:
    def __init__(self, mode: str = "combined"):
        records = [json.loads(line) for line in (ROOT / "data/evidence_seed.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.mode = mode
        if mode == "rules":
            return
        self.features = vectorizers(mode)
        matrix = self.features.fit_transform([record["text"] for record in records])
        self.labels = MultiLabelBinarizer(classes=list(CODES))
        self.tags = OneVsRestClassifier(LogisticRegression(C=10, class_weight="balanced", max_iter=500))
        self.tags.fit(matrix, self.labels.fit_transform([record["tags"] for record in records]))
        self.kinds = LogisticRegression(C=8, class_weight="balanced", max_iter=500)
        self.kinds.fit(matrix, [record["construct"] for record in records])
        self.examples = matrix

    def extract(self, text: str) -> list[Observation]:
        result = []
        for clause, start, end in span_candidates(text):
            plain = normalize(clause)
            if EXTERNAL.search(plain):
                result.append(Observation(clause, start, end, "external_preference"))
                continue
            if EXPOSURE.search(plain):
                result.append(Observation(clause, start, end, "limited_exposure"))
                continue
            negative = bool(NEGATIVE.search(plain))
            interested = negative or bool(INTEREST.search(plain))
            done = bool(ACTION.search(plain)) and not describes_someone_elses_action(clause)
            strength = bool(STRENGTH.search(plain))
            if not (interested or done or strength):
                continue
            tags, predicted_kind = self.predict(clause)
            if not tags and interested and not negative:
                # Keep a supported activity if a less familiar second activity weakened the whole-clause match.
                first, *rest = JOINED_ACTIVITIES.split(clause, maxsplit=1)
                if rest:
                    tags, predicted_kind = self.predict(first)
                    if tags:
                        clause, end = first, start + len(first)
                        done = bool(ACTION.search(normalize(first))) and not describes_someone_elses_action(first)
            if not tags:
                continue
            kind = "interest" if interested else "demonstrated_action" if done else "self_reported_strength"
            if predicted_kind in {"external_preference", "limited_exposure"}:
                continue
            result.append(Observation(clause, start, end, kind, tags, -1 if negative else 1))
            if interested and done and not negative:
                result.append(Observation(clause, start, end, "demonstrated_action", tags, 1))
        return result

    def predict(self, text: str) -> tuple[tuple[str, ...], str]:
        if self.mode == "rules":
            keywords = {
                "R": "build|built|repair|fix|circuit|prototype|assembling",
                "I": "data|pattern|survey|math|experiment|analys|analy|hypothesis",
                "A": "story|stories|poetry|poem|draw|sketch|design|music|kahani",
                "S": "teach|taught|help|listen|interview|padhana|madad",
                "E": "lead|pitch|market|business|coordinat|fundrais",
                "C": "ledger|accounts|records|receipts|reconcil|organis",
            }
            return tuple(tag for tag, terms in keywords.items() if re.search(terms, normalize(text))), ""
        vector = self.features.transform([text])
        if (vector @ self.examples.T).max() < 0.35:
            return (), ""
        scores = self.tags.predict_proba(vector)[0]
        if max(scores) < 0.45:
            return (), ""
        order = sorted(range(len(CODES)), key=lambda index: scores[index], reverse=True)
        tags = tuple(CODES[index] for index in order[:2] if scores[index] >= max(0.40, max(scores) - 0.19))
        return tags, str(self.kinds.predict(vector)[0])

    def answer_to_interest_probe(self, text: str) -> list[Observation]:
        # An activity fragment can answer a question about preferences without restating "I like".
        candidates = list(span_candidates(text))
        if len(candidates) != 1:
            return []
        clause, start, end = candidates[0]
        plain = normalize(clause)
        if not INTEREST_FRAGMENT.search(plain) or NO_INTEREST_FRAGMENT.search(plain):
            return []
        tags, kind = self.predict(clause)
        if tags and kind not in {"external_preference", "limited_exposure", "no_evidence"}:
            return [Observation(clause, start, end, "interest", tags, 1)]
        return []


@lru_cache(maxsize=4)
def get_extractor(mode: str = "combined") -> EvidenceExtractor:
    return EvidenceExtractor(mode)
