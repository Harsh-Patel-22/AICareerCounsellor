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
            "nahin": "nahi", "nhi": "nahi", "mujko": "mujhe"}
SPLIT = re.compile(r"[.!?;\n]+|\b(?:but|lekin|magar|however)\b", re.I)
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
INTEREST = re.compile(r"\b(?:enjoy(?:ed|s)?|likes?|liked|love(?:d|s)?|prefer(?:red|s)?|"
                      r"hate|dislike|pasand|maz[ae]|achha lag|curious|interesting)\b", re.I)
ACTION = re.compile(r"\b(?:built|tested|wrote|made|taught|volunteered|assembled|repaired|"
                    r"debugged|recorded|reconciled|compared|analysed|analyzed|"
                    r"interviewed|explained|designed|sketched|coordinated|"
                    r"organised|organized|pitched|checked|kiye|kiya)\b", re.I)
STRENGTH = re.compile(r"\b(?:good at|comfortable with|do well in|comes easily|achhi aati|achchi aati)\b", re.I)


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
        records = [json.loads(line) for line in (ROOT / "data/evidence_seed.jsonl").read_text(encoding="utf-8").splitlines()]
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
            done = bool(ACTION.search(plain))
            strength = bool(STRENGTH.search(plain))
            if not (interested or done or strength):
                continue
            tags, predicted_kind = self.predict(clause)
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


@lru_cache(maxsize=4)
def get_extractor(mode: str = "combined") -> EvidenceExtractor:
    return EvidenceExtractor(mode)
