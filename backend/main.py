import re
from functools import lru_cache
from typing import Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.catalog import get_catalog
from backend.dialogue import add_turn, new_session, report
from backend.ollama import make_question
from backend.store import Conflict, Sessions

app = FastAPI(title="AI Career Counsellor")
DISTRESS = re.compile(r"\b(?:kill myself|end my life|hurt myself|want to die|suicide|mar jana chahta)\b", re.I)
CONTACT = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}|\b(?:\+?91[ -]?)?[6-9]\d{9}\b", re.I)


class SessionInput(BaseModel):
    stage: Literal[10, 12]
    language: Literal["english", "hinglish"] = "english"
    school_subjects: list[str] = Field(default_factory=list, max_length=16)


class TurnInput(BaseModel):
    request_id: UUID
    revision: int = Field(ge=0)
    text: str = Field(default="", max_length=1200)
    skip: bool = False


class RevisionInput(BaseModel):
    revision: int = Field(ge=0)


class AdmissionInput(RevisionInput):
    facts: dict[str, bool] = Field(max_length=12)


@lru_cache(maxsize=1)
def sessions() -> Sessions:
    return Sessions()


def require_session(session_id: str) -> dict:
    if state := sessions().get(session_id):
        return state
    raise HTTPException(404, "This session is missing or has expired.")


def save_state(state: dict, revision: int, request_id: str | None = None) -> dict:
    try:
        return sessions().update(state, revision, request_id)
    except Conflict:
        raise HTTPException(409, "The session changed. Refresh it and try again.") from None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/sessions", status_code=201)
def start_session(data: SessionInput) -> dict:
    subjects = [CONTACT.sub("[redacted]", item.strip()[:80]) for item in data.school_subjects if item.strip()]
    return sessions().create(new_session(data.stage, data.language, subjects))


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    return require_session(session_id)


@app.post("/sessions/{session_id}/turns")
def turn(session_id: str, data: TurnInput) -> dict:
    state = require_session(session_id)
    request_id = str(data.request_id)
    if sessions().has_request(session_id, request_id):
        return require_session(session_id)
    if state["revision"] != data.revision:
        raise HTTPException(409, "The session changed. Refresh it and try again.")
    if state["finished"]:
        raise HTTPException(409, "This conversation is complete.")
    if not data.skip and not data.text.strip():
        raise HTTPException(422, "Enter a reply or skip the question.")
    text = CONTACT.sub("[redacted]", data.text.strip())
    if not data.skip and DISTRESS.search(text):
        state["messages"].append({"role": "user", "text": "[Sensitive message omitted]"})
        message = ("That sounds difficult. If you might hurt yourself now, tell a trusted adult nearby "
                   "or contact local emergency services right away. You can pause career exploration here.")
        state["messages"].append({"role": "assistant", "text": message})
        return save_state(state, data.revision, request_id)
    return save_state(add_turn(state, text, data.skip, make_question), data.revision, request_id)


@app.get("/sessions/{session_id}/recommendations")
def recommendations(session_id: str) -> dict:
    return report(require_session(session_id))


@app.patch("/sessions/{session_id}/evidence/{evidence_id}")
def retract_evidence(session_id: str, evidence_id: str, data: RevisionInput) -> dict:
    state = require_session(session_id)
    matched = next((entry for entry in state["ledger"] if entry["id"] == evidence_id), None)
    if not matched:
        raise HTTPException(404, "Evidence was not found.")
    for entry in state["ledger"]:
        if (entry["turn"], entry["start"], entry["end"]) == (matched["turn"], matched["start"], matched["end"]):
            entry["active"] = False
    state["substantive_turns"] = len({entry["turn"] for entry in state["ledger"] if entry["active"] and entry["tags"]})
    state["report"] = report(state)
    return save_state(state, data.revision)


@app.patch("/sessions/{session_id}/admissions/{rule_id}")
def admission_facts(session_id: str, rule_id: str, data: AdmissionInput) -> dict:
    state = require_session(session_id)
    rule = next((entry for entry in get_catalog().rules if entry["id"] == rule_id), None)
    if not rule or state["stage"] != 12:
        raise HTTPException(404, "This requirement check was not found.")
    if data.facts.keys() - rule["questions"].keys():
        raise HTTPException(422, "Unexpected requirement facts.")
    state.setdefault("admission_facts", {})[rule_id] = data.facts
    state["report"] = report(state)
    return save_state(state, data.revision)


@app.delete("/sessions/{session_id}", status_code=204)
def delete(session_id: str) -> None:
    sessions().delete(session_id)
