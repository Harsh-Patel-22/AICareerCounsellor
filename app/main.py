import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import streamlit as st

API = os.environ.get("CAREER_API_URL", "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="Career Counsellor", layout="centered")


def call(method: str, path: str, body: dict | None = None):
    request = Request(
        API + path, data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Content-Type": "application/json"}, method=method,
    )
    with urlopen(request, timeout=75) as response:
        return None if response.status == 204 else json.load(response)


def perform(method: str, path: str, body: dict | None = None) -> bool:
    try:
        st.session_state.career_session = call(method, path, body)
        st.session_state.pop("career_error", None)
        return True
    except HTTPError as exc:
        if exc.code == 404:
            st.session_state.career_session = None
            message = "This session expired. Start a new conversation to continue."
        elif exc.code == 409:
            message = "The session changed. Refresh it before sending another reply."
        else:
            message = f"The counsellor couldn't complete that step (HTTP {exc.code}). Try again."
    except (URLError, TimeoutError, OSError, ValueError):
        message = "The counsellor API is unreachable. Start the backend, then try again."
    st.session_state.career_error = message
    return False


def send_turn(state: dict, text: str, skip: bool = False, finish: bool = False) -> None:
    previous = st.session_state.get("pending_turn")
    if not previous or (previous["text"], previous["skip"], previous.get("finish")) != (text, skip, finish):
        previous = {"request_id": str(uuid4()), "revision": state["revision"], "text": text, "skip": skip, "finish": finish}
        st.session_state.pending_turn = previous
    if perform("POST", f"/sessions/{state['id']}/turns", previous):
        st.session_state.pop("pending_turn", None)
    st.rerun()


def requirement_check(state: dict, check: dict) -> None:
    labels = {"recorded_conditions_met": "Recorded conditions met", "recorded_condition_unmet": "Recorded condition unmet",
              "verification_needed": "Verification needed"}
    with st.expander(check["scope"]):
        st.write(check["detail"])
        st.write(labels[check["status"]])
        current = state.get("admission_facts", {}).get(check["id"], {})
        choices = ["Unknown", "Yes", "No"]
        with st.form(f"check-{check['id']}-{state['revision']}"):
            facts = {}
            for field, prompt in check["questions"].items():
                index = 0 if field not in current else 1 if current[field] else 2
                answer = st.radio(prompt, choices, index=index, horizontal=True)
                if answer != "Unknown":
                    facts[field] = answer == "Yes"
            if st.form_submit_button("Check recorded conditions"):
                perform("PATCH", f"/sessions/{state['id']}/admissions/{check['id']}",
                        {"revision": state["revision"], "facts": facts})
                st.rerun()


def show_report(state: dict) -> None:
    result = state["report"]
    if not result or not result["options"]:
        st.markdown("Your shortlist will appear once you share something specific you enjoyed or tried.")
        return
    st.markdown(result["note"])
    for index, option in enumerate(result["options"]):
        if result["status"] == "lead":
            label = "Suggested route" if index == 0 else "Alternative"
        else:
            label = "Route to explore"
        with st.container(border=True):
            header_col1, header_col2 = st.columns([3, 1])
            with header_col1:
                st.caption(f"{label} · {option['stream']}")
                st.subheader(option["title"])
            with header_col2:
                score = option.get("score", 75)
                st.metric("Overall Match", f"{score}%")

            if "breakdown" in option:
                bd = option["breakdown"]
                st.markdown("**Multi-Factor Alignment:**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.caption(f"🎯 Interest Fit: **{bd.get('interest_fit', 0)}%**")
                    st.progress(bd.get("interest_fit", 0) / 100.0)
                with c2:
                    st.caption(f"💪 Demonstrated Skills: **{bd.get('skills_fit', 0)}%**")
                    st.progress(bd.get("skills_fit", 0) / 100.0)
                with c3:
                    st.caption(f"🔍 Activity Alignment: **{bd.get('activity_fit', 0)}%**")
                    st.progress(bd.get("activity_fit", 0) / 100.0)

            st.write(option["focus"])
            st.write(option["rationale"])
            st.markdown("**What you shared**")
            for entry in option["evidence"]:
                context = f" · {entry['context']}" if entry["context"] else ""
                st.text(f"“{entry['text']}”{context}")
                st.caption("Interest" if entry["kind"] == "interest" else "Strength or activity evidence")
            st.markdown("**What to check**")
            st.write(option["feasibility"]["detail"])
            st.link_button(option["source"]["label"], option["source"]["url"])
            for check in option["checks"]:
                requirement_check(state, check)
            st.markdown("**Something to try**")
            st.write(option["next_step"])


def sidebar(state: dict) -> None:
    path = f"/sessions/{state['id']}"
    with st.sidebar:
        st.header("Your session")
        language = "Hinglish" if state["language"] == "hinglish" else "English"
        st.caption(f"After Class {state['stage']} · {language}")
        entries = {entry["id"]: entry for entry in state["ledger"] if entry["active"]}
        if entries:
            st.subheader("Review what we heard")
            selected = st.selectbox(
                "Choose an answer to remove from the assessment",
                list(entries),
                format_func=lambda key: entries[key]["text"],
                index=None,
                placeholder="Choose an answer",
            )
            if st.button("Remove from assessment", disabled=selected is None) and selected:
                perform("PATCH", f"{path}/evidence/{selected}", {"revision": state["revision"]})
                st.rerun()
        if st.button("Refresh session"):
            if perform("GET", path):
                st.session_state.pop("pending_turn", None)
            st.rerun()
        with st.expander("Delete conversation"):
            st.write("This removes the chat and its assessment from the local database.")
            if st.button("Delete this session"):
                if perform("DELETE", path):
                    st.session_state.pop("pending_turn", None)
                st.rerun()


st.title("Career Counsellor")
error = st.session_state.get("career_error")
if error:
    st.error(error)
state = st.session_state.get("career_session")
if not state:
    st.markdown("Explore subject choices after Class 10 or degree options after Class 12.")
    with st.form("student_setup"):
        stage = st.radio("Which decision are you making?", [10, 12], format_func=lambda value: f"After Class {value}")
        language = st.selectbox("Conversation language", ["English", "Hinglish"])
        subjects = st.text_input("Subjects your school offers (optional, for Class 10)", placeholder="e.g. Economics, Mathematics")
        submitted = st.form_submit_button("Start conversation", type="primary")
    st.caption("Please leave out your name and contact details. You can delete this session at any time.")
    if submitted:
        perform("POST", "/sessions", {"stage": stage, "language": language.lower(),
                                      "school_subjects": [part.strip() for part in subjects.split(",") if part.strip()][:16] if stage == 10 else []})
        st.rerun()
    st.stop()

sidebar(state)
conversation, recommendations = st.tabs(["Conversation", "Your report"])
with conversation:
    for message in state["messages"]:
        with st.chat_message(message["role"]):
            st.text(message["text"])

    rep = state.get("report")
    if rep and rep.get("options"):
        with st.expander("📊 Live Recommendations & Match Scores (Updated in real-time)", expanded=True):
            st.caption("Here is your current shortlist based on what you have shared so far:")
            for opt in rep["options"]:
                st.markdown(f"⭐ **{opt['title']}** ({opt['stream']}) — **{opt.get('score', 75)}% Match**")
                if "breakdown" in opt:
                    b = opt["breakdown"]
                    st.caption(f"🎯 Interest Fit: {b.get('interest_fit')}%  ·  💪 Skills Fit: {b.get('skills_fit')}%  ·  🔍 Activity Alignment: {b.get('activity_fit')}%")
            if not state["finished"]:
                st.write("")
                col_fin, col_hint = st.columns([1, 2])
                with col_fin:
                    if st.button("✅ Done with this — Finalize Shortlist", type="primary", key="btn_finalize_early"):
                        send_turn(state, "", finish=True)
                with col_hint:
                    st.caption("Satisfied with these options? Finalize now, or reply below to refine further.")

    if not state["finished"]:
        if st.button("Skip this question"):
            send_turn(state, "", skip=True)
        if reply := st.chat_input("Write a reply to refine further, or type 'done' if finished...", max_chars=1200):
            with st.spinner("Preparing the next response…"):
                send_turn(state, reply)
with recommendations:
    show_report(state)
