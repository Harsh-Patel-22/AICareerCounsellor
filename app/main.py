import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import streamlit as st

API = os.environ.get("CAREER_API_URL", "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="AI Career Counsellor", page_icon="🧭", layout="centered")

CUSTOM_CSS = """
<style>
    /* Clean layout styling */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .stream-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    .stream-pcm { background: #dbeafe; color: #1e40af; }
    .stream-comm { background: #dcfce7; color: #166534; }
    .stream-arts { background: #f3e8ff; color: #6b21a8; }
    /* Smoother chat message rendering */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
    result = state.get("report")
    if not result or not result.get("options"):
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


def show_nlp_inspector(state: dict) -> None:
    st.header("🔬 NLP Pipeline & Linguistic Explainability")
    st.markdown(
        "This interactive dashboard visualizes the internal **NLP feature representations**, "
        "**information extraction spans**, **intent attribution**, and **vector space retrieval** "
        "derived in real time from the conversation."
    )

    # 1. Pipeline Overview
    with st.expander("📐 End-to-End NLP Architecture Flowchart", expanded=False):
        st.markdown("""
```mermaid
flowchart LR
    A["Raw User Utterance<br/>(English / Hinglish)"] --> B["NLP Normalization<br/>(NFKC + Hinglish Transliteration)"]
    B --> C["Clause Segmentation<br/>& Negation Detection"]
    C --> D["Intent Attribution<br/>(Self vs Family Pressure vs Exposure)"]
    D --> E["FeatureUnion<br/>(Word n-grams 1-2 + Char n-grams 3-5)"]
    E --> F["Multi-Label RIASEC<br/>Classifier (OneVsRest)"]
    F --> G["TF-IDF Vector Space<br/>Cosine Similarity"]
    G --> H["Bayesian Belief &<br/>Constrained Dialogue Policy"]
    H --> I["Pathway Shortlist &<br/>Dynamic Ollama Question"]
```
        """)
        st.caption("Architecture: Raw input undergoes text normalization, linguistic clause segmentation, entity disambiguation, subword n-gram vectorization, and multi-label RIASEC classification before computing catalog cosine similarity.")

    # 2. Real-time Holland RIASEC Psychometric Vector Distribution
    st.subheader("1. Holland RIASEC Psychometric Vector Space")
    st.caption("Predicted posterior probability distribution across the 6 Holland personality & career dimensions:")

    rep = state.get("report") or {}
    profile_data = rep.get("riasec_profile") or {}
    if profile_data and any(profile_data.values()):
        import pandas as pd
        df_profile = pd.DataFrame(list(profile_data.items()), columns=["Holland Dimension", "Alignment Score (%)"])
        st.bar_chart(df_profile.set_index("Holland Dimension"), color="#2563eb", height=280)

        # Highlight Dominant Code
        sorted_dims = sorted(profile_data.items(), key=lambda x: x[1], reverse=True)
        top1, top2 = sorted_dims[0], sorted_dims[1]
        col_code1, col_code2 = st.columns(2)
        with col_code1:
            st.info(f"🎯 **Primary Holland Code:** `{top1[0][0]}` ({top1[0]} — **{top1[1]}%**)")
        with col_code2:
            st.info(f"⚡ **Secondary Holland Code:** `{top2[0][0]}` ({top2[0]} — **{top2[1]}%**)")
    else:
        st.info("Share an activity or interest in the conversation to compute the live RIASEC psychometric distribution.")

    st.write("---")

    # 3. Linguistic Information Extraction & Attribution Table
    st.subheader("2. Linguistic Clause Segmentation & Intent Attribution")
    st.caption(
        "Demonstrates how the NLP engine isolates authentic student interests while filtering out "
        "parental pressure (*'Papa chahte hain...'*) and lack of exposure (*'Maine coding kabhi nahi ki'*):"
    )

    ledger_entries = [entry for entry in state.get("ledger", []) if entry.get("active")]
    if ledger_entries:
        import pandas as pd
        rows = []
        for e in ledger_entries:
            kind = e.get("kind", "")
            pol = e.get("polarity", 1)
            tags = ", ".join(e.get("tags", [])) or "None"

            if kind == "external_preference":
                decision = "🛡️ Family/External Pressure (Filtered - 0 penalty)"
                kind_badge = "Parental Advice"
            elif kind == "limited_exposure":
                decision = "ℹ️ Lack of Access (Zero penalty)"
                kind_badge = "No Prior Access"
            elif pol < 0:
                decision = "⛔ Negative Preference (-1 evidence)"
                kind_badge = "Disliked Activity"
            elif kind == "demonstrated_action":
                decision = "💪 Student Demonstrated Action (+1 evidence)"
                kind_badge = "Action Performed"
            else:
                decision = "🎯 Genuine Student Interest (+1 evidence)"
                kind_badge = "Personal Interest"

            rows.append({
                "Turn": e.get("turn", "-"),
                "Extracted Text Span": f"“{e.get('text', '')}”",
                "Construct Kind": kind_badge,
                "Holland RIASEC": tags,
                "NLP Attribution": decision,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.markdown("*No active extracted clauses yet. Start chatting to see real-time NLP attribution.*")

    st.write("---")

    # 4. Vector Space Model: Cosine Similarity Matching
    st.subheader("3. Catalog Vector Space Semantic Retrieval")
    st.markdown(r"Mathematical Formulation: $\quad \text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2}$")
    st.caption("Top matching study pathway activities retrieved via TF-IDF Vector Space similarity against student utterances:")

    cat_matches = rep.get("catalog_matches") or []
    if cat_matches:
        import pandas as pd
        st.dataframe(pd.DataFrame(cat_matches), use_container_width=True, hide_index=True)
    else:
        st.markdown("*Vector space matches will appear once activities are discussed.*")

    st.write("---")

    # 5. Downloadable NLP Dossier
    st.subheader("4. Student Career & NLP Evaluation Dossier")
    st.caption("Download the complete machine-readable audit report for evaluation and submission grading:")
    dossier = {
        "session_id": state["id"],
        "stage": state["stage"],
        "language": state["language"],
        "turns": state["turns"],
        "messages": state["messages"],
        "riasec_profile": profile_data,
        "extracted_ledger": ledger_entries,
        "recommended_options": rep.get("options") or [],
    }
    st.download_button(
        label="📥 Download Complete NLP Assessment Dossier (.json)",
        data=json.dumps(dossier, indent=2, ensure_ascii=False),
        file_name=f"career_nlp_dossier_{state['id'][:8]}.json",
        mime="application/json",
        type="primary",
    )


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


st.markdown("""
<div style="background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%); padding: 18px 22px; border-radius: 12px; margin-bottom: 20px; color: white; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
    <h2 style="margin: 0; font-size: 24px; font-weight: 700; color: white;">🧭 AI Career Counsellor</h2>
    <p style="margin: 4px 0 0 0; font-size: 13px; opacity: 0.9;">
        Conversational Career Guidance for Indian Students after Class 10 & 12 • Holland RIASEC & NLP Pipeline
    </p>
</div>
""", unsafe_allow_html=True)
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
conversation, recommendations, nlp_inspector = st.tabs(["💬 Conversation", "📋 Your Report", "🔬 NLP Pipeline & Explainability"])
with conversation:
    for message in state["messages"]:
        role = message["role"]
        avatar = "🤖" if role == "assistant" else "🧑‍🎓"
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["text"])
            if message.get("engine") == "ollama":
                st.caption("⚡ *Contextual question synthesized live by Ollama (Llama-3.2)*")
            elif message.get("engine") == "ollama-offline":
                st.error("⚠️ Local Ollama service is unreachable. Please run `ollama serve`.")

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
        if st.button("⏭️ Skip this question"):
            with st.spinner("Moving to next topic..."):
                send_turn(state, "", skip=True)
        if reply := st.chat_input("Write a reply to refine further, or type 'done' if finished...", max_chars=1200):
            # Optimistically render user message immediately in chat
            with st.chat_message("user", avatar="🧑‍🎓"):
                st.markdown(reply)
            # Show assistant thinking container with spinner
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking & generating personalized question with AI model…"):
                    send_turn(state, reply)
with recommendations:
    show_report(state)
with nlp_inspector:
    show_nlp_inspector(state)
