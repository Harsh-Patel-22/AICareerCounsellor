import json
import os
import re
from urllib.error import URLError
from urllib.request import Request, urlopen

BASE_URL = os.environ.get("CAREER_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


def resolve_model() -> str:
    if env := os.environ.get("CAREER_OLLAMA_MODEL"):
        return env
    try:
        req = Request(f"{BASE_URL}/api/tags")
        with urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read())
            models = [m.get("name") or m.get("model") for m in data.get("models", [])]
            for m in models:
                if "llama3.2" in m:
                    return m
            if models:
                return models[0]
    except Exception:
        pass
    return "llama3.2:3b"


MODEL = os.environ.get("CAREER_OLLAMA_MODEL", "llama3.2:3b")
QUESTION_SCHEMA = {
    "type": "object", "properties": {"question": {"type": "string"}},
    "required": ["question"], "additionalProperties": False,
}
INSTRUCTIONS = (
    "You are an encouraging, expert AI Career Counsellor for students in India exploring academic streams after Class 10/12 "
    "(Science PCM/PCB, Commerce, Humanities/Arts).\n"
    "GOAL: Guide the student towards their ideal stream, sub-stream, or college degree by discovering their authentic strengths, "
    "cognitive problem-solving style (analytical, creative, practical, investigative), and interests without rigid surveys.\n"
    "CRITICAL RULES:\n"
    "1. NEVER ask cliché questions like 'what part of that stayed with you', 'what caught your interest', or repeat what the student just answered.\n"
    "2. Every question MUST add clear DIAGNOSTIC VALUE towards differentiating streams and specializations "
    "(e.g. software/algorithm engineering vs computer systems/hardware, pure science research vs applied technology, "
    "finance/accounting vs corporate management, creative media vs clinical psychology).\n"
    "3. Ground your question in real-world applications, concrete projects, or problem-solving scenarios.\n"
    "4. Return valid JSON with key 'question' containing exactly ONE clear, conversational question ending with '?'."
)
CLARIFY_INSTRUCTIONS = (
    "Ask an insightful question exploring what specific applications, real-world problems, or projects the student would be curious to tackle."
)
COMPARE_INSTRUCTIONS = (
    "Ask an engaging question comparing the two activities or directions provided. Ask which type of work, challenge, or practical problem they would find more rewarding, allowing both or neither."
)
TOPIC_INSTRUCTIONS = (
    "The student expressed interest in an academic/practical domain. Ask an insightful diagnostic question exploring what specific concepts, applications, or challenges in that domain excite them the most."
)
DETAIL_INSTRUCTIONS = (
    "Explore deeper into the student's interest. Ask what specific practical problems, creations, or projects within this domain they would be eager to solve or build."
)
PROBE_INSTRUCTIONS = (
    "Ask an insightful question about the student's preferred role and problem-solving approach in this domain: do they enjoy building, analyzing, designing, or leading?"
)
PROBE_EXAMPLES = {
    "english": "For an activity about organising a club timetable, ask: What part of organising the club timetable did you find interesting?",
    "hinglish": "For an activity about organising a club timetable, ask: Club ka timetable banate waqt aapko kaunsa part interesting laga?",
}
HINGLISH = re.compile(r"\b(?:aap\w*|kya|kaunsa|kaunsi|kis|bata|pasand)\b", re.I)
UNSUPPORTED_CHOICE = re.compile(
    r"\b(?:between|prefer(?:red|s)?|compare|or (?:the )?other|(?:other |the )?(?:activities|options|ones) "
    r"(?:that )?you mentioned|in dono|ya doosr[eia])\b", re.I,
)
ACTIVITY_ROLE = re.compile(r"\b(?:participat\w*|present\w*|organi[sz]\w*|host\w*|your project)\b", re.I)
PROBE_CUES = {
    "english": re.compile(r"\b(?:interest\w*|enjoy\w*|curios\w*|liked?|appeal\w*|fun|favou?rite)\b", re.I),
    "hinglish": re.compile(r"\b(?:interest\w*|dilchaspi|maza|maz[e]?|pasand|achh?[ae]|curiosity)\b", re.I),
}
PROBE_QUESTION = {
    "english": re.compile(r"^(?:what(?:, if anything,)?\s+(?:specifically\s+)?(?:part|aspect|did|caught|interested|made|kept|was)|which\s+(?:part|aspect))\b", re.I),
    "hinglish": re.compile(r"\b(?:kya|kis|kaunsa|kaunsi|kaunse)\b", re.I),
}
PROBE_FILLER = set(
    "about after before anything did find found feel first from have interesting liked part "
    "that their there they thing this tracking tracked doing was were what when which while with would "
    "your aap aapko agar apne baad bata bike check chiz cheez dilchaspi kaam karne kaunsa kaunsi kiya kuch "
    "kya laga liye mein pasand raha rahe usmein wali".split()
)


def topical_words(text: str) -> set[str]:
    words = (word for word in re.findall(r"\b[a-z]{4,}\b", text.casefold()) if word not in PROBE_FILLER)
    return {word[:-1] if word.endswith("s") and not word.endswith("ss") else word for word in words}


def address_student(text: str) -> str:
    return re.sub(r"\b(?:my|our)\b", "your", text, flags=re.I)


def fallback_question(intent: dict, language: str, latest: str, observations) -> str:
    external = any(item.kind == "external_preference" for item in observations)
    hinglish = language == "hinglish"
    if intent["kind"] == "explore_example":
        example = address_student(intent["example"])
        if hinglish:
            return (f"“{example}” ke baare mein, aapko zyada curiosity thi ki kaise kaam karta hai, "
                    "ya mann hua ki aap bhi kuch aisa try karein? Sirf dekhna achha laga ho, woh bhi theek hai.")
        example = re.sub(r"^(?:a|an)\s+", "the ", example, flags=re.I)
        return (f"Thinking about {example}, were you more curious about how it worked, "
                "or did it make you want to try something similar yourself? Just enjoying it is fine too.")
    if intent["kind"] == "scaffold_interest":
        focus = intent["detail"] or intent["topic"]
        quoted = intent.get("quote_detail", False)
        if not quoted:
            focus = address_student(focus)
        if hinglish:
            intro = f'Aapne kaha, “{focus}”. ' if quoted else f"{focus.capitalize()} ke baare mein sochiye. "
            return (intro + "Aapko dekhna achha laga, "
                    "samajhna tha kaise hua, ya aisa kuch khud try karna chahenge? Pata nahi bhi theek hai.")
        intro = f'You mentioned “{focus}”. Did' if quoted else f"Thinking about {focus}, did"
        return (f"{intro} you mostly enjoy seeing what happened, feel curious about how it worked, "
                "or want to try something similar? It's fine if you're unsure.")
    if intent["kind"] == "park_topic":
        if hinglish:
            return "Koi baat nahi. Jab mann ho, kisi aur activity ke baare mein bata sakte hain."
        return "That's okay. You don't have to put it into words. We can leave it there for now."
    if intent["kind"] == "access_or_pressure" and not external:
        return ("Koi aisi activity bata sakte hain jo aapko try karne ka mauka mila aur jo aapko pasand aayi?" if hinglish
                else "What is an activity you've had a chance to try that you enjoyed?")
    if intent["kind"] in {"recall", "rephrase"}:
        topic, detail = intent["topic"], intent["detail"]
        if intent["kind"] == "recall":
            parts = []
            if topic:
                parts.append(f"Aapne {topic} ka zikr kiya tha." if hinglish else f"You mentioned {topic}.")
            if detail:
                parts.append(f'Aapne kaha, “{detail}”.' if hinglish else f'You said, “{detail}”.')
            if parts:
                return " ".join(parts)
            return "Abhi aapne koi activity share nahi ki." if hinglish else "You haven't described an activity yet."
        if detail:
            return (f'“{detail}” mein aapko sabse interesting kya laga?' if hinglish
                    else f'You said, “{detail}”. What were you most curious about?')
        if topic:
            return f"{topic.capitalize()} mein aapko kya pasand tha?" if hinglish else f"What did you enjoy about {topic}?"
        return ("Koi ek activity bata sakte hain jismein aapko maza aaya?" if hinglish
                else "Think of one activity you enjoyed recently. What did you enjoy about it?")
    if intent["kind"] == "repair_reference":
        topic = intent["topic"]
        if topic:
            if hinglish:
                return f"Aapne {topic} ka zikr kiya tha. Usmein aapko kya pasand tha?"
            if latest.rstrip().endswith("?"):
                return f"You mentioned {topic}. What stood out to you?"
            return f"You're right. We can focus on {topic}. What do you remember enjoying?"
        if hinglish:
            return "Aapko activities compare karne ki zaroorat nahi hai. Aap kis ek cheez ke baare mein aur batana chahenge?"
        return "You can tell me about one activity at a time. What would you like to tell me more about?"
    if intent["kind"] == "probe_topic":
        topic = intent["topic"]
        return f"{topic.capitalize()} mein aapko kya pasand tha?" if hinglish else f"What did you enjoy about {topic}?"
    if intent["kind"] == "probe_detail":
        detail = intent["detail"]
        norm_det = detail.strip().casefold()
        if (len(norm_det.split()) <= 2
                or bool(re.match(r"^(?:yes|no|idk|ok|okay|sure|yeah|yep|maybe|not sure|haan|nahi|pata nahi|theek hai|yes it is)\b", norm_det, re.I))):
            return ("Aap school ya college mein kis tarah ke practical projects ya problems solve karna pasand karenge?" if hinglish
                    else "What is one specific subject, project, or practical problem you enjoy exploring?")
        if len(detail.split()) > 7:
            clean = re.sub(r"^(?:i\s+(?:really\s+)?(?:love|like|enjoy)\s+(?:the way\s+)?(?:how\s+)?|the\s+way\s+(?:how\s+)?|i\s+(?:can|could)\s+)", "", detail, flags=re.I).strip()
            detail = clean if clean else detail
        return (f'“{detail}” mein aapko kya interesting laga?' if hinglish
                else f'You mentioned “{detail}”. What part of that stayed with you?')
    if intent["kind"] == "probe_interest":
        anchor = intent["anchor"]
        if hinglish:
            if intent["roman_hindi"]:
                return f"Jab aapne {anchor}, tab aapko kya interesting laga, agar kuch laga?"
            return f"“{anchor}” — usmein aapko kya interesting laga, agar kuch laga?"
        if intent["roman_hindi"]:
            return f"What, if anything, interested you about the activity you described as “{anchor}”?"
        return f"When you {address_student(anchor)}, what part, if any, did you find interesting?"
    if intent["kind"] == "compare":
        first, second = intent["activities"]
        if hinglish:
            return f"Aap kis activity par zyada time dena chahenge, {first}, {second}, dono, ya koi bhi nahi?"
        return f"Which would you spend more time on, {first} or {second}, or would you choose both or neither?"
    if external:
        if hinglish:
            return "Ghar ki expectations ke saath aapki apni pasand bhi zaroori hai. Aapne khud se kaunsi activity enjoy ki hai?"
        return "It can be hard to weigh family advice alongside your own interests. What is an activity you chose for yourself and enjoyed?"
    raw_anchor = intent.get("anchor") if intent.get("anchor") is not None else latest[:90]
    anchor = re.sub(r"[\r\n?.!]+", " ", raw_anchor).strip()
    is_filler = (
        len(anchor.split()) <= 2
        or bool(re.match(r"^(?:yes|no|idk|ok|okay|sure|yeah|yep|maybe|not sure|haan|nahi|pata nahi|theek hai|yes it is)\b", anchor, re.I))
    )
    if is_filler or not anchor:
        if hinglish:
            return "Aap school ya college mein kis tarah ke practical projects ya subjects mein apna time spend karna pasand karenge?"
        return "What is one specific subject, project, or practical problem you enjoy exploring?"
    if hinglish:
        if anchor:
            if re.search(r"\b(?:program|code|coding|computer|c language|software)\b", anchor, re.I):
                return "Programming ya computers mein aapko kis tarah ke projects ya problems solve karna sabse zyada pasand hai?"
            return f"Jab aapne “{anchor}” kaha, toh usmein se kaunsi specific activity ya baat aapke dhyan mein aati hai?"
        return "Kya aap koi ek recent activity bata sakte hain jisme aapko sabse zyada maza aaya?"
    if anchor:
        if re.search(r"\b(?:program|code|coding|computer|c language|software)\b", anchor, re.I):
            return "What kind of projects or problems do you most enjoy creating or solving with programming?"
        return f"When you say “{anchor}”, what is one specific activity or part of that subject that comes to mind?"
    return "What is one activity you enjoyed recently, at school or outside it?"


def local_chat(system: str, data: dict, schema: dict, timeout: int = 65, max_tokens: int = 170, messages: list[dict] | None = None) -> dict:
    model_name = os.environ.get("CAREER_OLLAMA_MODEL") or MODEL
    chat_messages = [{"role": "system", "content": system}]
    if messages:
        for m in messages[-4:]:
            role = "user" if m.get("role") == "user" else "assistant"
            txt = m.get("text", "")
            if txt and not txt.startswith("Skipped"):
                chat_messages.append({"role": role, "content": txt})
    chat_messages.append({"role": "user", "content": json.dumps(data, ensure_ascii=False)})
    request = Request(
        f"{BASE_URL}/api/chat",
        data=json.dumps({
            "model": model_name, "stream": False, "think": False, "format": schema, "keep_alive": "15m",
            "options": {"temperature": 0.3, "num_ctx": 2048, "num_predict": max_tokens},
            "messages": chat_messages,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    if result.get("done_reason") == "length":
        raise ValueError("Model reached its output limit")
    return json.loads(result["message"]["content"])


def make_question(intent: dict, language: str, latest: str, observations, messages: list[dict] | None = None) -> tuple[str, str]:
    kind = intent["kind"]
    is_offline_test = (
        os.environ.get("CAREER_OLLAMA_URL", "").endswith("/missing-ollama")
        or os.environ.get("CAREER_OFFLINE_TEST") == "1"
    )
    if kind in {"repair_reference", "recall", "rephrase", "access_or_pressure", "scaffold_interest", "explore_example", "park_topic"}:
        return fallback_question(intent, language, latest, observations), "local-repair"
    hinglish = language == "hinglish"
    probing = kind in {"probe_interest", "probe_topic", "probe_detail"}
    instructions = INSTRUCTIONS + {
        "probe_interest": PROBE_INSTRUCTIONS, "probe_topic": TOPIC_INSTRUCTIONS, "probe_detail": DETAIL_INSTRUCTIONS,
        "compare": COMPARE_INSTRUCTIONS, "clarify": CLARIFY_INSTRUCTIONS,
        "scaffold_interest": DETAIL_INSTRUCTIONS, "explore_example": PROBE_INSTRUCTIONS,
    }.get(kind, PROBE_INSTRUCTIONS)
    if hinglish:
        instructions += " Use natural conversational Roman Hindi with familiar English words."
    past_questions = [m.get("text", "") for m in (messages or []) if m.get("role") == "assistant"]
    if past_questions:
        instructions += (
            "\nQuestions ALREADY asked previously (DO NOT REPEAT OR CLOSELY REPHRASE ANY OF THESE):\n"
            + "\n".join(f"- {q}" for q in past_questions[-4:])
            + "\nAsk a fresh, distinct diagnostic question exploring a different facet."
        )
    data = {"language": "Roman Hindi mixed with English" if hinglish else "English", "intent": kind}
    if probing or kind in {"scaffold_interest", "explore_example"}:
        if kind == "probe_interest":
            instructions += " " + PROBE_EXAMPLES[language]
        data["activity"] = intent.get("anchor") or intent.get("detail") or intent.get("topic") or intent.get("example", "")
    else:
        data.update(latest_reply=latest[:240], activities=intent.get("activities", []),
                    context=[entry.kind for entry in observations])
    try:
        result = local_chat(
            instructions,
            data,
            QUESTION_SCHEMA,
            max_tokens=220,
            messages=messages,
        )
        question = result["question"].strip()
        if question.count("?") > 1:
            parts = [p.strip() for p in question.split("?") if p.strip()]
            question = parts[-1] + "?" if parts else question
        if not question.endswith("?") or len(question) < 5 or len(question) > 400:
            raise ValueError("Invalid question")
        if kind != "compare" and question.casefold().strip(" .!?") == latest.casefold().strip(" .!?"):
            raise ValueError("Question repeated the student's reply")
        if hinglish and not HINGLISH.search(question):
            raise ValueError("Missing Hinglish wording")
        if kind != "compare" and UNSUPPORTED_CHOICE.search(question):
            raise ValueError("Unexpected comparison or reference to other activities")
        if kind in {"probe_topic", "probe_detail"} and ACTIVITY_ROLE.search(question) and not ACTIVITY_ROLE.search(data.get("activity", "")):
            raise ValueError("Question assumed how the student took part")
        if probing:
            topics = topical_words(data.get("activity", ""))
            overlap = topics & topical_words(question)
            if (not PROBE_QUESTION[language].search(question) or not PROBE_CUES[language].search(question)
                    or len(overlap) < min(3, len(topics)) or re.search(r"[.!]|\n|<think", question[:-1], re.I)
                    or re.search(r"^what did you (?:notice|learn|track|do|find out)\b", question, re.I)):
                raise ValueError("Follow-up missed the activity or interest intent")
            if language == "english" and re.search(r"\b(?:my|our)\b", data.get("activity", ""), re.I):
                question = address_student(question)
        if kind == "compare":
            first, second = (question.casefold().find(text.casefold()) for text in intent["activities"])
            if 0 <= second <= first and not messages:
                raise ValueError("Comparison changed")
            if (first < 0 or second < 0) and not messages:
                raise ValueError("Comparison changed")
        if past_questions:
            q_clean = set(re.findall(r"\w+", question.lower()))
            for prev in past_questions[-3:]:
                prev_clean = set(re.findall(r"\w+", prev.lower()))
                if len(q_clean) >= 4 and len(prev_clean) >= 4:
                    if len(q_clean & prev_clean) / max(len(q_clean), len(prev_clean)) > 0.85:
                        raise ValueError("Question too similar to a previously asked question")
        return question, "ollama"
    except (URLError, OSError, ConnectionRefusedError):
        if is_offline_test or not messages:
            return fallback_question(intent, language, latest, observations), "local-fallback"
        offline_msg = (
            "⚠️ The AI Career Counsellor engine (Ollama) is currently unreachable. "
            "Please make sure Ollama is running (`ollama run llama3.2:3b` or `ollama serve`) "
            "so the system can generate dynamic, personalized questions."
        )
        return offline_msg, "ollama-offline"
    except (ValueError, TypeError, KeyError, AttributeError):
        if is_offline_test or not messages:
            return fallback_question(intent, language, latest, observations), "local-fallback"
        return fallback_question(intent, language, latest, observations), "local-fallback"
