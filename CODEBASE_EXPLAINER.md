# 📖 Codebase Explainer & Architecture Reference
**Project Title:** AI-Based Conversational Career Counsellor for Indian Students  
**Subject:** Natural Language Processing (NLP Hackathon / End-Semester Exam)  
**Purpose:** Quick 1-line reference for every file, class, and function in the codebase for viva defense and code walkthrough.

---

## 🏗️ System Architecture at a Glance

The project follows a modular, decoupled architecture:
1. **`backend/nlp.py`**: Linguistic Information Extraction, transliteration, clause segmentation, and Holland RIASEC classification.
2. **`backend/catalog.py`**: Academic pathway definitions, TF-IDF Vector Space Model, and prerequisite rule checks.
3. **`backend/dialogue.py`**: Bayesian belief tracking, value-of-information comparison policy, and turn orchestration.
4. **`backend/ollama.py`**: Local Llama-3.2 integration, JSON schema enforcement, and "Disha" counsellor system prompt.
5. **`backend/main.py`**: FastAPI REST API providing atomic, optimistic-concurrency endpoints.
6. **`backend/store.py`**: SQLite database layer handling concurrent writes, idempotency, and revision conflict detection.
7. **`backend/evaluate.py`**: Precision/recall evaluation scripts against expert-authored student benchmarks.
8. **`app/main.py`**: Streamlit interactive user interface, live report cards, and the NLP Explainability Inspector.

---

## 1. `backend/nlp.py` — NLP & Linguistic Extraction Core
> **File Purpose:** Implements all core Natural Language Processing logic: text normalization, Hinglish transliteration, clause segmentation, negation handling, semantic intent attribution, subword n-gram vectorization, and multi-label Holland RIASEC classification.

### Classes:
* **`Observation`**: Data structure representing an extracted linguistic span, its character boundaries, attribution kind, mapped Holland codes, and polarity (+1 for interest, -1 for dislike).
* **`EvidenceExtractor`**: Orchestrates the multi-stage NLP pipeline to parse raw student utterances into structured psychological and career observations.

### Functions:
* **`normalize(text: str) -> str`**: Performs Unicode NFKC normalization, strips punctuation noise, case-folds text, and standardizes informal spacing.
* **`span_candidates(text: str)`**: Splits compound sentences into distinct candidate clauses based on syntactic conjunctions, semicolons, and commas.
* **`activity_to_probe(text: str) -> dict | None`**: Detects if an utterance describes an actionable activity or experience suitable for follow-up probing.
* **`described_example(text: str) -> str | None`**: Extracts concrete project or object nouns (e.g. "bluetooth car", "water filter") from conversational stories.
* **`describes_someone_elses_action(text: str) -> bool`**: Identifies third-person observer framing (e.g., "my classmate made a car") to avoid falsely attributing other people's actions to the student.
* **`stated_topic(text: str) -> str`**: Identifies explicit academic subject keywords (e.g., "physics", "computers", "accounts") mentioned directly in the user utterance.
* **`vectorizers() -> FeatureUnion`**: Builds a scikit-learn feature union combining word n-grams (1, 2) and subword character n-grams (3, 5) to robustly handle Indian English spelling variations and Hinglish morphology.
* **`get_extractor() -> EvidenceExtractor`**: Returns a cached, thread-safe singleton instance of the `EvidenceExtractor`.
* **`EvidenceExtractor.__init__(self)`**: Initializes the TF-IDF feature pipeline and trains the One-vs-Rest multi-label logistic regression RIASEC classifier on curated curriculum activity spans.
* **`EvidenceExtractor.extract(self, text: str) -> list[Observation]`**: End-to-end pipeline method that segments text, checks for parental framing (`external_preference`) and lack of resources (`limited_exposure`), and classifies remaining clauses into Holland RIASEC codes.
* **`EvidenceExtractor.predict(self, text: str) -> tuple[str, ...]`**: Runs the trained subword classifier on a single text span and returns the predicted Holland RIASEC code tuple.
* **`EvidenceExtractor.answer_to_interest_probe(self, text: str) -> list[Observation]`**: Fallback extractor that recovers implicit positive or negative sentiment when a student answers a direct interest follow-up question.

---

## 2. `backend/catalog.py` — Academic Pathways & Vector Space Model
> **File Purpose:** Loads the curriculum catalog (CBSE Class 10 streams & UGC/JoSAA Class 12 college degrees), constructs the TF-IDF Vector Space Model, computes Cosine Similarity, and evaluates prerequisite eligibility rules.

### Classes:
* **`Pathway`**: Immutable dataclass defining a study route (ID, stage 10 or 12, stream, degree title, focus description, Holland codes, representative activities, and institutional source).
* **`Catalog`**: Manages the complete repository of academic pathways, pre-computes TF-IDF vector embeddings for all activities, and executes semantic retrieval.

### Functions:
* **`evaluate_requirement(expression: dict, facts: dict[str, bool]) -> bool | None`**: Recursively evaluates 3-valued boolean logic (`True`, `False`, or `None` for unconfirmed) for admission and subject prerequisite trees (`all_of`, `any_of`, `fact`).
* **`get_catalog() -> Catalog`**: Returns an LRU-cached singleton instance of the `Catalog` to avoid re-reading JSON files and re-fitting TF-IDF matrices on every request.
* **`Catalog.__init__(self)`**: Loads `catalog/pathways.json`, creates lookup tables by ID, extracts activity phrases, and builds the scikit-learn TF-IDF vector space matrix.
* **`Catalog.retrieve(self, evidence: list[str], stage: int, limit: int = 5) -> list[tuple[str, str, float]]`**: Computes mathematical Cosine Similarity between student evidence vectors and catalog pathway activity vectors, returning top-k matching pathways for the student's academic stage.
* **`Catalog.feasibility(self, pathway: Pathway, school_subjects: list[str]) -> dict`**: Validates whether a Class 10 student's local school actually offers the required subject combination (e.g. verifying Economics or Mathematics availability).
* **`Catalog.checks(self, pathway: Pathway, facts: dict[str, dict[str, bool]]) -> list[dict]`**: Evaluates degree-level statutory conditions (e.g. CUET Mathematics requirement for Delhi University B.A. Economics, or ICAI Class XII registration period).

---

## 3. `backend/dialogue.py` — Bayesian Belief Tracking & Dialogue Policy
> **File Purpose:** Tracks student psychological beliefs using Bayesian probability distributions, calculates Value of Information (VOI) to decide the next best question, computes multi-factor recommendation match percentages, and manages conversational state transitions.

### Functions:
* **`is_conversation_repair(text: str) -> bool`**: Detects student meta-conversation utterances such as confusion ("What do you mean?"), recall ("What did I say?"), or reference corrections.
* **`lacks_detail(text: str) -> bool`**: Identifies uninformative or monosyllabic replies ("nothing", "just fun", "idk") that require conversational scaffolding.
* **`brief_examples(text: str) -> str`**: Isolates concrete activity nouns from short mixed student lists while discarding generic emotional filler words.
* **`detail_excerpt(text: str) -> str`**: Extracts the most informative, substantive clause from a user turn while discarding filler and affirmative affirmations.
* **`recent_context(messages: list[dict]) -> dict`**: Scans the conversation history backwards to retrieve the most recent active academic topic or activity anchor.
* **`new_session(stage: int, language: str, school_subjects: list[str]) -> dict`**: Initializes a fresh session state dictionary with stage-aware greetings (Class 10 stream discovery vs Class 12 degree discovery) and empty evidence ledgers.
* **`active_evidence(state: dict) -> list[dict]`**: Filters the session ledger to return only user-verified, non-retracted evidence items.
* **`profile(state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]`**: Computes the positive, negative, and demonstrated skill frequency vectors across the 6 Holland RIASEC codes from active evidence.
* **`rank_paths(state: dict, draws: int = 256)`**: Runs a Monte Carlo simulation drawing from Beta distributions across RIASEC dimensions, combining interest fit (60%), demonstrated skills (22%), and catalog cosine similarity (18%) to rank pathways.
* **`decision_risk(fit: np.ndarray) -> np.ndarray`**: Computes expected regret across Monte Carlo draws for choosing a lead recommendation versus offering a two-pathway comparison.
* **`comparison_value(interest: np.ndarray, fit: np.ndarray, first: str, second: str) -> float`**: Evaluates the Information Gain (Value of Information) of comparing two specific Holland codes to determine which question best reduces recommendation uncertainty.
* **`disagreement_value(interest: np.ndarray, fit: np.ndarray, first: str, second: str, paths: list) -> float`**: Computes policy value for exploring disputed borderline pathways when recommendation confidence is tied.
* **`best_activity(path, texts: list[str], catalog, used: set[str] | None = None) -> str`**: Selects the most pedagogically relevant activity string from a pathway to ground comparison questions without repeating previously discussed activities.
* **`select_intent(state: dict, current: list[Observation], latest: str, policy: str = "decision") -> dict`**: Dialogue policy core that determines the next conversational intent (`compare`, `clarify`, `probe_interest`, `explore_example`, `access_or_pressure`, etc.).
* **`comparison_answer(text: str, intent: dict | None) -> list[Observation]`**: Maps student answers to comparison questions ("first", "second", "both", "neither") back into corresponding Holland code observations.
* **`report(state: dict) -> dict`**: Generates the final recommendation report, calculating overall match percentages, multi-factor breakdowns (interest fit, skills fit, activity alignment), feasibility notices, and statutory prerequisite checks.
* **`add_turn(state: dict, text: str, skip: bool, render, policy: str = "decision") -> dict`**: Main dialogue transition orchestrator that records user text, runs NLP extraction, updates Bayesian beliefs, recalculates the report, triggers question synthesis, and checks stop conditions.

---

## 4. `backend/ollama.py` — Dynamic Question Synthesis & LLM Guardrails
> **File Purpose:** Manages communication with the local Ollama daemon (`llama3.2:3b`), injects the "Disha" counsellor persona, enforces strict JSON schemas, prevents SWE/job interview hallucinations, and provides graceful deterministic fallbacks.

### Functions:
* **`resolve_model() -> str`**: Auto-detects available local Ollama models via `http://127.0.0.1:11434/api/tags`, defaulting to `llama3.2:3b`.
* **`topical_words(text: str) -> set[str]`**: Strips stop words and filler terms to isolate the core substantive topical words in an utterance.
* **`address_student(text: str) -> str`**: Converts first-person pronouns ("my uncle's shop", "our project") into student-facing second-person phrasing ("your uncle's shop", "your project").
* **`fallback_question(intent: dict, language: str, latest: str, observations) -> str`**: Generates an intelligent, deterministic fallback question when Ollama is offline, maintaining strict grounding and zero cliché phrasing.
* **`local_chat(system: str, data: dict, schema: dict, timeout: int = 65, max_tokens: int = 170, messages: list[dict] | None = None) -> dict`**: Dispatches HTTP POST requests to Ollama's `/api/chat` with JSON schema constraints (`format: json`), zero thinking mode, and contextual multi-turn message history.
* **`make_question(intent: dict, language: str, latest: str, observations, messages: list[dict] | None = None) -> tuple[str, str]`**: Top-level question generator that builds the "Disha" prompt, incorporates academic stage context (Class 10 streams vs Class 12 degrees), queries Ollama, validates output format, and returns the question alongside its engine tag (`ollama` or `local-fallback`).

---

## 5. `backend/main.py` — FastAPI REST API Server
> **File Purpose:** Exposes the backend as a high-performance RESTful web service, handling request validation via Pydantic schemas, session lifecycle, and concurrency control.

### Classes:
* **`SessionInput`**: Pydantic schema for creating a new session (stage 10/12, language English/Hinglish, offered school subjects).
* **`TurnInput`**: Pydantic schema for conversational turns (UUID request ID, revision number, user text, skip flag, finish flag).
* **`RevisionInput`**: Pydantic schema enforcing revision numbers to prevent lost updates or stale writes.
* **`AdmissionInput`**: Pydantic schema for updating institutional admission check facts.

### Functions:
* **`sessions() -> Sessions`**: Returns the singleton SQLite session store instance.
* **`require_session(session_id: str) -> dict`**: Retrieves a session from the store or raises an HTTP 404 error if missing or expired.
* **`save_state(state: dict, revision: int, request_id: str | None = None) -> dict`**: Persists updated session state to the SQLite database with revision incrementing and conflict detection.
* **`health() -> dict`**: Health-check endpoint (`GET /health`) returning server status and catalog pathway counts.
* **`start_session(data: SessionInput) -> tuple[int, dict]`**: Endpoint (`POST /sessions`) that initializes a new counselling session and returns HTTP 201 Created.
* **`get_session(session_id: str) -> dict`**: Endpoint (`GET /sessions/{id}`) returning current session messages, ledger evidence, and recommendations.
* **`turn(session_id: str, data: TurnInput) -> dict`**: Endpoint (`POST /sessions/{id}/turns`) executing a full conversational turn with deduplication and concurrency protection.
* **`recommendations(session_id: str) -> dict`**: Endpoint (`GET /sessions/{id}/recommendations`) returning the latest computed recommendation report.
* **`retract_evidence(session_id: str, evidence_id: str, data: RevisionInput) -> dict`**: Endpoint (`PATCH /sessions/{id}/evidence/{evidence_id}`) allowing students to remove an incorrect or unwanted statement from their assessment.
* **`admission_facts(session_id: str, rule_id: str, data: AdmissionInput) -> dict`**: Endpoint (`PATCH /sessions/{id}/admissions/{rule_id}`) updating eligibility facts for institutional prerequisites.
* **`delete(session_id: str) -> tuple[int, None]`**: Endpoint (`DELETE /sessions/{id}`) purging session data from the database for student privacy (HTTP 204 No Content).

---

## 6. `backend/store.py` — SQLite Concurrency & Persistence Layer
> **File Purpose:** Implements thread-safe SQLite persistence with optimistic concurrency control, automated table migration, and idempotent request deduplication.

### Classes:
* **`Conflict`**: Exception raised when an incoming write has a stale revision number (HTTP 409 Conflict).
* **`Sessions`**: SQLite database client managing ACID storage, state serialization/deserialization, and active transactions.

### Functions:
* **`Sessions.__init__(self, db_path: Path | str | None = None)`**: Initializes database connection and creates tables (`sessions`, `requests`) with WAL mode enabled.
* **`Sessions.connect(self) -> sqlite3.Connection`**: Returns a SQLite connection with timeout and dictionary row factories.
* **`Sessions.now(self) -> str`**: Generates ISO-8601 UTC timestamp strings.
* **`Sessions.create(self, state: dict) -> dict`**: Inserts a new session record into the SQLite database.
* **`Sessions.get(self, session_id: str) -> dict | None`**: Reads and JSON-deserializes a session state by ID.
* **`Sessions.has_request(self, session_id: str, request_id: str) -> dict | None`**: Checks if an incoming turn request ID has already been executed to prevent duplicate submissions on network retries.
* **`Sessions.update(self, state: dict, expected_revision: int, request_id: str | None = None) -> dict`**: Atomically updates session state if and only if the revision matches `expected_revision`, raising `Conflict` on race conditions.
* **`Sessions.delete(self, session_id: str) -> bool`**: Permanently removes a session and its associated request log from the database.

---

## 7. `backend/evaluate.py` — Benchmark & Metric Evaluation
> **File Purpose:** Evaluates NLP extraction accuracy, retrieval quality, and dialogue policy efficiency against gold-standard annotated student conversation benchmarks.

### Functions:
* **`render(intent, language, latest, observations, messages=None)`**: Mock renderer used to isolate offline evaluation without invoking Ollama.
* **`counts(true_items, pred_items) -> tuple[int, int, int]`**: Calculates true positives, false positives, and false negatives between predicted and ground-truth sets.
* **`metrics(tp: int, fp: int, fn: int) -> dict`**: Computes mathematical Precision, Recall, and F1-score.
* **`evaluate_evidence(examples: list) -> dict`**: Runs extraction across gold-standard cases and reports macro/micro Precision, Recall, and F1 for Holland RIASEC code assignments.
* **`evaluate_retrieval(examples: list) -> dict`**: Evaluates Top-1 and Top-3 accuracy of TF-IDF Cosine Similarity catalog matching against expert-recommended pathways.
* **`evaluate_ollama(examples: list) -> dict`**: Evaluates extraction accuracy when using prompt-only LLM baseline without deterministic NLP rules.
* **`evaluate_policy(examples: list) -> dict`**: Evaluates dialogue turn efficiency, verifying that the system reaches stable recommendations within target turn limits.
* **`main()`**: CLI entry point executing the benchmark test suites and printing tabular evaluation metrics.

---

## 8. `app/main.py` — Streamlit Frontend & Explainability Dashboard
> **File Purpose:** Interactive web interface featuring a modern chat stream, live recommendation cards with multi-factor match bars, and an educational "NLP Pipeline & Explainability" dashboard for faculty examination.

### Functions:
* **`call(method: str, path: str, body: dict | None = None)`**: Low-level HTTP utility making REST calls to the FastAPI backend with timeout and error handling.
* **`perform(method: str, path: str, body: dict | None = None) -> bool`**: Executes API calls and updates Streamlit session state, managing user-facing error banners.
* **`send_turn(state: dict, text: str, skip: bool = False, finish: bool = False) -> None`**: Dispatches a conversational turn or finalize action to the backend and triggers a reactive UI rerun.
* **`requirement_check(state: dict, check: dict) -> None`**: Renders an interactive form allowing students to verify institutional admission criteria (e.g. CUET Math papers).
* **`show_report(state: dict) -> None`**: Renders the complete recommendation report with degree titles, match percentages, multi-factor progress bars, student evidence quotes, feasibility notices, and statutory links.
* **`show_nlp_inspector(state: dict) -> None`**: Renders the dedicated NLP dashboard: end-to-end architecture flowchart, live Holland RIASEC bar chart, linguistic clause attribution table with badges, cosine similarity formulas, and the downloadable `.json` audit dossier.
* **`sidebar(state: dict) -> None`**: Renders the application sidebar with conversation details, the "Start New Conversation" button, evidence retraction selector, and session deletion.

---

## 9. `catalog/pathways.json` — Curriculum Data Dictionary
> **File Purpose:** Machine-readable knowledge base encoding official CBSE Class 10 streams (Science PCM, PCB, PCMB, Commerce, Humanities) and UGC/JoSAA Class 12 college degrees (B.Tech CSE, BCA, B.Tech Mechanical, B.Sc Physics, CA Foundation, BBA, etc.) with associated RIASEC codes, representative curriculum tasks, and regulatory admission prerequisites.

---

## 10. `tests/test_counselling.py` & `tests/http_scenarios.py` — Automated Test Suites
> **File Purpose:** Contains the **92 automated tests** verifying 100% system reliability:
* **`test_counselling.py`** (22 Unit Tests): Verifies clause segmentation, Hinglish transliteration, parental pressure filtering, access-constraint neutrality, negative polarity, and session lifecycle.
* **`http_scenarios.py`** (70 Integration Tests): Runs an isolated server and verifies end-to-end multi-turn student personas across Class 10 and Class 12, regression suites, edge cases, and concurrent access.
