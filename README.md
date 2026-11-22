# AI Career Counsellor

An NLP course project for Indian students exploring subject choices after Class 10 and degree or professional routes after Class 12. The planned system uses an English/Hinglish evidence extractor and Holland’s RIASEC framework to guide a conversation. Local Ollama will generate contextual follow-up questions.

The local prototype covers Class 10 and Class 12, with 37 pathways, English and Hinglish evidence extraction, dynamic questions, and a report that links recommendations to student answers. The [implementation plan](IMPLEMENTATION_PLAN.md) describes the research design and future validation.

## Run locally on Windows

Use Python 3.11 and PowerShell. Create a virtual environment if needed and install the pinned dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3:1.7b
```

Start the API and UI in separate terminals from the project root.

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/main.py --server.address 127.0.0.1
```

Open the URL Streamlit prints. The backend expects Ollama at `http://127.0.0.1:11434`. If generation is unavailable or its response fails validation, the same policy composes a contextual question locally. The API exposes `/health` and interactive documentation at `http://127.0.0.1:8000/docs`.

You can override the backend's Ollama endpoint with `CAREER_OLLAMA_URL`, its model with `CAREER_OLLAMA_MODEL`, and the SQLite location with `CAREER_DB_PATH`. The Streamlit frontend reads `CAREER_API_URL`. Sessions live under the ignored `artifacts/` directory by default. They expire after 24 hours of inactivity and are purged when a new session is created. The UI also supports immediate deletion. Run both services on loopback for the course demo.

## Build phases implemented

1. The catalog in `catalog/pathways.json` defines stages, subjects, specializations, activity examples, and official reference links. It includes scoped checks for two admissions examples. Other routes give verification steps based on the student's context.
2. The CPU NLP pipeline fits word and character TF-IDF plus logistic regression from `data/evidence_seed.jsonl`. It keeps original spans and uses attribution and negation cues. Sparse activity retrieval and the evidence ledger inform ranking. Model fitting occurs when first needed and uses the authored dataset. There is no Colab or LLM training in the prototype.
3. The question policy compares candidate intents by expected reduction in decision regret. Ollama writes the selected question. The evidence ledger distinguishes family preference, exposure, interests, and activity evidence. Removing a disputed answer rebuilds the report from the remaining evidence.

## Evaluate

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m backend.evaluate
```

Add `--ollama` to the evaluation command to run local Qwen as a prompt-only extraction baseline. The default offline run compares rules, word features, character features, combined features, activity retrieval, and two question policies. The 28 short replies and eight simulated profiles are authored development fixtures. A counsellor-reviewed holdout and a school pilot remain research milestones. Admissions checks cover the recorded conditions and cycle shown in the report. Verify the current programme requirements with the linked authority.

On this development set, keyword rules and the combined extractor both scored 0.889 evidence-label F1. The current prompt-only Qwen extractor scored 0.105 after removing labels attached to invalid quotes. Both question policies placed an acceptable route in the top two for seven of eight simulated profiles. These fixtures are small and were used during development.
