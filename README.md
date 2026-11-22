# AI Career Counsellor

An NLP course project for Indian students exploring subject choices after Class 10 and degree or professional routes after Class 12. The planned system uses an English/Hinglish evidence extractor and Holland’s RIASEC framework to guide a conversation. Local Ollama will generate contextual follow-up questions.

The starter contains a FastAPI health endpoint and a Streamlit entry point. The [implementation plan](IMPLEMENTATION_PLAN.md) describes the counselling pipeline, datasets, and evaluation.

## Run locally on Windows

Use Python 3.11 and PowerShell. Create a virtual environment and install the pinned starter dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the API and UI in separate terminals from the project root.

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/main.py --server.address 127.0.0.1
```

The API health endpoint is at `http://127.0.0.1:8000/health`. Open the URL Streamlit prints to view the UI. Ollama is used when question generation is implemented.
