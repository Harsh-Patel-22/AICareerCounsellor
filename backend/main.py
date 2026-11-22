from fastapi import FastAPI

app = FastAPI(title="AI Career Counsellor")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
