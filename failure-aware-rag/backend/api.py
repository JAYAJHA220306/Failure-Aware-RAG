"""FastAPI server: JSON API for the pipeline + serves the web UI from /frontend."""
from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import models
from .config import ROOT, settings
from .ingest import SUPPORTED, build_store
from .pipeline import MODES, Pipeline
from .retrieval import Retriever

app = FastAPI(title="Failure-Aware RAG", version="1.0")
FRONTEND = ROOT / "frontend"

_lock = threading.Lock()
_state: dict = {"pipeline": None}
_jobs: dict[str, dict] = {}


def pipeline() -> Pipeline:
    with _lock:
        if _state["pipeline"] is None:
            _state["pipeline"] = Pipeline()
        return _state["pipeline"]


def rebuild() -> Pipeline:
    chunks = build_store()
    with _lock:
        _state["pipeline"] = Pipeline(Retriever(chunks))
        return _state["pipeline"]


# ----------------------------------------------------------------- schemas
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    mode: str = "failure_aware"
    options: dict[str, str] | None = None
    compare: list[str] = Field(default_factory=list, description="extra modes to run side by side")


class EvalRequest(BaseModel):
    dataset: str = "demo"
    modes: list[str] = Field(default_factory=lambda: ["vanilla", "failure_aware", "random"])
    injections: list[str] = Field(default_factory=lambda: ["none"])
    limit: int | None = None
    mirage_path: str | None = None


# ------------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    p = pipeline()
    return {"ok": True, **p.info(), "models": models.status()}


@app.post("/api/ask")
def ask(req: AskRequest):
    p = pipeline()
    runs = []
    for mode in [req.mode, *[m for m in req.compare if m != req.mode]]:
        if mode not in MODES:
            raise HTTPException(400, f"Unknown mode '{mode}'")
        try:
            runs.append(p.run(req.question, mode, req.options or None))
        except Exception as exc:  # surface LLM / config errors to the UI
            raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc
    return {"runs": runs}


@app.get("/api/documents")
def documents():
    p = pipeline()
    docs: dict[str, dict] = {}
    for c in p.retriever.chunks:
        d = docs.setdefault(c.doc_id, {"doc_id": c.doc_id, "title": c.title, "authority": c.authority, "chunks": 0, "words": 0})
        d["chunks"] += 1
        d["words"] += len(c.text.split())
    for d in docs.values():
        match = next((f for f in settings.corpus_dir.rglob(f"{d['doc_id']}.*")), None)
        d["file"] = match.name if match else None
    return sorted(docs.values(), key=lambda d: d["title"].lower())


@app.post("/api/documents")
async def upload(files: list[UploadFile] = File(...)):
    saved = []
    settings.corpus_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = Path(f.filename or "").name
        if Path(name).suffix.lower() not in SUPPORTED:
            raise HTTPException(400, f"{name}: only {', '.join(sorted(SUPPORTED))} files are supported")
        with (settings.corpus_dir / name).open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(name)
    p = rebuild()
    return {"saved": saved, "chunks": len(p.retriever.chunks)}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    matches = [f for f in settings.corpus_dir.rglob("*") if f.is_file() and f.stem == doc_id]
    if not matches:
        raise HTTPException(404, "Document not found")
    for f in matches:
        f.unlink()
    p = rebuild()
    return {"deleted": doc_id, "chunks": len(p.retriever.chunks)}


@app.post("/api/reindex")
def reindex():
    p = rebuild()
    return {"chunks": len(p.retriever.chunks)}


@app.post("/api/evaluate")
def start_evaluation(req: EvalRequest):
    from evaluation.datasets import load_dataset
    from evaluation.run_eval import evaluate, save

    try:
        items = load_dataset(req.dataset, req.mirage_path, req.limit, settings.eval_dir)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"status": "running", "done": 0, "total": 1, "started": time.time()}

    def work():
        try:
            report = evaluate(pipeline(), items, req.modes, req.injections,
                              lambda d, t: _jobs[job_id].update(done=d, total=t))
            report["config"] = req.model_dump()
            name = time.strftime("ui_%Y%m%d_%H%M%S")
            save(report, name)
            _jobs[job_id].update(status="done", summary=report["summary"], significance=report["significance"], file=f"{name}.json")
        except Exception as exc:
            _jobs[job_id].update(status="error", error=f"{type(exc).__name__}: {exc}")

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/evaluate/{job_id}")
def evaluation_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Unknown job")
    return _jobs[job_id]


@app.get("/api/results")
def results():
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for f in sorted(settings.results_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            out.append({"file": f.name, "summary": data.get("summary", []), "config": data.get("config", {})})
        except Exception:
            continue
    return out


@app.get("/api/results/{name}")
def download_result(name: str):
    path = (settings.results_dir / Path(name).name)
    if not path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(path)


# ------------------------------------------------------------------ frontend
@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
