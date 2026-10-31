# Failure-Aware RAG: Self-Repairing Retrieval for Reliable Clinical Generation

A retrieval-augmented question-answering system that checks every answer before returning it. When a check fails, it **diagnoses why** (missing evidence, poor ranking, noisy context, position effect, unsupported claims, multi-hop gap or conflicting sources) and applies **the repair that matches that failure**. If the answer still cannot be verified, it **abstains** instead of guessing.

The project includes a web interface, every baseline and ablation from the experiment plan, failure injection for ground-truth labels, and an evaluation runner that writes the results tables.

```
retrieve -> generate -> detect --pass--> answer with citations
                          |
                          fail -> diagnose -> matched repair -> regenerate -> detect ... (max 2 loops)
                                                                   budget spent -> abstain
```

---

## 1. Quick start (5 minutes, no API key)

```bash
cd failure-aware-rag
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Open **http://localhost:8000**. Click one of the example questions. The left card is your pipeline; the right card is vanilla RAG on the same question.

This runs in **mock mode**: answers are extracted sentences, so you can test the pipeline and UI for free. For real results, connect an LLM (step 2).

Run the tests: `pytest -q`

## 2. Connect a real LLM

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Edit `.env` and pick one option:

| Option | Cost | Settings in `.env` |
|---|---|---|
| Ollama on your laptop | Free | `LLM_PROVIDER=openai`, `LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=llama3.1:8b` |
| Groq | Free tier | `LLM_PROVIDER=openai`, `LLM_BASE_URL=https://api.groq.com/openai/v1`, `LLM_API_KEY=...`, `LLM_MODEL=llama-3.1-8b-instant` |
| OpenAI | Paid | `LLM_PROVIDER=openai`, `LLM_API_KEY=sk-...`, `LLM_MODEL=gpt-4o-mini` |

Restart `python run.py`. The top-right corner of the UI shows which model is active.

## 3. Turn on the neural models (recommended for experiments)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-ml.txt
```

This adds three models, downloaded automatically on first use (about 300 MB in total):

- **Dense retrieval** (MiniLM embeddings), fused with BM25 using Reciprocal Rank Fusion.
- **Cross-encoder reranker**, which gives the relevance scores used by the detectors.
- **NLI claim checker** (DeBERTa-v3), which tests whether each answer sentence is entailed by or contradicts the passages.

Without these models the system uses lexical fallbacks: BM25 only, with term-overlap relevance and support. The fallbacks are fine for demos but too weak for the paper.

## 4. Add your own documents

Drop PDFs, Markdown or text files into `data/corpus/`, or upload them in the **Documents** tab. The index rebuilds automatically.

**Set each document's authority.** The conflict repair keeps the source with the higher authority (guideline > textbook > article > forum), so this setting matters:

- **Markdown:** use front-matter (`authority: guideline`).
- **PDF:** put the word in the file name (`nice_hypertension_guideline.pdf`).

**Remove the five demo files before real experiments.** They are simplified test texts, and one is deliberately wrong.

## 5. Run the experiments

```bash
# Main comparison on your question set, no injected failures
python -m evaluation.run_eval --dataset data/eval/demo_questions.jsonl

# Every pipeline x every injected failure (fills Tables 1-3 of the plan)
python -m evaluation.run_eval --modes all --inject all --name full_run

# Tune thresholds on the dev split only, then run the test split once
python -m evaluation.run_eval --split dev  --name dev_tuning
python -m evaluation.run_eval --split test --modes all --inject all --name final

# MIRAGE benchmark (download benchmark.json from github.com/Teddy-XiongGZ/MIRAGE into data/eval/)
python -m evaluation.run_eval --dataset mirage:medqa --mirage-path data/eval/benchmark.json --limit 200
```

Each run writes three files to `results/`:

- `<name>.json`: everything from the run.
- `<name>_summary.csv`: one row per pipeline × injection, which is your results table.
- `<name>_rows.csv`: one row per question, for error analysis.

The **Evaluate** tab runs the same thing from the browser.

Your own question file uses one JSON object per line:

```json
{"id": "q1", "question": "...", "answer_keywords": ["0.5 mg"], "gold_doc": "anaphylaxis_guideline", "counterfactual": "optional contradicting text"}
{"id": "q2", "question": "...", "options": {"A": "...", "B": "..."}, "answer": "B", "gold_doc": "..."}
{"id": "q3", "question": "...", "answerable": false}
```

- **`gold_doc`** is the file name, without its extension, of the document that holds the answer. Failure injection needs it.
- **`counterfactual`** enables the conflict injection.
- **`answerable: false`** marks a question whose correct behaviour is to abstain.

---

## 6. Folder structure

```
failure-aware-rag/
├── run.py                      # starts the web app
├── requirements.txt            # core dependencies
├── requirements-ml.txt         # optional neural models
├── .env.example                # all settings; copy to .env
├── README.md
│
├── backend/                    # the RAG system itself
│   ├── config.py               # settings and thresholds (read from .env)
│   ├── textutils.py            # tokenising, sentence splitting, term coverage
│   ├── ingest.py               # read PDF/MD/TXT, chunk, save the chunk store
│   ├── models.py               # lazy-load embedder, reranker, NLI model
│   ├── retrieval.py            # BM25 + dense hybrid search (RRF fusion)
│   ├── llm.py                  # OpenAI-compatible client + mock mode
│   ├── generator.py            # answer / query-rewrite / decomposition prompts
│   ├── verifier.py             # relevance, claim support, conflict scoring
│   ├── detectors.py            # turns one answer into failure signals
│   ├── diagnosis.py            # signals -> one failure type (the taxonomy)
│   ├── repairs.py              # one repair action per failure type
│   ├── pipeline.py             # the loop + all baselines and ablations
│   └── api.py                  # FastAPI endpoints, serves the UI
│
├── frontend/                   # web interface (no build step)
│   ├── index.html              # page layout: Ask, Documents, Evaluate tabs
│   ├── styles.css              # visual design, light and dark mode
│   └── app.js                  # calls the API and renders results
│
├── evaluation/                 # experiments from the plan
│   ├── datasets.py             # load JSONL or MIRAGE; dev/test split
│   ├── inject.py               # inject known failures (ground-truth labels)
│   ├── metrics.py              # accuracy, hallucination, abstention, AURC, McNemar
│   └── run_eval.py             # command-line experiment runner
│
├── data/
│   ├── corpus/                 # documents the system answers from
│   ├── eval/demo_questions.jsonl
│   └── index/                  # generated: chunks + embeddings cache
├── results/                    # generated: experiment outputs
└── tests/test_pipeline.py      # pytest checks for the main behaviours
```

## 7. Pipelines you can compare

| Mode | Plan ID | What it does |
|---|---|---|
| `failure_aware` | Ours | Detect, diagnose, matched repair, abstain |
| `vanilla` | B1 | Single-pass RAG |
| `rerank` | B2 | Retrieval + reranker |
| `crag_lite` | B4* | Simplified CRAG: corrects retrieval based on retrieval score only |
| `llm_only` | B0 | No retrieval |
| `detect_abstain` | A1 | Detects failures, abstains, no repair |
| `generic` | A2 | Any failure leads to re-retrieval (no diagnosis) |
| `random` | A3 | Random repair, same budget; the key test of diagnosis value |
| `oracle` | A4 | Uses the true injected label (evaluation only) |
| `no_abstain` | A5 | Repairs but never abstains |

\* `crag_lite` is a simplified re-implementation. For the paper, also run the official Self-RAG / Self-BioRAG, CRAG and i-MedRAG code on the same corpus and questions.

## 8. Known limits

- **Mock mode** is extractive, so it cannot hallucinate. Real hallucination rates only appear with a real LLM.
- **Mock lost-in-the-middle:** the mock reads only the first 3 passages and the last one, which imitates the position effect.
- **Lexical conflict detection** only catches differing quantities with the same unit. Install the NLI model for real contradiction detection.
- **Missing-evidence injection** hides the gold passage from the first retrieval only, so re-querying can find it again. For truly absent content, use `answerable: false` questions.
- **Thresholds** in `.env` are starting points. Tune them on the dev split and freeze them before the test run.
