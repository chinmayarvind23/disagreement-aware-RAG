Disagreement-Aware RAG (Answer/Abstain)
One-liner: A retrieval-augmented QA system that predicts disagreement risk and abstains when answers are likely contentious or weakly grounded—improving reliability at a chosen coverage level. Inspired by Kang et al., Everyone’s Voice Matters (AAAI 2023).

Why this matters
Large language models can sound confident while being wrong—especially on subjective or under-specified questions. Following Kang et al.’s framing that disagreement is signal, not noise, this project treats “people would disagree here” as a first-class prediction target. We then enforce an answer/abstain policy: answer when risk is low and evidence support is strong; abstain otherwise. That yields a clear coverage–risk trade-off you can tune to your application.

Kang, Dongyeop et al., 2023 — Everyone’s Voice Matters: Quantifying Annotation Disagreement Using Demographic Information. AAAI. DOI: https://doi.org/10.1609/aaai.v37i12.26698

What the app does
Ask & Cite: You type a question. The system retrieves documents, synthesizes an answer with citations, computes disagreement risk, and either answers or abstains.

Metrics: The UI plots Coverage vs. Hallucination as you vary the abstention threshold τ (tau). The backend also reports an ROC-AUC against a lightweight entailment-based auditor.

Decision rule (simple):

scss
Copy
Edit
Answer  iff  (p_disagree < τ)  AND  (overlap ≥ min_overlap)  AND  (sc_var ≤ max_sc)
Else → Abstain
p_disagree: predicted probability that humans (or annotators) would disagree

overlap: ROUGE-L–style semantic overlap between answer and retrieved evidence

sc_var: self-consistency variance across k re-sampled answers (higher = less stable)

How it works (end-to-end)
Retrieve top-k passages from a small indexed corpus (hybrid BM25+vectors via LlamaIndex).

Generate an answer and k temperature-varied re-samples (e.g., T ≈ 0.49, 0.60, 0.70, 0.80, 0.91) to measure self-consistency.

Compute features:

sc_var — variance across the k sampled answers

overlap — ROUGE-L–like overlap between the final answer and concatenated evidence

entropy_proxy — light uncertainty proxy from the answer text

Predict disagreement risk with a tiny logistic regression head → p_disagree.

Gate with the rule above → answer or abstain.

Evaluate (offline): Use a zero-shot NLI auditor (facebook/bart-large-mnli) to score entailment of the answer against the evidence. If best-entailment < threshold, count it as a hallucination.

Model card: https://huggingface.co/facebook/bart-large-mnli

BART paper: Lewis et al., 2020 — BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.

Results (toy slice; reproducible)
From scripts/evals.py on a small test set (30 questions):

τ = 0.55 → coverage ≈ 96.7% with hallucination ≈ 3.4%

τ = 0.60–0.65 → coverage ≈ 100% with hallucination ≈ 3.3%

The UI renders the coverage–risk curve directly from data/coverage_curve.tsv.
Note: ROC-AUC on the proxy labels is modest (expected on tiny, synthetic data), but the policy curve is stable and useful.

Tech stack
Backend: FastAPI, Pydantic, scikit-learn, numpy, python-dotenv

RAG: LlamaIndex (retrieval & synthesis)

Eval auditor: facebook/bart-large-mnli (Transformers)

Frontend: Next.js (React), Recharts (Coverage–Risk plot)

Tooling: Poetry (or uv), .env configuration

Repository layout
bash
Copy
Edit
app/
  backend/
    main.py             # FastAPI app: /qa, /metrics, /healthz
    rag.py              # index/retrieval + answer synthesis (LlamaIndex)
    features.py         # sc_var, overlap, entropy features
    disagreement.py     # logistic head + decision rule
  frontend/
    app/page.tsx        # Next.js page (Ask & Cite + plot)
  scripts/
    evals.py            # builds coverage vs hallucination curve + AUC
    train_head.py       # optional: train the logistic head
  data/
    index/...           # cached index
    coverage_curve.tsv  # produced by evals
    test_preds.npz      # produced by evals (scores + labels)
    disagree_head.joblib# saved head (if trained)
  .env                  # runtime knobs (see below)
Run everything from app/ so Python can import backend.* modules cleanly.

Setup & run
0) Requirements
Python 3.11

Node 20+

Poetry (or uv) recommended

1) Backend install
bash
Copy
Edit
cd app
poetry install              # or: uv sync
2) Configure knobs (app/.env)
ini
Copy
Edit
# Abstention policy
HEAD_TAU=0.60            # τ (risk tolerance). Higher → answer more.
DEC_MIN_OVERLAP=0.35     # require at least this overlap to answer
DEC_MAX_SC=0.30          # require self-consistency variance ≤ this
SC_SAMPLES=5             # re-samples to estimate sc_var (k)

# Generation sampling (for sc_var)
RAG_TEMP=0.7             # base temperature; code derives a small range around it

# Evaluation (entailment auditor threshold)
HALLUC_THRESHOLD=0.35

# Frontend -> Backend (direct call; skip dev proxy)
# Set this in frontend/.env.local or your shell when running npm dev:
# NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
3) Start the backend
bash
Copy
Edit
# from app/
poetry run uvicorn backend.main:app --reload --port 8000
# health check
curl http://127.0.0.1:8000/healthz
4) Start the frontend
bash
Copy
Edit
cd app/frontend
npm install
# Either export once (Linux/macOS) or set in your shell (Windows):
# export NEXT_PUBLIC_API_BASE="http://127.0.0.1:8000"
# PowerShell:  $env:NEXT_PUBLIC_API_BASE = "http://127.0.0.1:8000"
npm run dev
# open http://localhost:3000
(If you prefer Next.js rewrites instead, keep axios pointed to /api/... and add the rewrite rules in next.config.ts.)

5) Reproduce the metrics plot
bash
Copy
Edit
# from app/
poetry run python -m scripts.evals
# writes data/coverage_curve.tsv and data/test_preds.npz
# refresh the UI to see the curve & AUC
API (brief)
POST /qa
Body

json
Copy
Edit
{ "query": "your question" }
Response

json
Copy
Edit
{
  "answer": "...",
  "sources": [{"title": "...", "text": "..."}],
  "risk": {
    "p_disagree": 0.42, "sc_var": 0.08, "overlap": 0.61, "entropy_proxy": 0.12
  },
  "decision": "answer"   // or "abstain"
}
GET /metrics
json
Copy
Edit
{
  "coverage_curve": [
    { "tau": 0.55, "coverage": 0.967, "halluc_rate": 0.034 },
    { "tau": 0.60, "coverage": 1.000, "halluc_rate": 0.033 }
  ],
  "roc_auc": 0.17
}
GET /healthz
json
Copy
Edit
{ "ok": true }
Interpreting τ (risk tolerance)
Lower τ → more conservative (answer less; abstain more; fewer mistakes).

Higher τ → more permissive (answer more; slightly higher risk).
On this toy set, τ ≈ 0.60 gave ~100% coverage at ~3–4% hallucination.

Limitations & notes
Data scale: small corpus + small eval set → the ROC-AUC can be noisy. The policy curve (coverage vs. hallucination) is the key artifact.

Proxy labels: the head is trained against LLM-based proxies (self-consistency, entailment, overlap), not human-annotator disagreement labels. That’s intentional for a laptop-scale demo; it trades some calibration for speed.

Calibration: if you want stable probabilities without hand-tuning τ, add isotonic calibration on data/test_preds.npz or auto-pick τ from coverage_curve.tsv to hit a target risk (both are a few lines).

References
Human-centric disagreement
Wan, R., Kim, J., & Kang, D. (2023). Everyone’s Voice Matters: Quantifying Annotation Disagreement Using Demographic Information. AAAI.
DOI: https://doi.org/10.1609/aaai.v37i12.26698

Zero-shot NLI auditor (eval)
Lewis, M. et al. (2020). BART: Denoising Sequence-to-Sequence Pre-training… ACL.
HF model: https://huggingface.co/facebook/bart-large-mnli