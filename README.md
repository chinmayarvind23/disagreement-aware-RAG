Disagreement-Aware RAG (Answer/Abstain)
=======================================

This app is a retrieval-augmented QA system that **predicts disagreement risk** and **abstains** when answers are likely contentious or weakly grounded. This improves reliability at a chosen coverage (questions answered out of total asked) level. This project was created inspired by Kang et al., *Everyone's Voice Matters* (AAAI 2023).

* * * * *

Why this matters
----------------

Large language models can be confident about incorrect facts. Kang et al.' states that **disagreement is signal, not noise**, this project treats "people would disagree here" as a prediction target and enforce an **answer/abstain policy**: answer when risk is low and evidence support is strong; abstain otherwise. This yields a clear **coverage--risk trade-off** that one can tune with larger datasets and for safety critical use cases.

-   Kang, Dongyeop et al., 2023 --- *Everyone's Voice Matters: Quantifying Annotation Disagreement Using Demographic Information.* AAAI. DOI: <https://doi.org/10.1609/aaai.v37i12.26698>

* * * * *

What the app does
-----------------

-   **Ask & Cite**: User types in a query. The system retrieves documents, synthesizes an answer with citations, computes **disagreement risk**, and either **answers** or **abstains**.

-   **Metrics**: The UI plots **Coverage vs. Hallucination** as the abstention threshold **τ** (tau) changes. The backend also reports an ROC-AUC against an entailment-based auditor.

**Decision rule (simple):**

`Answer  iff  (p_disagree < τ)  AND  (overlap ≥ min_overlap)  AND  (sc_var ≤ max_sc)
Else → Abstain`

-   `p_disagree`: predicted probability that humans (or annotators) would disagree

-   `overlap`: ROUGE-L--style semantic overlap between answer and retrieved evidence

-   `sc_var`: self-consistency variance across k re-sampled answers (higher => less stable)

* * * * *

How it works
-------------------------

1.  **Retrieve** top-k passages from a small indexed corpus (BM25 + FAISS vectors via LlamaIndex).

2.  **Generate** an answer and **k** temperature-varied re-samples (e.g., *T* ≈ 0.49, 0.60, 0.70, 0.80, 0.91) to measure **self-consistency**.

3.  **Compute features**:

    -   `sc_var` --- variance across the k-sampled answers

    -   `overlap` --- ROUGE-L--like overlap between the final answer and concatenated evidence

    -   `entropy_proxy` --- light uncertainty proxy from the answer text

4.  **Predict disagreement risk** with a **logistic regression head** → `p_disagree`.

5.  **Answer/Abstain** with the rule above → **answer** or **abstain**.

6.  **Evaluate (offline)**: Uses a zero-shot NLI auditor (**facebook/bart-large-mnli**) to score **entailment** of the answer against the evidence. If best-entailment < threshold, count it as a hallucination.

    -   Model card: https://huggingface.co/facebook/bart-large-mnli

    -   BART paper: Lewis et al., 2020 --- *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.*

* * * * *

Results
---------------------------------

From `scripts/evals.py` on a small test set (30 questions):

-   **τ = 0.55** → **coverage ≈ 96.7%** with **hallucination ≈ 3.4%**

-   **τ = 0.60--0.65** → **coverage ≈ 100%** with **hallucination ≈ 3.3%**

The UI renders the **coverage--risk curve** directly from `data/coverage_curve.tsv`.\
Note: ROC-AUC on the proxy labels is modest (expected on tiny, synthetic data), but the **policy curve** is stable and useful.

* * * * *

Tech stack
----------

-   **Backend**: FastAPI, Pydantic, scikit-learn, numpy, python-dotenv, HF Transformers

-   **RAG**: LlamaIndex (retrieval & synthesis)

-   **Eval auditor and Experiment Tracking**: `facebook/bart-large-mnli` (HF Transformers), Weights and Biases

-   **Frontend**: Next.js (React), Recharts (Coverage--Risk plot)

-   **Tooling**: Poetry (or `uv`), `.env` configuration

* * * * *

```text
app/
├── backend/
│   ├── main.py              # FastAPI app: /qa, /metrics
│   ├── rag.py               # index/retrieval + answer synthesis (LlamaIndex)
│   ├── features.py          # sc_var, overlap, entropy features
│   └── disagreement.py      # logistic head + decision rule
├── frontend/
│   └── app/page.tsx         # Next.js page (Ask & Cite + plot)
├── scripts/
│   ├── evals.py             # builds coverage vs hallucination curve + AUC
│   └── train_head.py        # train the logistic head
└── data/
    ├── index/...            # cached index
    ├── coverage_curve.tsv   # evals
    ├── test_preds.npz       # evals (scores + labels)
    └── disagree_head.joblib # saved head
```

> **Run from `app/`**

* * * * *

Setup & run
-----------

0) Requirements

-   Python 3.11

-   Node 20+

-   Poetry (or `uv`) recommended

1) Backend install

```
cd app
poetry install or uv sync
```

2) Configure params (`app/.env`)

```
HEAD_TAU=0.60            # τ (risk tolerance). Higher → answer more.
DEC_MIN_OVERLAP=0.35     # require at least this overlap to answer
DEC_MAX_SC=0.30          # require self-consistency variance ≤ this
SC_SAMPLES=5             # re-samples to estimate sc_var (k)
```

Generation sampling (for sc_var)

`
RAG_TEMP=0.7
`

Evaluation (entailment auditor threshold)

`
HALLUC_THRESHOLD=0.35
`

3) Start the backend

```
# from app/
poetry run uvicorn backend.main:app --reload --port 8000
```

Health check

curl http://127.0.0.1:8000/healthz

4) Start the frontend

```
cd app/frontend
npm install
npm run dev
```

open http://localhost:3000



5) Reproduce the metrics plot

```
# from app/
poetry run python -m scripts.evals
```

Writes data/coverage_curve.tsv and data/test_preds.npz
Refresh the UI to see the curve & AUC

* * * * *

API description
-----------

### `POST /qa`

**Body**

`{ "query": "your question" }`

**Response**

`{
  "answer": "...",
  "sources": [{"title": "...", "text": "..."}],
  "risk": {
    "p_disagree": 0.42, "sc_var": 0.08, "overlap": 0.61, "entropy_proxy": 0.12
  },
  "decision": "answer"   // or "abstain"
}`

### `GET /metrics`

`{
  "coverage_curve": [
    { "tau": 0.55, "coverage": 0.967, "halluc_rate": 0.034 },
    { "tau": 0.60, "coverage": 1.000, "halluc_rate": 0.033 }
  ],
  "roc_auc": 0.82
}`

* * * * *

Interpreting τ (risk tolerance)
-------------------------------

-   **Lower τ** → more conservative (answer less; abstain more; fewer mistakes).

-   **Higher τ** → more permissive (answer more; slightly higher risk).\
    On this toy set, **τ ≈ 0.60** gave **~100% coverage** at **~3--4% hallucination**.

* * * * *

Limitations & notes
-------------------

-   **Data scale**: small corpus + small eval set → the ROC-AUC can be noisy. The coverage vs. hallucination curve is the main focus here.

-   **Proxy labels**: the head is trained against **LLM-response proxies** (self-consistency, entailment, overlap), not human-annotator disagreement labels. That was done due to a lack of human disagreement data. The **logistic regression head** is trained on a few samples of synthetic data, so it may not generalize to all domains.

* * * * *

References
----------

-   **Human-centric disagreement**\
    Wan, R., Kim, J., & Kang, D. (2023). *Everyone's Voice Matters: Quantifying Annotation Disagreement Using Demographic Information.* AAAI.\
    DOI: <https://doi.org/10.1609/aaai.v37i12.26698>

-   **Zero-shot NLI auditor (eval)**\
    Lewis, M. et al. (2020). *BART: Denoising Sequence-to-Sequence Pre-training...* ACL.\
    HF model: https://huggingface.co/facebook/bart-large-mnli
