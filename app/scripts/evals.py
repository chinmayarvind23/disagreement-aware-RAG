# evals.py
import os
import re
import csv
import numpy as np
from pathlib import Path

from backend.rag import load_query_bundle, DOC_DIR
from backend.features import feature_vector
from backend.disagreement import DisagreeHead, decision_from_feats

from transformers import pipeline
from llama_index.core.response_synthesizers import get_response_synthesizer
from dotenv import load_dotenv
load_dotenv()

CURVE_OUT = Path("data/coverage_curve.tsv")
SC_SAMPLES        = int(os.getenv("SC_SAMPLES", "5"))
DEC_MIN_OVERLAP   = float(os.getenv("DEC_MIN_OVERLAP", "0.35"))
DEC_MAX_SC        = float(os.getenv("DEC_MAX_SC", "0.30"))
HALLUC_THRESHOLD  = float(os.getenv("HALLUC_THRESHOLD", "0.35"))
LIMIT_QUESTIONS   = int(os.getenv("EVAL_LIMIT", "30"))

def iter_questions(root="data/test", limit=LIMIT_QUESTIONS):
    q_pat = re.compile(r"^Question:\s*(.+)$", re.I)
    n = 0
    for p in Path(root).glob("*.txt"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line in txt.splitlines():
            m = q_pat.match(line.strip())
            if m:
                yield m.group(1).strip()
                n += 1
                if n >= limit:
                    return

# NLI model to score support of answer by evidence
nli = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",
    top_k=None,
    device=-1
)

def best_entail(passages: list[str], hypothesis: str) -> float:
    """Return max entailment score of hypothesis against any passage."""
    claim_sents = [s.strip() for s in re.split(r'(?<=[.?!])\s+', hypothesis)
                   if 3 <= len(s.strip()) <= 300] or [hypothesis.strip()]
    scores = []
    for prem in passages:
        pairs = [{"text": str(prem), "text_pair": s} for s in claim_sents]
        scores_list = nli(pairs, truncation=True, max_length=512)
        ent = [next(it["score"] for it in scores if it["label"].upper() == "ENTAILMENT")
               for scores in scores_list]
        scores.append(max(ent) if ent else 0.0)
    return max(scores) if scores else 0.0

def sample_answers(q: str, nodes, synthesizer, k: int = SC_SAMPLES) -> list[str]:
    outs = []
    for i in range(k):
        resp = synthesizer.synthesize(f"{q}\n(variation {i+1})", nodes)
        outs.append(str(resp))
    return outs

def main():
    retriever, _ = load_query_bundle(DOC_DIR)
    synthesizer = get_response_synthesizer(response_mode="compact")
    head = DisagreeHead.load("data/disagree_head.joblib")

    Q = list(iter_questions(root="data/test", limit=LIMIT_QUESTIONS))
    rows = []

    for i, q in enumerate(Q, 1):
        nodes = retriever.retrieve(q)
        resp = synthesizer.synthesize(q, nodes)
        passages = [n.get_text() for n in nodes[:3]]
        samples = sample_answers(q, nodes, synthesizer, k=SC_SAMPLES)
        feats = feature_vector(str(resp), passages, samples)
        p_dis = head.predict_proba(feats)
        ent = best_entail(passages, str(resp))

        rows.append({
            "q": q,
            "p_dis": p_dis,
            "resp_text": str(resp),
            "passages": passages,
            "best_entail": ent,
            **feats
        })
        if i % 2 == 0:
            print(f"[evals] {i}/{len(Q)}")

    # Ground truth for ROC: low entailment == hallucination/high-risk
    y_true = np.array([int(r["best_entail"] < HALLUC_THRESHOLD) for r in rows])  # 1 = hallucination
    y_score = np.array([r["p_dis"] for r in rows])
    np.savez("data/test_preds.npz", y_true=y_true, y_score=y_score)

    # Coverage vs hallucination across different tau values
    taus = [round(x, 2) for x in np.linspace(0.1, 0.8, 15)]
    results = []
    for tau in taus:
        answered = errs = 0
        for r in rows:
            decision = decision_from_feats(
                r["p_dis"], r, tau=tau, min_overlap=DEC_MIN_OVERLAP, max_sc=DEC_MAX_SC
            )
            if decision == "answer":
                answered += 1
                errs += int(r["best_entail"] < HALLUC_THRESHOLD)
        cov = answered / max(1, len(rows))
        err_rate = (errs / answered) if answered else 0.0
        results.append({"tau": tau, "coverage": round(cov, 3), "halluc_rate": round(err_rate, 3)})

    CURVE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CURVE_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tau", "coverage", "halluc_rate"], delimiter="\t")
        w.writeheader()
        w.writerows(results)

    print("\ncoverage vs hallucination-rate")
    for r in results:
        print(f"tau={r['tau']:.2f}  coverage={r['coverage']:.2f}  halluc_rate={r['halluc_rate']:.2f}")
    print(f"\n[wrote] {CURVE_OUT}")

if __name__ == "__main__":
    main()