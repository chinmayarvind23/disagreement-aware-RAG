from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from backend.rag import load_query_bundle, answer_query, DOC_DIR
from backend.features import feature_vector
from backend.disagreement import DisagreeHead, decision_from_feats
import csv
import numpy as np
from sklearn.metrics import roc_auc_score

app = FastAPI(title="Disagreement-Aware RAG")

# load RAG and head
_retriever, _synth = load_query_bundle(DOC_DIR)
_head = None
try:
    _head = DisagreeHead.load("data/disagree_head.joblib")
except Exception:
    _head = DisagreeHead()

class QARequest(BaseModel):
    query: str

class QAResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]
    risk: Dict[str, float]
    decision: str

@app.get("/healthz")
def health():
    return {"ok": True}

@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest):
    # RAG answer + sources cited
    out = answer_query(req.query, _retriever, _synth)
    answer = out["answer"]
    sources = out["sources"]
    
    # Compute features for disagreement risk
    passages = [s["text"] for s in sources[:3]]
    feats = feature_vector(answer, passages, [answer])

    # predict disagreement probability
    if getattr(_head, "model", None) is not None:
        p_dis = _head.predict_proba(feats)
        decision = decision_from_feats(p_dis, feats, tau=_head.threshold)
    else:
        p_dis = 0.5
        decision = "answer"

    return QAResponse(
        answer=answer,
        sources=sources,
        risk={
            "p_disagree": float(p_dis),
            "sc_var": float(feats["sc_var"]),
            "overlap": float(feats["overlap"]),
            "entropy_proxy": float(feats["entropy_proxy"]),
        },
        decision=decision,
    )

@app.get("/metrics")
def metrics():
    curve = []
    with open("data/coverage_curve.tsv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            curve.append({
                "tau": float(row["tau"]),
                "coverage": float(row["coverage"]),
                "halluc_rate": float(row["halluc_rate"])
            })

    data = np.load("data/test_preds.npz")
    auc = float(roc_auc_score(data["y_true"], data["y_score"]))

    return {
        "coverage_curve": curve,
        "roc_auc": auc
    }