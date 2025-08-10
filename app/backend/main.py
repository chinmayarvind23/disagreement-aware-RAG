from fastapi import FastAPI, HTTPException
import traceback
from pydantic import BaseModel
from typing import List, Dict
from backend.rag import load_query_bundle, answer_query, DOC_DIR
from backend.features import feature_vector
from backend.disagreement import DisagreeHead, decision_from_feats
import csv
import numpy as np
from sklearn.metrics import roc_auc_score
from fastapi.middleware.cors import CORSMiddleware
import os
from llama_index.core import Settings, get_response_synthesizer
from llama_index.core import ServiceContext
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Disagreement-Aware RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000", "http://192.168.0.110:3000"
    ],
    allow_methods=["*"], allow_headers=["*"],
)

HEAD_TAU = float(os.getenv("HEAD_TAU", "0.55")) 
MIN_OVERLAP = float(os.getenv("DEC_MIN_OVERLAP", "0.35"))
MAX_SC = float(os.getenv("DEC_MAX_SC", "0.30"))
SC_SAMPLES = int(os.getenv("SC_SAMPLES", "5"))

# load RAG and head
_retriever, _synth = load_query_bundle(DOC_DIR)
_retriever.similarity_top_k = 3
try:
    _head = DisagreeHead.load("data/disagree_head.joblib")
except Exception:
    _head = DisagreeHead()
_head.threshold = HEAD_TAU

def _sample_answers(q: str, k: int = SC_SAMPLES) -> list[str]:
    nodes = _retriever.retrieve(q)
    base = float(os.getenv("RAG_TEMP", "0.7"))
    temps = [round(t, 2) for t in (base*0.7, base*0.85, base, base*1.15, base*1.3)][:k]

    outs: list[str] = []
    for i, t in enumerate(temps, 1):
        synth_t = None
        try:
            if hasattr(Settings.llm, "with_params"):
                llm_t = Settings.llm.with_params(temperature=float(t))
                sc = ServiceContext.from_defaults(llm=llm_t, embed_model=Settings.embed_model)
                synth_t = get_response_synthesizer(service_context=sc, response_mode="compact")
        except Exception:
            synth_t = None

        if synth_t is None:
            synth_t = _synth

        resp = synth_t.synthesize(f"{q}\n(variation #{i} @T={t})", nodes)
        outs.append(getattr(resp, "response", str(resp)))
    return outs


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
    try:
        # RAG answer + sources cited
        out = answer_query(req.query, _retriever, _synth)
        answer = out["answer"]
        sources = out["sources"]
        
        # Compute features for disagreement risk
        passages = [s["text"] for s in sources[:3]]
        samples = _sample_answers(req.query, SC_SAMPLES)
        feats = feature_vector(answer, passages, samples)

        # predict disagreement probability
        if getattr(_head, "model", None) is not None:
            p_dis = _head.predict_proba(feats)
            decision = decision_from_feats(
                p_dis, feats,
                tau=_head.threshold,
                min_overlap=MIN_OVERLAP,
                max_sc=MAX_SC
            )
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
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

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