import json, random, pathlib
from backend.rag import load_query_bundle, answer_query, DOC_DIR
from backend.features import feature_vector
from backend.disagreement import DisagreeHead, decision_from_feats

# Load retriever, LLM and disagreement head trained model
retr, synth = load_query_bundle(DOC_DIR)
head = DisagreeHead.load("data/disagree_head.joblib")

# Get questions
paths = list(pathlib.Path(DOC_DIR).glob("*.txt"))

def get_question(p: pathlib.Path):
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.lower().startswith("question:"):
            return line.split(":", 1)[1].strip()
    return None

picked = random.sample(paths, min(5, len(paths)))
for p in picked:
    q = get_question(p)
    if not q:
        continue
    resp = answer_query(q, retr, synth)
    passages = [s["text"] for s in resp["sources"]]
    feats = feature_vector(resp["answer"], passages, [resp["answer"]])
    pdis = head.predict_proba(feats)
    decision = decision_from_feats(pdis, feats)
    print(json.dumps({
        "file": p.name,
        "q": (q[:80] + "…") if len(q) > 80 else q,
        "p_disagree": round(pdis, 2),
        "overlap": round(feats["overlap"], 2),
        "sc_var": round(feats["sc_var"], 2),
        "decision": decision
    }, ensure_ascii=False))