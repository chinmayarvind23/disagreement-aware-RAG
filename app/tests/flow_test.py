import json, random, pathlib
from backend.rag import load_query_bundle, answer_query, DOC_DIR
from backend.features import feature_vector
from backend.disagreement import DisagreeHead, decision_from_feats
from llama_index.core import Settings
from llama_index.core.response_synthesizers import get_response_synthesizer

def sample_answers(q, retriever, _, n=3, k_passages=3):
    # get nodes
    nodes = retriever.retrieve(q)
    passages = [n.get_text()[:900] for n in nodes[:k_passages]]

    outs = []
    temps = (0.35, 0.55, 0.75)[:n]
    old_temp = getattr(Settings.llm, "temperature", None)
    try:
        for t in temps:
            synth_t = get_response_synthesizer(response_mode="compact")
            if hasattr(Settings.llm, "temperature"):
                Settings.llm.temperature = t
            resp = synth_t.synthesize(q, nodes)
            outs.append(str(resp))
    finally:
        if old_temp is not None and hasattr(Settings.llm, "temperature"):
            Settings.llm.temperature = old_temp
    return outs, passages

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

retr.similarity_top_k = 3

picked = random.sample(paths, min(5, len(paths)))
for p in picked:
    q = get_question(p)
    if not q:
        continue

    # sample multiple answers
    alts, passages = sample_answers(q, retr, synth, n=3)
    final_answer = alts[0]

    feats = feature_vector(final_answer, passages, alts)
    pdis = head.predict_proba(feats)

    decision = decision_from_feats(
        pdis, feats,
        tau=head.threshold,
        min_overlap=0.6,
        max_sc=0.2
    )

    print(json.dumps({
        "file": p.name,
        "q": (q[:80] + "…") if len(q) > 80 else q,
        "p_disagree": round(pdis, 2),
        "overlap": round(feats["overlap"], 2),
        "sc_var": round(feats["sc_var"], 2),
        "decision": decision
    }, ensure_ascii=False))

    if decision == "answer":
        print("\nANSWER:\n", final_answer)
        print("\nSOURCES:")
        for s in passages:
            print("-", s[:160].replace("\n", " "), "...")
        print("\n" + "-"*60 + "\n")