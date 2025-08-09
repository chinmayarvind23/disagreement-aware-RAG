import numpy as np, re, csv
from pathlib import Path
from backend.rag import load_query_bundle, DOC_DIR
from backend.features import feature_vector
from backend.disagreement import DisagreeHead, decision_from_feats
from transformers import pipeline

CURVE_OUT = Path("data/coverage_curve.tsv")

# Gives the question text from text files
def iter_questions(root="data/test", limit=30):
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

# Creating a text classification pipeline for NLI
# This is used to classify the agreement/disagreement of the answer with the evidence
nli = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",
    top_k=None,
    device=0
)

# Main function to evaluate the disagreement head
# It loads the retriever and synthesizer, collects features on a fixed set of questions,
# and computes the coverage and hallucination rate based on the disagreement head's predictions
def main():
    retriever, _ = load_query_bundle(DOC_DIR)
    from llama_index.core.response_synthesizers import get_response_synthesizer
    synthesizer = get_response_synthesizer(response_mode="compact")
    head = DisagreeHead.load("data/disagree_head.joblib")
    Q = list(iter_questions(root="data/test", limit=30))
    rows = []
    for i, q in enumerate(Q, 1):
        nodes = retriever.retrieve(q)
        resp = synthesizer.synthesize(q, nodes)
        passages = [n.get_text() for n in nodes[:3]]
        feats = feature_vector(str(resp), passages, [str(resp)])

        p_dis = head.predict_proba(feats)
        rows.append({
            "q": q,
            "p_dis": p_dis,
            "resp_text": str(resp),
            "passages": passages,
            **feats
        })
        if i % 2 == 0:
            print(f"[evals] {i}/{len(Q)}")
    y_true = np.array([
        int(
            (r["overlap"] < 0.45)
            or (r["sc_var"] > 0.2)
            or (r["entropy_proxy"] > 1.5)
        ) for r in rows
            ])
    y_score = np.array([r["p_dis"] for r in rows])
    np.savez("data/test_preds.npz", y_true=y_true, y_score=y_score)
    taus = [round(x, 2) for x in np.linspace(0.1, 0.8, 15)]
    results = []
    halluc_threshold = 0.35
    for tau in taus:
        answered, errs = 0, 0
        for r in rows:
            decision = decision_from_feats(r["p_dis"], r, tau=tau)
            if decision == "answer":
                answered += 1
                claim_sents = [s.strip() for s in re.split(r'(?<=[.?!])\s+', r["resp_text"]) if 3 <= len(s.strip()) <= 300]
                if not claim_sents:
                    claim_sents = [r["resp_text"].strip()]
                entail_scores_per_passage = []
                for prem in r["passages"]:
                    pairs = [{"text": str(prem), "text_pair": s} for s in claim_sents]
                    scores_list = nli(pairs, truncation=True, max_length=512)
                    ent_scores = [
                        next(item["score"] for item in scores if item["label"].upper() == "ENTAILMENT")
                        for scores in scores_list
                    ]
                    entail_scores_per_passage.append(max(ent_scores) if ent_scores else 0.0)
                best_entail = max(entail_scores_per_passage) if entail_scores_per_passage else 0.0
                errs += int(best_entail < halluc_threshold)
        cov = answered / max(1, len(rows))
        err_rate = (errs / answered) if answered else 0.0
        results.append({
            "tau": tau,
            "coverage": round(cov, 3),
            "halluc_rate": round(err_rate, 3)
        })

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