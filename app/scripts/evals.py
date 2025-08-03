import numpy as np, re, csv
from pathlib import Path
from backend.rag import load_query_bundle, DOC_DIR
from backend.features import feature_vector
from backend.disagreement import DisagreeHead, decision_from_feats

CURVE_OUT = Path("data/coverage_curve.tsv")

# Gives the question text from text files
def iter_questions(root="data/raw", limit=150):
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

# Main function to evaluate the disagreement head
# It loads the retriever and synthesizer, collects features on a fixed set of questions,
# and computes the coverage and hallucination rate based on the disagreement head's predictions
def main():
    retriever, synthesizer = load_query_bundle(DOC_DIR)
    head = DisagreeHead.load("data/disagree_head.joblib")
    Q = list(iter_questions(DOC_DIR, limit=150))
    rows = []
    for i, q in enumerate(Q, 1):
        nodes = retriever.retrieve(q)
        resp = synthesizer.synthesize(q, nodes)
        passages = [n.get_text() for n in nodes[:3]]
        feats = feature_vector(str(resp), passages, [str(resp)])

        p_dis = head.predict_proba(feats)
        rows.append({"q": q, "p_dis": p_dis, **feats})
        if i % 20 == 0:
            print(f"[evals] {i}/{len(Q)}")

    taus = [round(x, 2) for x in np.linspace(0.1, 0.8, 15)]
    out = []
    for tau in taus:
        answered, errs = 0, 0
        for r in rows:
            decision = decision_from_feats(r["p_dis"], r, tau=tau)
            if decision == "answer":
                answered += 1
                # hallucination proxy: low overlap -> error
                errs += int(r["overlap"] < 0.35)
        cov = answered / max(1, len(rows))
        err_rate = (errs / answered) if answered else 0.0
        out.append({"tau": tau, "coverage": round(cov, 3), "halluc_rate": round(err_rate, 3)})

    CURVE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CURVE_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tau", "coverage", "halluc_rate"], delimiter="\t")
        w.writeheader()
        w.writerows(out)

    print("\ncoverage vs hallucination-rate")
    for r in out:
        print(f"tau={r['tau']:.2f}  coverage={r['coverage']:.2f}  halluc_rate={r['halluc_rate']:.2f}")
    print(f"\n[wrote] {CURVE_OUT}")

if __name__ == "__main__":
    main()