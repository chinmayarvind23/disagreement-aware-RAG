import argparse, random, re
from pathlib import Path
import numpy as np
from backend.rag import load_query_bundle, DOC_DIR
from backend.features import feature_vector
from backend.disagreement import DisagreeHead
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import BaseRetriever

DATA_DIR = Path("data")
MODEL_OUT = DATA_DIR / "disagree_head.joblib"

# Gives the question text and file path of question in text files
def iter_questions_from_files(root="data/raw", limit=300):
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
                q = m.group(1).strip()
                if len(q) > 6:
                    yield q, p
                    n += 1
                    if n >= limit:
                        return

# Get top k passages from retrieved nodes as evidence
def _nodes_to_passages(nodes, k=3):
    out = []
    for n in nodes[:k]:
        try:
            out.append(n.get_text())
        except Exception:
            pass
    return out

# Answer question with the synthesizer and retriever
# Returns answer, passages of evidence, and nodes retrieved (similarity search results)
def _answer_once(q: str, retriever: BaseRetriever, synthesizer):
    nodes = retriever.retrieve(q)
    resp = synthesizer.synthesize(q, nodes)
    passages = _nodes_to_passages(nodes)
    return str(resp), passages, nodes

# Sample answers for a question using the retriever and synthesizer
# This function tries to get different phrasings of the answer by varying the temperature
def _sample_answers(q: str, retriever: BaseRetriever, base_synth, n=3):
    outs = []
    try:
        for t in (0.7, 0.9, 0.5)[:n]:
            synth = get_response_synthesizer(response_mode="compact")
            a, passages, nodes = _answer_once(q, retriever, synth)
            outs.append((a, passages, nodes))
    except Exception:
        a, passages, nodes = _answer_once(q, retriever, base_synth)
        outs.append((a, passages, nodes))
    return outs

# Main function to train the disagreement head
# It loads the retriever and synthesizer, samples answers for questions,
# computes features, and trains the DisagreeHead model
def main(n: int, out_path: str):
    retriever, synthesizer = load_query_bundle(DOC_DIR)

    X, y = [], []
    questions = list(iter_questions_from_files(DOC_DIR, limit=max(n, 200)))
    if not questions:
        raise SystemExit("No questions found in data/raw/*.txt. Please add some.")

    random.shuffle(questions)
    questions = questions[:n]

    for i, (q, _) in enumerate(questions, 1):
        samples = _sample_answers(q, retriever, synthesizer, n=3)
        # use the first sample as the "final" answer for features and rest used in self-consistency
        final_answer, passages, _ = samples[0]
        alt_answers = [a for (a, _, __) in samples]

        feats = feature_vector(final_answer, passages, alt_answers)

        # flags high disagreement when overlap is low or self-consistency variance is high
        high_disagree = int(feats["overlap"] < 0.35 or feats["sc_var"] > 0.45)

        X.append([feats["sc_var"], feats["overlap"], feats["entropy_proxy"]])
        y.append(high_disagree)

        if i % 20 == 0:
            print(f"[train_head] processed {i}/{len(questions)}")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    head = DisagreeHead()
    head.fit(X, y)
    head.save(out_path)
    print(f"[train_head] saved model to {out_path}")
    print(f"[train_head] proxy positives: {y.mean():.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="number of Qs to build proxy labels")
    ap.add_argument("--out", default=str(MODEL_OUT))
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    main(args.n, args.out)