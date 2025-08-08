import argparse, random, re
from pathlib import Path
import numpy as np
from backend.rag import load_query_bundle, DOC_DIR
from backend.features import feature_vector
from backend.disagreement import DisagreeHead
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import Settings
import wandb

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
def _nodes_to_passages(nodes, k=3, max_chars=900):
    out = []
    for n in nodes[:k]:
        try:
            txt = n.get_text()
            out.append(txt[:max_chars])
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
def _sample_answers(q: str, retriever: BaseRetriever, _, n=3):
    nodes = retriever.retrieve(q)
    passages = _nodes_to_passages(nodes)
    outs = []
    temps = (0.35, 0.55, 0.75)[:n]
    for t in temps:
        synth = get_response_synthesizer(response_mode="compact")
        if hasattr(Settings.llm, "temperature"):
            Settings.llm.temperature = t
        resp = synth.synthesize(q, nodes)
        outs.append((str(resp), passages, nodes))
    return outs

# Main function to train the disagreement head
# It loads the retriever and synthesizer, samples answers for questions,
# computes features, and trains the DisagreeHead model
def main(n: int, out_path: str):
    wandb.init(project="disagreement-rag-v2", entity="chinmayarvind23-student", name=f"training-n{n}-2")
    wandb.config.update({
        "n": n,
        "train_overlap_label_cut": 0.45,
        "train_scvar_label_cut": 0.2,
        "train_entropy_label_cut": 4.2,
        "alpha_blend": 0.65
    })

    retriever, synthesizer = load_query_bundle(DOC_DIR)
    retriever.similarity_top_k = 3

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
        high_disagree = int(
            feats["overlap"] < 0.45 or
            feats["sc_var"]  > 0.2 or
            feats["entropy_proxy"] > 1.5
        )


        X.append([feats["sc_var"], feats["overlap"], feats["entropy_proxy"]])
        y.append(high_disagree)

        if i % 20 == 0:
            print(f"[train_head] processed {i}/{len(questions)}")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    wandb.log({
    "proxy_positive_rate": float(y.mean())
    })
    
    head = DisagreeHead()
    head.fit(X, y)
    head.save(out_path)
    print(f"[train_head] saved model to {out_path}")
    print(f"[train_head] proxy positives: {y.mean():.2f}")
    wandb.log({
    "feat_overlap": float(feats["overlap"]),
    "feat_sc_var": float(feats["sc_var"]),
    "feat_entropy": float(feats["entropy_proxy"])})
    wandb.finish()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="number of Qs to build proxy labels")
    ap.add_argument("--out", default=str(MODEL_OUT))
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    main(args.n, args.out)