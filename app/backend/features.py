# Basic features that estimate "disagreement risk" from the model's behavior.
import itertools
import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util as st_util
from pathlib import Path

# Sentence encoder for similarity scoring of multiple answers from the LLM (do they agree with each other?)
_embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ROUGE-L score to measure match between answer and evidence (is the answer supported by the sources?)
_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# Try to import llama.cpp for entropy estimation, if not available, set to None.
# Used to compute mean token entropy of answer, which indicates uncertainty in the LLM.
try:
    from llama_cpp import Llama
    _LLAMA_OK = True
except Exception:
    Llama = None
    _LLAMA_OK = False

_MODELS_DIR = (Path(__file__).resolve().parents[2] / "models")
_LLAMA_PATH = None
if _LLAMA_OK and _MODELS_DIR.exists():
    ggufs = list(_MODELS_DIR.glob("*.gguf"))
    if ggufs:
        _LLAMA_PATH = str(ggufs[0])

_llm_entropy = None

def _get_llama_entropy_model():
    """Load llama.cpp with logits_all=True once, if a GGUF exists."""
    global _llm_entropy
    if _llm_entropy is None and _LLAMA_PATH:
        _llm_entropy = Llama(
            model_path=_LLAMA_PATH,
            logits_all=True,   # per-token logits
            n_threads=8,
            verbose=False,
        )
    return _llm_entropy

# Stable softmax function to avoid overflow/underflow issues -> compute probabilities from logits
def _softmax_stable(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x, dtype=np.float64)
    return (e / (e.sum() + 1e-12)).astype(np.float64)

# Compute mean Shannon entropy over the tokens of `text` using llama.cpp
def mean_token_entropy_on_text(text: str) -> float | None:
    m = _get_llama_entropy_model()
    if m is None or not text:
        return None
    try:
        out = m(
            text,
            max_tokens=0,
            temperature=0.0, # deterministic
            logits_all=True,
            echo=True,        # return logits for prompt tokens
        )
        logits = out.get("logits", None)
        if not logits:
            return None
        ents = []
        for row in logits:
            row = np.asarray(row, dtype=np.float32)
            p = _softmax_stable(row)
            ents.append(float(-(p * np.log(p + 1e-12)).sum()))
        return float(np.mean(ents)) if ents else None
    except Exception:
        return None

# Compute variance of self-consistency across many answers, higher variance means more disagreement among answers
def self_consistency_variance(answers: list[str]) -> float:
    # 0 variance if no multiple answers
    if len(answers) < 2:
        return 0.0
    # Encode all answers and compute pairwise cosine similarities
    emb = _embed.encode(answers, convert_to_tensor=True, normalize_embeddings=True)
    sims = []
    for i, j in itertools.combinations(range(len(answers)), 2):
        sims.append(float(st_util.cos_sim(emb[i], emb[j]).cpu().numpy()))
    # Higher variance => lower mean similarity
    return float(1.0 - np.mean(sims))

# Computing ROUGE-L overlap between answer and top-k passages of evidence (how well does the answer match the sources?)
def evidence_overlap(answer: str, passages: list[str]) -> float:
    # Concatenate top-k passages of evidence and compute ROUGE-L F-measure for answer
    joined = "\n".join(passages)[:4000]
    scores = _scorer.score(joined, answer)
    return float(scores["rougeL"].fmeasure)

# Create feature vector for classifier head for disagreement risk
def feature_vector(answer: str, passages: list[str], samples: list[str]) -> dict:
    scv = self_consistency_variance(samples)
    ovl = evidence_overlap(answer, passages)
    
    # Try true token-level entropy on the produced answer; fallback if unavailable
    true_entropy = mean_token_entropy_on_text(answer)
    entropy_val = true_entropy if true_entropy is not None else (len(answer) / 1000.0 + scv)
    # Return the feature vector
    return {
        "sc_var": scv,
        "overlap": ovl,
        "entropy_proxy": float(entropy_val),
    }