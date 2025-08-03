# scripts/get_data.py
import argparse, hashlib
from pathlib import Path
from datasets import load_dataset

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

def write_text(path: Path, url: str, title: str, question: str, answer: str, context: str, unans: bool=False):
    header = (
        f"URL: {url}\n"
        f"Title: {title}\n"
        f"Unanswerable: {str(unans)}\n\n"
        f"Question: {question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Context:\n{context}\n"
    )
    path.write_text(header, encoding="utf-8")

def add_squad_v2(split="train", limit=200):
    ds = load_dataset("squad_v2", split=split)
    n = 0
    for row in ds:
        q = (row.get("question") or "").strip()
        ctx = (row.get("context") or "").strip()
        is_unans = bool(row.get("is_impossible", False))
        ans_list = row.get("answers", {}).get("text", []) if isinstance(row.get("answers"), dict) else []
        ans = (ans_list[0] if (ans_list and not is_unans) else "").strip()
        if not q or not ctx:
            continue
        url = f"hf://squad_v2/{split}"
        h = hashlib.md5((q + ctx[:100]).encode()).hexdigest()[:12]
        write_text(OUT / f"qa_squadv2_{h}.txt", url, "SQuADv2", q, ans, ctx, unans=is_unans)
        n += 1
        if n >= limit:
            break
    return n

def add_boolq(split="train", limit=200):
    ds = load_dataset("google/boolq", split=split)
    n = 0
    for row in ds:
        q = (row.get("question") or "").strip()
        ctx = (row.get("passage") or "").strip()
        ans = "Yes" if bool(row.get("answer", False)) else "No"
        if not q or not ctx:
            continue
        url = f"hf://boolq/{split}"
        h = hashlib.md5((q + ctx[:100]).encode()).hexdigest()[:12]
        write_text(OUT / f"qa_boolq_{h}.txt", url, "BoolQ", q, ans, ctx, unans=False)
        n += 1
        if n >= limit:
            break
    return n

def main(sources: list[str], split: str, limit: int):
    per = max(1, limit // max(1, len(sources)))
    total = 0
    for name in sources:
        if name == "squad_v2":
            total += add_squad_v2(split=split, limit=per)
        elif name == "boolq":
            total += add_boolq(split=split, limit=per)
    print(f"[get_data] wrote ~{total} files to {OUT}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="squad_v2,boolq")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    main(sources, args.split, args.limit)