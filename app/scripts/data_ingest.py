import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from backend.rag import build_index

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw", help="directory with .txt/.md files")
    args = ap.parse_args()
    build_index(args.src)
    print("Index built at data/index")