from huggingface_hub import snapshot_download
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. bartowski/Mistral-7B-Instruct-v0.2-GGUF")
    ap.add_argument("--pattern", default="*Q4_K_M.gguf", help="glob to pick quant")
    ap.add_argument("--out", default="models", help="target directory")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo,
        allow_patterns=[args.pattern],
        local_dir=args.out,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("Model(s) downloaded to", args.out)

if __name__ == "__main__":
    main()