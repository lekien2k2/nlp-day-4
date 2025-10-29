# app/prepare_data.py
import argparse
import sys, io
from pathlib import Path
import pandas as pd
from datasets import load_dataset

# (tùy chọn) fix UTF-8 cho Windows console
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_rows", type=int, default=5000, help="Số mẫu tối đa để export")
    args = parser.parse_args()

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Tải UIT-ViQuAD2.0 từ HuggingFace (split=train)...")
    # ViQuAD v2 trên HF: taidng/UIT-ViQuAD2.0
    ds = load_dataset("taidng/UIT-ViQuAD2.0", split="train")

    rows = []
    target = max(1, int(args.max_rows))
    for ex in ds:
        q = (ex.get("question") or "").strip()
        # answers là một dict với key "text" là list các đáp án
        ans = ""
        answers = ex.get("answers") or {}
        texts = answers.get("text") or []
        if texts:
            ans = (texts[0] or "").strip()

        if q and ans:
            rows.append({"question": q, "gold": ans})
            if len(rows) >= target:
                break

    if not rows:
        print("⚠️ Không thu được dòng nào. Kiểm tra lại phiên bản `datasets` hoặc kết nối mạng.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.drop_duplicates(subset=["question"], inplace=True)
    df = df.head(target)

    out_path = out_dir / "qa_viquad.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Ghi {out_path} với {len(df)} dòng")

if __name__ == "__main__":
    main()
