import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


print("📖 Đọc data/qa_viquad.csv ...")
df = pd.read_csv(DATA_DIR / "qa_viquad.csv")
questions = df["question"].astype(str).tolist()
answers = df["gold"].astype(str).tolist()


print("🧠 Nạp model embedding đa ngữ...")
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)


print("🧮 Tính embeddings (chuẩn hoá)...")
emb = model.encode(questions, normalize_embeddings=True)
emb = np.asarray(emb, dtype=np.float32)


print("💾 Lưu index & metadata...")
(DATA_DIR / "questions.json").write_text(
    json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8"
)
(DATA_DIR / "answers.json").write_text(
    json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8"
)
np.save(DATA_DIR / "embeddings.npy", emb)
print("✅ Hoàn tất: data/questions.json, data/answers.json, data/embeddings.npy")
