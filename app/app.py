import json, os, numpy as np, pandas as pd
from pathlib import Path
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from prompts import ANSWER_PROMPT, CRITIQUE_PROMPT

# OpenAI là optional – chỉ import khi thật sự cần
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

st.set_page_config(page_title="VN Semantic QA (ViQuAD)", page_icon="🇻🇳", layout="centered")
st.title("🇻🇳 VN Semantic QA – Demo map câu hỏi đồng nghĩa")
st.caption("Nguồn dữ liệu: ViQuAD. Tìm câu tương tự nhất bằng embedding đa ngữ và trả lời bằng tiếng Việt.")



DATA_DIR = Path("data")
if not (DATA_DIR / "embeddings.npy").exists():
    st.error(
        "Chưa có index. Hãy chạy: python prepare_data.py rồi python build_index.py"
    )
    st.stop()


# Load index
questions = json.loads((DATA_DIR / "questions.json").read_text(encoding="utf-8"))
answers = json.loads((DATA_DIR / "answers.json").read_text(encoding="utf-8"))
emb = np.load(DATA_DIR / "embeddings.npy")


# Embedder cho truy vấn (cùng model)
embedder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)


with st.sidebar:
    st.subheader("⚙️ Tuỳ chọn")
    topk = st.slider("Số câu tương tự xem xét (k)", 1, 10, 5)
    threshold = st.slider("Ngưỡng chấp nhận (cosine sim)", 0.50, 0.95, 0.70)
    want_crit = st.checkbox("Bật Self-Critique (OpenAI)", value=bool(OPENAI_API_KEY))


q = st.text_input("Nhập câu hỏi bằng tiếng Việt", value="Thủ đô CHXHCN Việt Nam là gì?")
btn = st.button("Hỏi")


if btn and q.strip():
    with st.spinner("Đang tìm câu hỏi tương đồng..."):
        q_vec = embedder.encode([q], normalize_embeddings=True)
        # cosine scores shape (1, N)
        scores = np.matmul(q_vec, emb.T).reshape(-1)
        idx_sorted = np.argsort(-scores)[:topk]
        candidates = [(int(i), float(scores[i])) for i in idx_sorted]

    st.markdown("### 🔎 Kết quả tìm gần nhất")
    for rank, (i, sc) in enumerate(candidates, start=1):
        st.write(f"**#{rank}** · score = {sc:.3f}")
        with st.expander(questions[i]):
            st.markdown(f"**Đáp án gold:** {answers[i]}")

    best_i, best_score = candidates[0]
    if best_score < threshold:
        st.warning(
            "⚠️ Không tìm thấy câu hỏi tương tự đủ ngưỡng.\n\n→ Gợi ý: thử diễn đạt lại hoặc hạ ngưỡng threshold."
        )
        st.stop()

    candidate_answer = answers[best_i]

    # Baseline: trả thẳng gold gần nhất
    st.markdown("---")
    st.subheader("🟦 Baseline (không phản biện)")
    st.markdown(f"**Câu trả lời (đề xuất):** {candidate_answer}")
    st.caption(f"Nguồn: câu hỏi gần nhất · score={best_score:.3f}")

    # Optional self-critique via OpenAI
    st.markdown("---")
    st.subheader("🟪 Self-Critique (tự phản biện)")
    critique_text = None
    if want_crit and OPENAI_API_KEY and OpenAI:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)  # khởi tạo TRONG try/catch
            prompt = CRITIQUE_PROMPT.format(question=q, candidate=candidate_answer)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",  # hoặc gpt-4-turbo / gpt-3.5-turbo nếu tài khoản không có 4o
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            critique_text = resp.choices[0].message.content
            st.text_area("Phản biện & Đáp án cuối", value=critique_text, height=200)
        except Exception as e:
            st.error(f"Không bật được Self-Critique (sẽ dùng Baseline). Lý do: {e}")
            st.text_area("Phản biện & Đáp án cuối", value=f"(Baseline) Đáp án cuối: {candidate_answer}", height=120)
    else:
        st.info("Chưa bật Self-Critique. Đang hiển thị đáp án Baseline.")
        st.text_area("Phản biện & Đáp án cuối", value=f"(Baseline) Đáp án cuối: {candidate_answer}", height=120)

    # Nhãn nhanh về độ tự tin dựa vào score
    conf = (
        "cao"
        if best_score >= 0.85
        else ("trung bình" if best_score >= 0.75 else "thấp")
    )
    st.markdown(f"**Độ tự tin ngữ nghĩa**: {conf} (cosine={best_score:.3f})")
