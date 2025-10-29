# 1) Tạo môi trường & cài thư viện

pip install -r requirements.txt

# 2) Chuẩn bị dữ liệu từ ViQuAD (chọn 5k mẫu để nhẹ)

python app/prepare_data.py --max_rows 5000

# 3) Build index embeddings (SentenceTransformer đa ngữ)

python app/build_index.py

# 4) (Tuỳ chọn) Tạo file .env chứa OPENAI_API_KEY để bật Self‑Critique

cp .env.example .env

# rồi mở .env, điền OPENAI_API_KEY=sk-...

# 5) Chạy ứng dụng

streamlit run app.py
