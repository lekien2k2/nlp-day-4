# 📊 Hướng Dẫn Chạy Thí Nghiệm: Reducing Hallucinations

## 🎯 Mục Tiêu Nghiên Cứu

Thí nghiệm này kiểm tra xem kỹ thuật **Self-Critique prompting** có giảm hallucination và cải thiện factual accuracy so với **direct prompting** không.

## 📋 Yêu Cầu

1. **Gemini API Key**: Cần có API key từ Google AI Studio
2. **Dataset**: File `data/questions.json` và `data/answers.json` (đã có từ ViQuAD)
3. **Python packages**: Đã cài trong `pyproject.toml`

## 🚀 Cách Chạy

### Bước 1: Cấu hình API Key

Thêm GEMINI_API_KEY vào file `.env`:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### Bước 2: Cài đặt thư viện (nếu chưa có)

```bash
poetry add google-generativeai
```

### Bước 3: Chạy thí nghiệm

```bash
poetry run python app/gemini.py
```

### Bước 4: Xem kết quả

Sau khi chạy xong, bạn sẽ có:

- `experiment_results.csv` - Chi tiết từng câu hỏi
- `experiment_summary.txt` - Báo cáo tóm tắt
- Output trên console - Thống kê đầy đủ

## 📊 Kết Quả Mong Đợi

Script sẽ in ra màn hình:

```
📊 KẾT QUẢ THÍ NGHIỆM: REDUCING HALLUCINATIONS VỚI SELF-CRITIQUE
======================================================================

📌 Thông tin thí nghiệm:
   • Số câu hỏi test: 50
   • Dataset: ViQuAD
   • Model: Gemini 1.5 Pro
   • Similarity threshold: 0.6 (60%)

📈 KẾT QUẢ CHÍNH:

   1️⃣  BASELINE (Direct Prompt):
      • Accuracy: XX.XX%
      • Average Similarity: X.XXXX
      • Số câu trả lời đúng: XX/50

   2️⃣  SELF-CRITIQUE (3-Step Prompt):
      • Accuracy: XX.XX%
      • Average Similarity: X.XXXX
      • Số câu trả lời đúng: XX/50

   ✨ IMPROVEMENT:
      • Accuracy Improvement: +X.XX% (absolute)
      • Relative Improvement: +X.XX%
      • Similarity Improvement: +X.XXXX
```

## 📈 Các Thông Số Cho Báo Cáo

### 1. Primary Metrics

- **Baseline Accuracy** (%): Độ chính xác của prompt đơn giản
- **Self-Critique Accuracy** (%): Độ chính xác của prompt 3 bước
- **Accuracy Improvement**: Cải thiện tuyệt đối (%)
- **Relative Improvement**: Cải thiện tương đối (%)

### 2. Secondary Metrics

- **Average Similarity Score**: Điểm tương đồng trung bình (0-1)
- **Win/Loss Breakdown**:
  - Số câu Self-Critique tốt hơn
  - Số câu Baseline tốt hơn
  - Số câu bằng nhau

### 3. Detailed Analysis

File `experiment_results.csv` chứa:

- `question`: Câu hỏi
- `ground_truth`: Đáp án đúng
- `baseline_answer`: Câu trả lời của Baseline
- `baseline_correct`: True/False
- `baseline_similarity`: Điểm tương đồng (0-1)
- `critique_answer_final`: Câu trả lời cuối của Self-Critique
- `critique_correct`: True/False
- `critique_similarity`: Điểm tương đồng (0-1)

## ⚙️ Tùy Chỉnh

### Thay đổi số lượng mẫu test

Trong file `app/gemini.py`, dòng 123:

```python
NUM_SAMPLES = 50  # Thay đổi số này (vd: 100, 200)
```

### Thay đổi ngưỡng similarity

Trong hàm `evaluate_answer()`, dòng 93:

```python
def evaluate_answer(predicted, ground_truth, threshold=0.6):  # Thay 0.6 thành giá trị khác
```

### Thay đổi prompt design

Sửa hàm `get_critique_prompt()` để điều chỉnh cách AI phản biện.

## 💡 Tips Cho Báo Cáo

### Phần Method

- Mô tả 2 prompt templates
- Giải thích evaluation metric (similarity score)
- Nêu rõ dataset và sample size

### Phần Results

- Trích dẫn Accuracy Improvement
- So sánh Baseline vs Self-Critique
- Thêm biểu đồ từ CSV (nếu cần)

### Phần Discussion

- Phân tích tại sao Self-Critique tốt/không tốt hơn
- Đề cập limitation (sample size, evaluation method)
- Đề xuất future work

## 🐛 Troubleshooting

### Lỗi API Key

```
Không tìm thấy GEMINI_API_KEY trong file .env
```

→ Kiểm tra file `.env` đã có key chưa

### Lỗi Rate Limit

```
429 Too Many Requests
```

→ Giảm NUM_SAMPLES hoặc thêm delay giữa các requests

### File không tồn tại

```
Không tìm thấy file questions.json
```

→ Chạy `python app/build_index.py` trước

## 📚 Tham Khảo

- Paper idea dựa trên nghiên cứu về reducing hallucinations in LLMs
- Self-critique technique: Chain of Verification (CoVe) và tương tự

---

Good luck với báo cáo! 🚀
