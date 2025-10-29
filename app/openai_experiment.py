"""
Thí nghiệm Reducing Hallucinations với OpenAI GPT
Thay thế cho Gemini nếu không có API access
"""
import os, json, re
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher
from openai import OpenAI

# --- 1. CẤU HÌNH ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if not OPENAI_API_KEY:
    print("Không tìm thấy OPENAI_API_KEY trong file .env")
    exit()

client = OpenAI(api_key=OPENAI_API_KEY)

# --- 2. ĐỊNH NGHĨA PROMPT ---

def get_baseline_prompt(question):
    """Trả về prompt đơn giản"""
    return f"Hãy trả lời câu hỏi sau một cách ngắn gọn và chính xác: {question}"

def get_critique_prompt(question):
    """Trả về prompt tự phản biện 3 bước"""
    return f"""
Bạn là một trợ lý AI cẩn trọng, luôn kiểm tra lại thông tin.
Nhiệm vụ của bạn là trả lời câu hỏi sau bằng quy trình 3 bước.

Câu hỏi: {question}

---
[BẮT ĐẦU QUY TRÌNH]

**Bước 1: Câu trả lời ban đầu:**
[Hãy tạo câu trả lời ban đầu của bạn ở đây]

**Bước 2: Tự phản biện:**
[Hãy xem xét lại câu trả lời ở Bước 1. Nó có chính xác không? Có "hallucinate" điểm nào không? Có thể cải thiện ở đâu?]

**Bước 3: Câu trả lời cuối cùng (đã xác minh):**
[Dựa trên phản biện ở Bước 2, hãy đưa ra câu trả lời cuối cùng, chính xác nhất.]
"""

def extract_final_answer(critique_text):
    """Trích xuất câu trả lời cuối cùng từ Bước 3"""
    patterns = [
        r'\*\*Bước 3[:\s]+.*?\*\*[:\s]*(.*?)(?=\n\n|\Z)',
        r'Câu trả lời cuối cùng[:\s]*\*\*[:\s]*(.*?)(?=\n\n|\Z)',
        r'Bước 3[:\s]+(.*?)(?=\n\n|\Z)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, critique_text, re.DOTALL | re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            answer = answer.strip('*[](){}')
            return answer
    
    return critique_text.strip()

def calculate_similarity(text1, text2):
    """Tính độ tương đồng giữa 2 string (0-1)"""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def evaluate_answer(predicted, ground_truth, threshold=0.6):
    """Đánh giá câu trả lời"""
    if predicted.lower().strip() == ground_truth.lower().strip():
        return True, 1.0
    
    sim = calculate_similarity(predicted, ground_truth)
    is_correct = sim >= threshold
    
    return is_correct, sim

# --- 3. TẢI DATASET ---

DATA_DIR = Path("data")
if not (DATA_DIR / "questions.json").exists():
    print("Không tìm thấy file questions.json hoặc answers.json trong thư mục data/")
    exit()

all_questions = json.loads((DATA_DIR / "questions.json").read_text(encoding="utf-8"))
all_answers_ground_truth = json.loads((DATA_DIR / "answers.json").read_text(encoding="utf-8"))

NUM_SAMPLES = 10  # Bắt đầu với 10 mẫu

benchmark_data = []
for i in range(min(NUM_SAMPLES, len(all_questions))):
    benchmark_data.append({
        "id": i,
        "question": all_questions[i],
        "ground_truth": all_answers_ground_truth[i]
    })

print(f"Đã tải {len(benchmark_data)} câu hỏi từ ViQuAD để làm benchmark.")
print(f"Bắt đầu chạy thí nghiệm so sánh Baseline vs Self-Critique...\n")

# --- 4. CHẠY THÍ NGHIỆM ---

results = []

for item in tqdm(benchmark_data, desc="Đang chạy thí nghiệm"):
    q = item["question"]
    gt = item["ground_truth"]
    
    # 1. Chạy Baseline
    try:
        response_bl = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": get_baseline_prompt(q)}],
            temperature=0
        )
        answer_bl = response_bl.choices[0].message.content.strip()
    except Exception as e:
        print(f"\nLỗi khi chạy Baseline câu {item['id']}: {e}")
        answer_bl = f"[LỖI: {e}]"

    # 2. Chạy Self-Critique
    try:
        response_sc = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": get_critique_prompt(q)}],
            temperature=0
        )
        answer_sc_full = response_sc.choices[0].message.content.strip()
        answer_sc_final = extract_final_answer(answer_sc_full)
    except Exception as e:
        print(f"\nLỗi khi chạy Self-Critique câu {item['id']}: {e}")
        answer_sc_full = f"[LỖI: {e}]"
        answer_sc_final = f"[LỖI: {e}]"

    # 3. Đánh giá
    bl_correct, bl_sim = evaluate_answer(answer_bl, gt)
    sc_correct, sc_sim = evaluate_answer(answer_sc_final, gt)

    # 4. Lưu kết quả
    results.append({
        "id": item["id"],
        "question": q,
        "ground_truth": gt,
        "baseline_answer": answer_bl,
        "baseline_correct": bl_correct,
        "baseline_similarity": bl_sim,
        "critique_answer_full": answer_sc_full,
        "critique_answer_final": answer_sc_final,
        "critique_correct": sc_correct,
        "critique_similarity": sc_sim
    })

# --- 5. PHÂN TÍCH KẾT QUẢ ---
df_results = pd.DataFrame(results)

baseline_accuracy = df_results['baseline_correct'].sum() / len(df_results) * 100
critique_accuracy = df_results['critique_correct'].sum() / len(df_results) * 100

baseline_avg_similarity = df_results['baseline_similarity'].mean()
critique_avg_similarity = df_results['critique_similarity'].mean()

accuracy_improvement = critique_accuracy - baseline_accuracy
relative_improvement = (accuracy_improvement / baseline_accuracy * 100) if baseline_accuracy > 0 else 0

sc_better = (df_results['critique_similarity'] > df_results['baseline_similarity']).sum()
bl_better = (df_results['baseline_similarity'] > df_results['critique_similarity']).sum()
equal = (df_results['baseline_similarity'] == df_results['critique_similarity']).sum()

# --- 6. LƯU KẾT QUẢ ---
output_file = "experiment_results_openai.csv"
df_results.to_csv(output_file, index=False, encoding="utf-8-sig")

# --- 7. IN BÁO CÁO ---
print("\n" + "="*70)
print("📊 KẾT QUẢ THÍ NGHIỆM: REDUCING HALLUCINATIONS VỚI SELF-CRITIQUE")
print("="*70)
print(f"\n📌 Thông tin thí nghiệm:")
print(f"   • Số câu hỏi test: {len(df_results)}")
print(f"   • Dataset: ViQuAD")
print(f"   • Model: GPT-4o-mini (OpenAI)")
print(f"   • Similarity threshold: 0.6 (60%)")

print(f"\n📈 KẾT QUẢ CHÍNH:")
print(f"\n   1️⃣  BASELINE (Direct Prompt):")
print(f"      • Accuracy: {baseline_accuracy:.2f}%")
print(f"      • Average Similarity: {baseline_avg_similarity:.4f}")
print(f"      • Số câu trả lời đúng: {df_results['baseline_correct'].sum()}/{len(df_results)}")

print(f"\n   2️⃣  SELF-CRITIQUE (3-Step Prompt):")
print(f"      • Accuracy: {critique_accuracy:.2f}%")
print(f"      • Average Similarity: {critique_avg_similarity:.4f}")
print(f"      • Số câu trả lời đúng: {df_results['critique_correct'].sum()}/{len(df_results)}")

print(f"\n   ✨ IMPROVEMENT:")
print(f"      • Accuracy Improvement: {accuracy_improvement:+.2f}% (absolute)")
print(f"      • Relative Improvement: {relative_improvement:+.2f}%")
print(f"      • Similarity Improvement: {critique_avg_similarity - baseline_avg_similarity:+.4f}")

print(f"\n   📊 So sánh từng câu:")
print(f"      • Self-Critique tốt hơn: {sc_better} câu ({sc_better/len(df_results)*100:.1f}%)")
print(f"      • Baseline tốt hơn: {bl_better} câu ({bl_better/len(df_results)*100:.1f}%)")
print(f"      • Bằng nhau: {equal} câu ({equal/len(df_results)*100:.1f}%)")

print(f"\n💾 Chi tiết đầy đủ đã được lưu vào: {output_file}")
print("="*70)

print("\n✅ HOÀN TẤT THÍ NGHIỆM!")

