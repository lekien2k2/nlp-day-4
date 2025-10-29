import google.generativeai as genai
import os, json, re
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher # Để đánh giá độ tương đồng
import glob # Để tìm file benchmark

# --- 1. CẤU HÌNH ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

if not GEMINI_API_KEY:
    print("Không tìm thấy GEMINI_API_KEY trong file .env")
    exit()

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Lỗi cấu hình Gemini: {e}")
    exit()

# Cấu hình model (dùng cho cả 2 prompt)
config = genai.types.GenerationConfig(temperature=0.0)

# Safety settings - (Giữ nguyên format đúng của bạn)
safety = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

print("Đang khởi tạo model 'gemini-1.5-pro-latest'...")
model = genai.GenerativeModel(
    'gemini-1.5-pro-latest', # THAY ĐỔI: Dùng 1.5 Pro cho mạnh mẽ hơn
    generation_config=config,
    safety_settings=safety
)
print("Model đã sẵn sàng.")

# --- 2. CÁC HÀM HELPERS (Giữ nguyên logic của bạn) ---

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
    """
    Trích xuất câu trả lời cuối cùng từ Bước 3 trong self-critique output
    """
    # Regex tìm "Bước 3" (hoặc biến thể) và lấy nội dung sau nó
    patterns = [
        # Match 'Bước 3: [nội dung]'
        r'\*\*Bước 3[:\s]+.*?\*\*[:\s]*(.*?)(?=\n\n|\n\*\*\s*Bước|\Z)', 
        # Match 'Câu trả lời cuối cùng: [nội dung]'
        r'Câu trả lời cuối cùng[:\s]*\*\*[:\s]*(.*?)(?=\n\n|\n\*\*\s*Bước|\Z)',
        # Match 'Bước 3' không có **
        r'Bước 3[:\s]+(.*?)(?=\n\n|\nBước|\Z)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, critique_text, re.DOTALL | re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Loại bỏ các ký tự đặc biệt đầu/cuối
            answer = answer.strip('*[](){}_- ')
            if answer: # Đảm bảo không phải string rỗng
                return answer
    
    # Fallback: Nếu không tìm thấy, cố gắng lấy dòng cuối cùng
    last_line = critique_text.split('\n')[-1].strip().strip('*[](){}_- ')
    if last_line:
        return last_line
        
    return critique_text.strip() # Fallback cuối cùng

def calculate_similarity(text1, text2):
    """
    Tính độ tương đồng giữa 2 string (0-1)
    """
    return SequenceMatcher(None, str(text1).lower(), str(text2).lower()).ratio()

def evaluate_answer(predicted, ground_truth, threshold=0.6):
    """
    Đánh giá câu trả lời dự đoán so với ground truth
    Returns: (is_correct, similarity_score)
    """
    # Xử lý trường hợp predicted hoặc ground_truth là None
    predicted_str = str(predicted).lower().strip()
    ground_truth_str = str(ground_truth).lower().strip()

    if not predicted_str or not ground_truth_str:
        return False, 0.0

    # Exact match (case-insensitive)
    if predicted_str == ground_truth_str:
        return True, 1.0
    
    # Partial match với similarity
    sim = calculate_similarity(predicted_str, ground_truth_str)
    is_correct = sim >= threshold
    
    return is_correct, sim

# --- 3. HÀM CHẠY THÍ NGHIỆM CHÍNH ---

def run_and_evaluate_dataset(benchmark_path, similarity_threshold=0.6):
    """
    Hàm chính: Đọc 1 file benchmark, chạy, đánh giá, và in báo cáo.
    """
    
    dataset_name = benchmark_path.stem.replace("benchmark_", "")
    print("\n" + "="*70)
    print(f"📊 BẮT ĐẦU THÍ NGHIỆM VỚI DATASET: {dataset_name.upper()}")
    print("="*70)

    # Đọc benchmark CSV
    try:
        benchmark_df = pd.read_csv(benchmark_path)
    except Exception as e:
        print(f"❌ Lỗi khi đọc file {benchmark_path}: {e}")
        return

    results = [] # Nơi lưu trữ tất cả kết quả
    
    # Chạy qua từng hàng trong file benchmark
    for _, item in tqdm(benchmark_df.iterrows(), total=len(benchmark_df), desc=f"   -> Đang chạy {dataset_name}"):
        q = item["question"]
        gt = item["ground_truth"]
        
        # 1. Chạy Baseline
        try:
            prompt_bl = get_baseline_prompt(q)
            response_bl = model.generate_content(prompt_bl)
            answer_bl = response_bl.text.strip()
        except Exception as e:
            answer_bl = f"[LỖI: {e}]"

        # 2. Chạy Self-Critique
        try:
            prompt_sc = get_critique_prompt(q)
            response_sc = model.generate_content(prompt_sc)
            answer_sc_full = response_sc.text.strip() # Toàn bộ output
            answer_sc_final = extract_final_answer(answer_sc_full) # Chỉ lấy câu trả lời cuối
        except Exception as e:
            answer_sc_full = f"[LỖI: {e}]"
            answer_sc_final = f"[LỖI: {e}]"

        # 3. Đánh giá cả 2 phương pháp
        bl_correct, bl_sim = evaluate_answer(answer_bl, gt, similarity_threshold)
        sc_correct, sc_sim = evaluate_answer(answer_sc_final, gt, similarity_threshold)

        # 4. Lưu kết quả
        results.append({
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

    # --- 5. PHÂN TÍCH KẾT QUẢ (CHO DATASET NÀY) ---
    df_results = pd.DataFrame(results)

    # Tính toán các metrics
    baseline_accuracy = df_results['baseline_correct'].sum() / len(df_results) * 100
    critique_accuracy = df_results['critique_correct'].sum() / len(df_results) * 100

    baseline_avg_similarity = df_results['baseline_similarity'].mean()
    critique_avg_similarity = df_results['critique_similarity'].mean()

    # Cải thiện
    accuracy_improvement = critique_accuracy - baseline_accuracy
    relative_improvement = (accuracy_improvement / baseline_accuracy * 100) if baseline_accuracy > 0 else (100.0 if accuracy_improvement > 0 else 0.0)

    # Đếm số câu
    sc_better = (df_results['critique_similarity'] > df_results['baseline_similarity']).sum()
    bl_better = (df_results['baseline_similarity'] > df_results['critique_similarity']).sum()
    equal = (df_results['baseline_similarity'] == df_results['critique_similarity']).sum()

    # --- 6. LƯU KẾT QUẢ RA FILE (CHO DATASET NÀY) ---
    output_csv_file = Path("results") / f"results_{dataset_name}.csv"
    output_txt_file = Path("results") / f"summary_{dataset_name}.txt"
    
    df_results.to_csv(output_csv_file, index=False, encoding="utf-8-sig")

    # --- 7. TẠO VÀ IN BÁO CÁO (CHO DATASET NÀY) ---
    summary_report = f"""
=== BÁO CÁO NGHIÊN CỨU: REDUCING HALLUCINATIONS ===

** 1. RESEARCH QUESTION **
Liệu kỹ thuật Self-Critique prompting có giảm hallucination và cải thiện 
factual accuracy so với direct prompting không?

** 2. METHODOLOGY **
- Dataset: {dataset_name.upper()} ({len(df_results)} câu hỏi)
- Model: Gemini 1.5 Pro
- Baseline: Direct prompt đơn giản
- Treatment: Self-Critique 3-step prompt (Initial Answer → Critique → Final Answer)
- Evaluation: Similarity score với ground truth (threshold = {similarity_threshold})

** 3. RESULTS **

Baseline Accuracy:       {baseline_accuracy:.2f}%
Self-Critique Accuracy:  {critique_accuracy:.2f}%
Improvement:             {accuracy_improvement:+.2f}% (absolute)
Relative Improvement:    {relative_improvement:+.2f}%

Average Similarity:
  - Baseline:        {baseline_avg_similarity:.4f}
  - Self-Critique: {critique_avg_similarity:.4f}
  - Difference:      {critique_avg_similarity - baseline_avg_similarity:+.4f}

** 4. CONCLUSION **
"""

    if accuracy_improvement > 0:
        summary_report += f"""
Self-Critique prompting ĐÃ THÀNH CÔNG trong việc giảm hallucination, 
cải thiện accuracy {accuracy_improvement:.2f}% so với baseline.
Kỹ thuật này cho thấy tiềm năng trong việc tăng độ tin cậy của LLM responses.
"""
    elif accuracy_improvement == 0:
        summary_report += f"""
Self-Critique prompting KHÔNG CHO THẤY SỰ KHÁC BIỆT so với baseline.
Cần xem xét kỹ hơn các trường hợp cụ thể.
"""
    else:
        summary_report += f"""
Self-Critique prompting cho kết quả KÉM HƠN baseline ({accuracy_improvement:.2f}%) trong thí nghiệm này.
Có thể prompt design chưa tối ưu hoặc model gặp khó khăn trong việc tự sửa lỗi.
"""

    summary_report += f"""

** 5. DETAILED BREAKDOWN **
Self-Critique performs better: {sc_better} cases ({sc_better/len(df_results)*100:.1f}%)
Baseline performs better:      {bl_better} cases ({bl_better/len(df_results)*100:.1f}%)
Equal performance:           {equal} cases ({equal/len(df_results)*100:.1f}%)

===================================================
"""

    # Lưu summary report
    with open(output_txt_file, "w", encoding="utf-8") as f:
        f.write(summary_report)

    # In báo cáo ra console
    print(summary_report)
    print(f"💾 Chi tiết đầy đủ đã được lưu vào: {output_csv_file}")
    print(f"📄 Báo cáo tóm tắt đã được lưu vào: {output_txt_file}")


# --- 4. HÀM MAIN ĐỂ CHẠY TẤT CẢ DATASET ---
if __name__ == "__main__":
    
    # Tạo thư mục data/ và results/ nếu chưa có
    Path("data").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    # Tìm tất cả các file benchmark đã được chuẩn bị
    benchmark_files = list(Path("data").glob("benchmark_*.csv"))
    
    if not benchmark_files:
        print("⚠️ Không tìm thấy file benchmark nào trong thư mục 'data/'.")
        print("➡️  Vui lòng chạy 'python app/prepare_data.py' trước tiên.")
        exit()
        
    print(f"Tìm thấy {len(benchmark_files)} bộ dataset benchmark để chạy:")
    for f in benchmark_files:
        print(f"  - {f.name}")

    # Lặp qua từng file benchmark và chạy thí nghiệm
    for benchmark_path in benchmark_files:
        run_and_evaluate_dataset(benchmark_path, similarity_threshold=0.6)

    print("\n🎉🎉🎉 Tất cả 5 thí nghiệm đã hoàn tất! 🎉🎉🎉")
    print("Kiểm tra thư mục 'results/' để xem 5 file CSV và 5 file TXT báo cáo.")
