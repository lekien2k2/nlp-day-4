# app/prepare_data.py
import sys, io
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# (tùy chọn) fix UTF-8 cho Windows console
try:
    sys.stdout = io.TextIWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

# --- THAY ĐỔI CHÍNH Ở ĐÂY ---
# Cấu hình các dataset tiếng Việt cần tải
DATASET_CONFIG = {
    "viquad_v2_train": {
        "path": "taidng/UIT-ViQuAD2.0",
        "config": None, # Không có config cụ thể
        "split": "train", # Chỉ có split 'train' (test không có answers)
        "q_col": "question",
        "a_col": "answers.text[0]" # Lấy câu trả lời đầu tiên trong list
    },
    "xquad_vi": {
        "path": "xquad",
        "config": "xquad.vi", # Vietnamese config
        "split": "validation",
        "q_col": "question",
        "a_col": "answers.text[0]"
    },
    "mlqa_vi": {
        "path": "mlqa",
        "config": "mlqa-translate-train.vi", # Vietnamese translated train set
        "split": "train",
        "q_col": "question",
        "a_col": "answers.text[0]"
    },
    "mlqa_vi_test": {
        "path": "mlqa",
        "config": "mlqa.vi.vi", # Vietnamese test set (context + question in Vietnamese)
        "split": "test",
        "q_col": "question",
        "a_col": "answers.text[0]"
    }
}
# --- KẾT THÚC THAY ĐỔI ---

# Số lượng mẫu tối đa để lấy từ mỗi dataset
MAX_SAMPLES_PER_DATASET = 1000

def get_nested_value(item, key):
    """
    Hàm helper để lấy giá trị từ key lồng nhau (ví dụ: 'answer.value')
    """
    keys = key.split('.')
    value = item
    for k in keys:
        if '[' in k and ']' in k: # Xử lý trường hợp list index (ví dụ: text[0])
            list_name, index_str = k.split('[')
            index = int(index_str.replace(']', ''))
            value = value.get(list_name)
            if value and isinstance(value, list) and len(value) > index:
                value = value[index]
            else:
                return None # Không tìm thấy list hoặc index
        else:
            value = value.get(k)
        
        if value is None:
            return None
    return value

def main():
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Chuẩn bị tải {len(DATASET_CONFIG)} bộ dataset Tiếng Việt. Lấy tối đa {MAX_SAMPLES_PER_DATASET} mẫu/dataset.")
    
    for name, config in DATASET_CONFIG.items():
        print(f"\n📥 Đang tải {name} ({config['path']})...")
        try:
            # Lấy thêm một chút để bù trừ cho impossible questions
            samples_to_load = int(MAX_SAMPLES_PER_DATASET * 1.5)
            split_str = f"{config['split']}[:{samples_to_load}]"
            
            ds = load_dataset(
                config['path'], 
                config.get('config'), # config=None nếu không có
                split=split_str,
                trust_remote_code=True  # Cho phép custom code (cần cho một số datasets)
            )
            
            rows = []
            for item in tqdm(ds, desc=f"   -> Xử lý {name}"):
                # Skip impossible questions (cho ViQuAD)
                if item.get('is_impossible') == True:
                    continue
                
                q = get_nested_value(item, config['q_col'])
                a = get_nested_value(item, config['a_col'])
                
                # Đảm bảo q và a là string và không rỗng
                if isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip():
                    rows.append({"question": q.strip(), "ground_truth": a.strip()})
                
                # Dừng sớm nếu đã đủ số lượng mẫu
                if len(rows) >= MAX_SAMPLES_PER_DATASET:
                    break
            
            if not rows:
                print(f"⚠️ Không thu được dòng nào cho {name}.")
                continue
                
            df = pd.DataFrame(rows)
            df.drop_duplicates(subset=["question"], inplace=True)
            
            # Đảm bảo chúng ta chỉ lấy đúng MAX_SAMPLES sau khi lọc trùng
            if len(df) > MAX_SAMPLES_PER_DATASET:
                df = df.head(MAX_SAMPLES_PER_DATASET)
            
            out_path = out_dir / f"benchmark_{name}.csv"
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"✅ Đã lưu {len(df)} mẫu vào {out_path}")
            
        except Exception as e:
            print(f"❌ Lỗi khi tải hoặc xử lý {name}: {e}")

    print("\n--- Tải và chuẩn bị dữ liệu tiếng Việt hoàn tất! ---")

if __name__ == "__main__":
    main()