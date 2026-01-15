import os
import pandas as pd
import numpy as np

# =============================================================================
# CẤU HÌNH: CHỈ XÓA 6 CỘT ĐỊNH DANH (ĐỂ GIỮ LẠI ĐÚNG 79 CỘT)
# =============================================================================

# Danh sách 6 cột cần loại bỏ (Identifier)
# LƯU Ý: Đã loại 'Destination Port' ra khỏi danh sách xóa để giữ lại nó.
COLS_TO_DROP = [
    'Flow ID', 
    'Source IP', 
    'Source Port', 
    'Destination IP', 
    'Protocol', 
    'Timestamp'
]

# =============================================================================
# HÀM XỬ LÝ
# =============================================================================
def process_file_16_final(input_path, output_path):
    print(f"Dataset Target: 79 cột - Đang đọc file: {input_path}...")
    
    # Đọc file CSV (xử lý tên cột bị dính khoảng trắng)
    df = pd.read_csv(input_path, skipinitialspace=True, low_memory=False)
    
    # 1. Chuẩn hóa tên cột: Bỏ khoảng trắng đầu/cuối
    df.columns = df.columns.str.strip()
    
    # 2. XÓA 6 CỘT ĐỊNH DANH
    print(f"Đang xóa {len(COLS_TO_DROP)} cột định danh thừa...")
    # Chỉ xóa nếu cột đó có trong file
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    df.drop(columns=cols_present, inplace=True)
    
    # 3. CHUẨN HÓA NHÃN (LABEL)
    print("Đang chuẩn hóa nhãn (BENIGN -> Benign)...")
    if 'Label' in df.columns:
        # Chuyển thành chuỗi và cắt khoảng trắng
        df['Label'] = df['Label'].astype(str).str.strip()
        
        # Map về chuẩn
        mapping = {
            'BENIGN': 'Benign', 
            'DDoS': 'DDoS'
        }
        # Áp dụng map, giữ nguyên nếu không nằm trong danh sách map
        df['Label'] = df['Label'].apply(lambda x: mapping.get(x, x))
    
    # 4. KIỂM TRA KẾT QUẢ
    current_cols = df.shape[1]
    print(f" -> Số cột sau khi xử lý: {current_cols}")
    
    if current_cols == 79:
        print("THÀNH CÔNG! File đã có đúng 79 cột.")
    else:
        print(f"Cảnh báo: Số cột là {current_cols} (Khác 79). Hãy kiểm tra lại header file gốc.")
        # In ra các cột hiện tại để bạn debug nếu cần
        # print(df.columns.tolist())

    # 5. LƯU FILE
    print(f"Đang lưu file: {output_path}")
    df.to_csv(output_path, index=False)
    print("✓ Hoàn tất.")

# =============================================================================
# CHẠY CODE
# =============================================================================
# Sử dụng r"..." để tránh lỗi đường dẫn
# 1. Lấy vị trí thực tế của file code này (đang ở src/utils)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Tìm về thư mục gốc dự án (lùi 2 cấp: src/utils -> src -> Project_Root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

# 3. Cấu hình INPUT: Trỏ vào thư mục chứa dữ liệu test rời (data/raw)
# Lưu ý: Bạn cần copy file Friday...csv vào thư mục 'data/raw' của dự án trước
INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "Cross_Validate",  "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

# 4. Cấu hình OUTPUT: Lưu kết quả vào thư mục đã xử lý (data/processed)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "Cross_Validate")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "File16_Final_79cols.csv")

# --- AN TOÀN: Tự động tạo thư mục output nếu chưa có ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Đã tạo thư mục mới: {OUTPUT_DIR}")

# --- DEBUG: Kiểm tra xem tìm thấy file không ---
print(f"Đang tìm file Input tại: {INPUT_CSV}")
if os.path.exists(INPUT_CSV):
    print("Đã tìm thấy file Input.")
else:
    print("KHÔNG tìm thấy file Input.")
    print(f"Hãy copy file 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv' vào thư mục: {os.path.join(PROJECT_ROOT, 'data', 'external')}")
    
if __name__ == "__main__":
    process_file_16_final(INPUT_CSV, OUTPUT_CSV)