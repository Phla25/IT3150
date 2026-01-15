import pandas as pd
import numpy as np
import os
import gc

# =============================================================================
# CẤU HÌNH ĐƯỜNG DẪN TỰ ĐỘNG (AUTO PATH CONFIG)
# =============================================================================

# 1. Lấy vị trí thực tế của file code này (đang nằm trong src/utils/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Tìm về thư mục gốc dự án (Đi ngược ra 2 cấp: utils -> src -> Root)
# Lưu ý: Nếu file này bạn để ở chỗ khác, hãy chỉnh số lần os.path.dirname
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

# 3. Nối đường dẫn từ gốc vào thư mục data
# os.path.join giúp code chạy đúng cả trên Windows (\) và Mac/Linux (/)
FILE_2017 = os.path.join(PROJECT_ROOT, "data", "raw", "CIC_IDS_2017_Final_Merged.csv")
FILE_2018 = os.path.join(PROJECT_ROOT, "data", "raw", "CIC-IDS-2018_COMBINED_ATTACKS_ONLY.csv")
FILE_UNSW = os.path.join(PROJECT_ROOT, "data", "raw", "CIC_UNSW_NB15.csv")

# Đường dẫn file đầu ra
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FILE_OUTPUT = os.path.join(OUTPUT_DIR, "MASTER_DATASET_FINAL_ALL_V4.csv")

# Tự động tạo thư mục 'processed' nếu chưa có (tránh lỗi khi lưu file)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- DEBUG: IN RA ĐỂ KIỂM TRA ---
print("-" * 60)
print(f"📍 Đang chạy tại: {CURRENT_DIR}")
print(f"🏠 Gốc dự án:     {PROJECT_ROOT}")
print(f"📂 File 2017:     {FILE_2017}")
print("-" * 60)

# Kiểm tra xem file có tồn tại không trước khi đọc
if not os.path.exists(FILE_2017):
    print("❌ LỖI: Không tìm thấy file 2017. Hãy kiểm tra lại tên file trong thư mục data/raw!")
    # exit() # Có thể mở dòng này để dừng chương trình nếu muốn
else:
    print("✅ Đã tìm thấy file dữ liệu đầu vào.")

# ==============================================================================
# 2. KHUÔN MẪU CHUẨN (CIC 2017)
# ==============================================================================
TARGET_COLUMNS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean",
    "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max",
    "Idle Min", "Label"
]

# ==============================================================================
# 3. MAPPING (ĐÃ BỔ SUNG UNSW)
# ==============================================================================
# Ánh xạ tên cột UNSW -> 2017
MAP_UNSW_COLS = {
    "Total Fwd Packet": "Total Fwd Packets",
    "Total Bwd packets": "Total Backward Packets",
    "Total Length of Fwd Packet": "Total Length of Fwd Packets",
    "Total Length of Bwd Packet": "Total Length of Bwd Packets",
    "Packet Length Min": "Min Packet Length",
    "Packet Length Max": "Max Packet Length",
    "CWR Flag Count": "CWE Flag Count",
    "Fwd Segment Size Avg": "Avg Fwd Segment Size",
    "Bwd Segment Size Avg": "Avg Bwd Segment Size",
    "Fwd Bytes/Bulk Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Packet/Bulk Avg": "Fwd Avg Packets/Bulk",
    "Fwd Bulk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Bytes/Bulk Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Packet/Bulk Avg": "Bwd Avg Packets/Bulk",
    "Bwd Bulk Rate Avg": "Bwd Avg Bulk Rate",
    "FWD Init Win Bytes": "Init_Win_bytes_forward",
    "Bwd Init Win Bytes": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward"
}

# Ánh xạ tên cột 2018 -> 2017
MAP_2018_COLS = {
    "Bwd IAT Tot": "Bwd IAT Total"
}

# --- [QUAN TRỌNG] TỪ ĐIỂN ÁNH XẠ NHÃN TOÀN DIỆN ---
LABEL_MAPPING = {
    # === UNSW-NB15 ===
    "Normal": "Benign",
    "Generic": "Infiltration",     # Generic: Block cipher attacks -> Xâm nhập
    "Exploits": "Web Attack",      # Exploits: Thường là lỗ hổng web/app
    "Fuzzers": "Web Attack",       # Fuzzers: Tấn công tìm lỗ hổng
    "DoS": "DoS",                  # Giữ nguyên
    "Reconnaissance": "PortScan",  # Do thám -> PortScan
    "Analysis": "PortScan",        # Analysis (Port scan, spam) -> PortScan
    "Backdoor": "Infiltration",    # Cửa hậu -> Xâm nhập
    "Shellcode": "Infiltration",   # Mã độc thực thi -> Xâm nhập
    "Worms": "Bot",                # Sâu máy tính -> Botnet
    
    # === CIC-IDS-2018 ===
    "Benign": "Benign",
    "FTP-BruteForce": "BruteForce",
    "SSH-Bruteforce": "BruteForce",
    "DoS attacks-GoldenEye": "DoS",
    "DoS attacks-Slowloris": "DoS",
    "DoS attacks-SlowHTTPTest": "DoS",
    "DoS attacks-Hulk": "DoS",
    "DDoS attacks-LOIC-HTTP": "DDoS",
    "DDOS attack-HOIC": "DDoS",
    "DDOS attack-LOIC-UDP": "DDoS",
    "Brute Force -Web": "Web Attack",
    "Brute Force -XSS": "Web Attack",
    "SQL Injection": "Web Attack",
    "Infilteration": "Infiltration",
    "Bot": "Bot",
    
    # === CIC-IDS-2017 (Chuẩn hóa chính tả) ===
    "BENIGN": "Benign",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Web Attack  Brute Force": "Web Attack",
    "Web Attack – Brute Force": "Web Attack",
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack  XSS": "Web Attack",
    "Web Attack – XSS": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack  Sql Injection": "Web Attack", 
    "Web Attack – Sql Injection": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack",
    "Heartbleed": "Heartbleed" # Giữ nguyên hoặc gộp vào DoS tùy bạn
}

# ==============================================================================
# 4. HÀM XỬ LÝ
# ==============================================================================
def process_and_merge():
    print(">>> BẮT ĐẦU GỘP 3 DATASET (V4 - FIX UNSW LABELS) <<<")
    dfs = []

    # --- 1. CIC 2017 ---
    print(f"\n1. Đọc CIC-IDS-2017...")
    if os.path.exists(FILE_2017):
        df = pd.read_csv(FILE_2017, low_memory=False)
        df.columns = df.columns.str.strip()
        # Bổ sung cột thiếu
        for col in TARGET_COLUMNS:
            if col not in df.columns: df[col] = 0
        df = df[TARGET_COLUMNS]
        dfs.append(df)
        print(f"   -> OK. Rows: {len(df)}")
    else: print("   [MISSING] File 2017")

    # --- 2. CIC 2018 ---
    print(f"\n2. Đọc CIC-IDS-2018...")
    if os.path.exists(FILE_2018):
        df = pd.read_csv(FILE_2018, low_memory=False)
        df.columns = df.columns.str.strip()
        df.rename(columns=MAP_2018_COLS, inplace=True)
        
        df_clean = pd.DataFrame()
        for col in TARGET_COLUMNS:
            if col in df.columns: df_clean[col] = df[col]
            else: df_clean[col] = 0
        dfs.append(df_clean)
        print(f"   -> OK. Rows: {len(df_clean)}")
        del df
    else: print("   [MISSING] File 2018")

    # --- 3. UNSW NB15 ---
    print(f"\n3. Đọc UNSW-NB15...")
    if os.path.exists(FILE_UNSW):
        df = pd.read_csv(FILE_UNSW, low_memory=False)
        df.columns = df.columns.str.strip()
        df.rename(columns=MAP_UNSW_COLS, inplace=True)
        
        df_clean = pd.DataFrame()
        for col in TARGET_COLUMNS:
            if col in df.columns: df_clean[col] = df[col]
            elif col == "Destination Port": df_clean[col] = -1 # Điền -1
            else: df_clean[col] = 0
            
        dfs.append(df_clean)
        print(f"   -> OK. Rows: {len(df_clean)}")
        del df
    else: print("   [MISSING] File UNSW")

    # --- GỘP ---
    if dfs:
        print("\n4. Đang gộp và chuẩn hóa nhãn...")
        master = pd.concat(dfs, ignore_index=True)
        
        # CHUẨN HÓA NHÃN
        master['Label'] = master['Label'].astype(str).str.strip()
        master['Label'] = master['Label'].replace(LABEL_MAPPING)
        
        print(f"5. Đang lưu file: {FILE_OUTPUT}")
        master.to_csv(FILE_OUTPUT, index=False)
        
        print(f"\n>>> HOÀN TẤT! Tổng số dòng: {len(master)}")
        print("Phân bố nhãn cuối cùng:")
        print(master['Label'].value_counts())
    else:
        print("Không có dữ liệu.")

if __name__ == "__main__":
    process_and_merge()