# High-Performance NIDS using Adaptive Gaussian Naive Bayes

> **Hệ thống Phát hiện Xâm nhập Mạng (NIDS) hiệu năng cao sử dụng thuật toán Global Empirical MAP**

## Giới thiệu (Overview)

Dự án này tập trung giải quyết các hạn chế cốt lõi của các thuật toán xác suất truyền thống (như **Maximum Likelihood Estimation - MLE**) trong bài toán an ninh mạng, cụ thể là lỗi **"Zero-frequency problem"** (Điểm kỳ dị) và hiện tượng **Overfitting** khi gặp dữ liệu thưa.

Thay vì sử dụng các thư viện có sẵn (như `scikit-learn` với tham số làm trơn cố định), dự án đề xuất và hiện thực hóa thuật toán **Global Empirical MAP (Maximum A Posteriori)**. Thuật toán này sử dụng tham số làm trơn thích ứng (Adaptive Smoothing), được chứng minh toán học giúp hệ thống hoạt động ổn định trên dữ liệu lớn và giảm thiểu báo động giả (False Positives).

## Tính năng nổi bật (Key Features)

* **Thuật toán Global Empirical MAP:** Tự cài đặt (Custom Implementation) thuật toán Naive Bayes với cơ chế làm trơn dựa trên phương sai toàn cục, khắc phục lỗi chia cho 0.
* **Chứng minh Toán học:** Sử dụng **Khai triển Taylor** để tối ưu hóa tham số  tỉ lệ nghịch với kích thước mẫu ().
* **Master Dataset:** Tổng hợp và chuẩn hóa 3 bộ dữ liệu lớn nhất hiện nay (CIC-IDS-2017, CIC-IDS-2018, UNSW-NB15) với hơn 2.5 triệu mẫu.
* **Semantic Label Mapping:** Đồng bộ hóa không gian nhãn giữa các bộ dữ liệu khác nhau (Ví dụ: Gộp *Exploits*, *Fuzzers* thành *Web Attack*).
* **Giảm Báo động giả (False Positives):** Hiệu quả vượt trội trong việc nhận diện các mẫu dữ liệu chưa từng biết (Zero-day anomalies).

---

## Cơ sở Lý thuyết & Thuật toán

### 1. Vấn đề của MLE (Maximum Likelihood Estimation)

Trong các mô hình Naive Bayes truyền thống, phương sai được ước lượng bằng công thức MLE. Khi một đặc trưng có phương sai mẫu bằng 0 (do dữ liệu thưa hoặc bị tấn công), hàm mật độ xác suất sẽ tiến tới vô cùng, gây ra lỗi tính toán (Singularities).

### 2. Giải pháp MAP (Maximum A Posteriori)

Sử dụng ước lượng MAP với phân phối tiên nghiệm **Inverse-Gamma** . Dựa trên **Khai triển Taylor bậc 1**, tôi đã chứng minh được tham số làm trơn tối ưu () cần tuân theo quy luật:

$$\epsilon \approx \frac{2\beta}{N} \propto \frac{1}{N}$$

**Công thức cài đặt thực nghiệm:**

$$\epsilon_{ideal} = \frac{Mean(Var(X))}{N}$$

---

## Cấu trúc Dữ liệu (Dataset)

Dự án sử dụng bộ **Master Dataset** được xây dựng từ quy trình Data Engineering nghiêm ngặt:

| Bộ dữ liệu gốc | Vai trò | Số lượng mẫu (Approx) |
| --- | --- | --- |
| **CIC-IDS-2017** | Nền tảng chính | ~1.3 triệu |
| **CIC-IDS-2018** | Bổ sung lớp tấn công hiện đại | ~1 triệu |
| **UNSW-NB15** | Bổ sung các lớp hiếm (Web Attack) | ~200k |

**Quy trình xử lý:**

1. **Cleaning:** Loại bỏ các cột định danh (IP, Timestamp, Flow ID).
2. **Filtering:** Loại bỏ các cột có phương sai bằng 0 (Zero Variance).
3. **Mapping:** Ánh xạ nhãn về 8 lớp chuẩn: *Benign, DoS, DDoS, PortScan, Bot, Web Attack, Infiltration, BruteForce*.

---

## Cài đặt & Sử dụng (Installation & Usage)

### Yêu cầu hệ thống

* Python 3.8+
* Jupyter Notebook
* RAM: Tối thiểu 8GB (Do xử lý dataset lớn)

### Cài đặt thư viện

```bash
pip install numpy pandas scikit-learn matplotlib seaborn

```

### Hướng dẫn chạy

1. **Chuẩn bị dữ liệu:**
Chạy file `CleaningData.py` (hoặc Notebook tương ứng) để tạo ra file `MASTER_DATASET_FINAL.csv`.
2. **Huấn luyện & Đánh giá:**
Mở `MAP-MLE-Scikit.ipynb` và chạy lần lượt các cells:
* Cell 1-2: Load thư viện và Dữ liệu.
* Cell 3-4: Định nghĩa Class `GaussianNB_MLE` và `GaussianNB_MAP`.
* Cell 5: Chạy kịch bản so sánh (Hold-out / Cross-validation).
* Cell Final: Trực quan hóa kết quả.



---

## Kết quả Thực nghiệm (Results)

Dưới đây là bảng so sánh hiệu năng giữa thuật toán đề xuất (MAP) và các phương pháp khác trên tập kiểm thử (Hold-out 20%):

| Mô hình | Macro F1-Score | Accuracy | Training Time |
| --- | --- | --- | --- |
| **GNB-MLE (Baseline)** | 0.6822 | 76.21% | ~6.5s |
| **Scikit-learn (Benchmark)** | 0.6906 | 76.55% | ~6.4s |
| **GNB-MAP (Proposed)** | **0.6931** | **76.87%** | **~6.5s** |

### Điểm nhấn (Highlights):

* **Cải thiện đáng kể** độ chính xác cho các lớp tấn công khó: **Infiltration (+3.0%)** và **PortScan (+3.7%)**.
* **Robustness:** Trong kịch bản kiểm thử chéo trên bộ dữ liệu lạ (*Friday-DDoS*), mô hình MAP đã giảm thiểu được **351 cảnh báo giả (False Positives)** so với MLE.
* **Zero-overhead:** Không làm tăng chi phí tính toán so với thuật toán gốc.

---

## Biểu đồ (Visualization)
![Biểu đồ so sánh](confusion_matrices.png)
![Biểu đồ so sánh](confusion_matrix_script3.png)
---

## Tác giả (Author)

* **Họ và tên:** Phạm Tùng Lâm
* **Trường:** Đại học Bách Khoa Hà Nội (HUST)
* **Giảng viên hướng dẫn:** TS. Vũ Thị Hương Giang
* **Email:** lamcaro12212332@gmail.com

---
