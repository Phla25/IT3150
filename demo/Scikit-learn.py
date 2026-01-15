import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
# Tạo dữ liệu giả định
data = {
    'text': [
        "buy cheap now limited offer",
        "free gift card hurry",
        "meet me for lunch today",
        "let's catch up now",
        "urgent: claim your prize free"
    ],
    'label': ['spam', 'spam', 'ham', 'ham', 'spam']
}
# Tương tự hàm df = pd.read_csv('spam.csv', encoding='latin-1') 
# nhưng tạo DataFrame trực tiếp từ dữ liệu giả định 
df = pd.DataFrame(data)

# In 5 dòng đầu và thông tin cơ bản
print("--- DataFrame ban đầu ---")
print(df.head(5))
print("\n--- Thông tin dữ liệu ---")
df.info()

# Sử dụng Pandas để ánh xạ nhãn thành số (0 và 1)
df['label_encoded'] = df['label'].map({'spam': 1, 'ham': 0})

print("\n--- Dữ liệu đã ánh xạ ---")
print(df[['label', 'label_encoded']])
# 1. Tách dữ liệu
# x là phần văn bản đầu vào, y là phần nhãn đã mã hóa
X = df['text']
y = df['label_encoded']

# 2. Chia tập huấn luyện và kiểm tra
# Lưu ý: Với dữ liệu nhỏ, việc chia tập chỉ mang tính minh họa.
# Chia 60% huấn luyện, 40% kiểm tra
# Random_state để đảm bảo kết quả tái lập được
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 3. Tiền xử lý (Sử dụng CountVectorizer)
# fit_transform() học từ vựng từ tập train và biến nó thành ma trận đếm.
# transform() dùng lại từ vựng đó để xử lý tập test (không học thêm).
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test) # Chỉ dùng transform cho tập kiểm tra
# Giả sử có 11 từ duy nhất, và 3 câu train → ma trận kích thước (3, 11).

# In hình dạng ma trận đếm từ (5 mẫu, 11 từ duy nhất)
print("\n--- Ma trận đếm từ (dữ liệu huấn luyện) ---")
print(f"Hình dạng ma trận đếm: {X_train_counts.shape}")

# 4. Huấn luyện Mô hình MNB
# MultinomialNB: dùng cho dữ liệu đếm (như tần suất từ).
# Alpha là tham số điều chỉnh độ mượt (smoothing).

model = MultinomialNB(alpha=1.0)
model.fit(X_train_counts, y_train)

# 5. Dự đoán
y_pred = model.predict(X_test_counts)
print(f"\nNhãn dự đoán: {y_pred}")
print(f"Nhãn thực tế: {y_test.values}")

# 6. Đánh giá
# accuracy_score: tỉ lệ dự đoán đúng.
# confusion_matrix: ma trận nhầm lẫn (thường dạng 2×2):
accuracy = accuracy_score(y_test, y_pred)
print(f"\nĐộ chính xác: {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)