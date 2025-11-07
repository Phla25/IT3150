import pandas as pd
import numpy as np

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