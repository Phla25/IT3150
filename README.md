# CÀI ĐẶT THUẬT TOÁN NAIVE BAYES
<!-- TOC -->
## [Mục Lục](#mục-lục)
1. [Chương 1: Giới thiệu](#chương-1-giới-thiệu)
    1. [Công nghệ sử dụng trong dự án](#1-công-nghệ-sử-dụng-trong-dự-án)
    2. [Một số thuật ngữ cần lưu ý](#2-một-số-thuật-ngữ-cần-lưu-ý)
2. [Chương 2: Cơ sở lý thuyết và mô hình Naive Bayes](#chương-2-cơ-sở-lý-thuyết-và-mô-hình-naive-bayes)
    1. [Giới thiệu về xác suất Bayes](#1-giới-thiệu-về-xác-suất-bayes)
    2. [Giả thiết độc lập có điều kiện trong Naive Bayes](#2-giả-thiết-độc-lập-có-điều-kiện-trong-naive-bayes)
    3. [Các biến thể của Naive Bayes](#3-các-biến-thể-của-naive-bayes)

<!-- /TOC -->
### Chương 1: GIỚI THIỆU
#### 1. Công nghệ sử dụng trong dự án
    Dự án “Cài đặt thuật toán Naive Bayes để phân tích và định lượng rủi ro bảo mật trong DevOps pipelines” được xây dựng và triển khai hoàn toàn bằng ngôn ngữ Python, một ngôn ngữ lập trình mạnh mẽ, phổ biến trong lĩnh vực khoa học dữ liệu (Data Science) và học máy (Machine Learning). 
<br>
Ngôn ngữ lập trình được lựa chọn là Python với những ưu điểm:
    
- Cú pháp đơn giản, dễ đọc, dễ mở rộng.
- Có hệ sinh thái thư viện phong phú phục vụ cho thống kê, học máy, và trực quan hóa dữ liệu.
- Dễ dàng tích hợp với các pipeline DevOps.
<br>

| Công nghệ / Thư viện           | Vai trò / Mục đích sử dụng                                                                                                                                                   |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Python 3.11+**               | Ngôn ngữ lập trình chính để phát triển và huấn luyện mô hình Naive Bayes.                                                                                                    |
| **scikit-learn**               | Thư viện học máy nổi tiếng, cung cấp sẵn các thuật toán Naive Bayes (GaussianNB, MultinomialNB, BernoulliNB) cùng các công cụ hỗ trợ tiền xử lý dữ liệu và đánh giá mô hình. |
| **pandas**                     | Xử lý và phân tích dữ liệu dạng bảng (DataFrame), hỗ trợ đọc/ghi file CSV, thống kê mô tả, và chuyển đổi dữ liệu phục vụ huấn luyện.                                         |
| **numpy**                      | Cung cấp các phép toán đại số tuyến tính, xác suất và xử lý mảng dữ liệu hiệu năng cao.                                                                                      |
| **matplotlib / seaborn**       | Thư viện trực quan hóa dữ liệu, được dùng để vẽ biểu đồ phân bố rủi ro, confusion matrix và so sánh kết quả mô hình.                                                         |
| **jupyter notebook**           | Môi trường lập trình tương tác, phục vụ cho quá trình thử nghiệm, trực quan hóa và trình bày mô hình, cho phép thực nghiệm mô hình nhanh chóng và trình bày kết quả theo từng ô lệnh (cell).                                                                        |
| **joblib**                     | Dùng để lưu trữ và tải lại mô hình đã huấn luyện (serialization).                                                                                                            |
| **Git / GitHub / CI Pipeline** | Dùng để quản lý mã nguồn, tích hợp kiểm thử mô hình tự động vào pipeline DevOps.                                                                                             |
<br>

#### 2. Một số thuật ngữ cần lưu ý
<br>

| Thuật ngữ                          | Giải thích                                                                                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **DevOps pipeline**                | Quy trình tự động hóa việc xây dựng (build), kiểm thử (test), và triển khai (deploy) phần mềm.                                                               |
| **Security risk (rủi ro bảo mật)** | Khả năng một sự kiện hoặc thành phần trong hệ thống gây ra mối nguy cho bảo mật thông tin, ví dụ: lộ key, cấu hình sai, gói phụ thuộc độc hại.               |
| **Feature (đặc trưng)**            | Các thuộc tính đầu vào của mô hình học máy. Trong đề tài này, chúng có thể là số lượng lỗi, số cảnh báo bảo mật, tần suất commit, hoặc thời gian triển khai. |
| **Label (nhãn)**                   | Giá trị đầu ra mà mô hình dự đoán — ví dụ “Nguy cơ cao”, “Trung bình”, “Thấp”.                                                                               |
| **Training / Testing set**         | Tập dữ liệu dùng để huấn luyện và kiểm tra mô hình.                                                                                                          |
| **Naive Bayes Classifier**         | Bộ phân loại dựa trên công thức xác suất Bayes với giả định các đặc trưng là độc lập có điều kiện.                                                           |
| **Precision / Recall / F1-score**  | Các chỉ số đánh giá chất lượng mô hình phân loại.                                                                                                            |
<br>
### Chương 2: CƠ SỞ LÝ THUYẾT VÀ MÔ HÌNH NAIVE BAYES
#### 1. Giới thiệu về xác suất Bayes:
    Thuật toán Naive Bayes được xây dựng dựa trên định lý Bayes, một trong những nền tảng cơ bản của xác suất thống kê. Định lý Bayes mô tả mối quan hệ giữa xác suất tiên nghiệm, xác suất có điều kiện, và xác suất hậu nghiệm của các biến ngẫu nhiên.
<br>

$$ P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)} $$

<br>

Trong đó:
- $P(Y|X)$ là xác suất hậu nghiệm - Xác suất của lớp $Y$ khi biết các đặc trưng $X$
- $P(X|Y)$ là xác suất tiên ***likelihood*** - khả năng xảy ra dữ liệu $X$ khi biết lớp $Y$.
- $P(Y)$ là xác suất tiên nghiệm - xác suất xảy ra lớp $Y$ trước khi quan sát dữ liệu
- $P(X)$ là xác suất ***biên***, dùng để chuẩn hóa.
<br>

Trong bài toán phân loại, xác suất ***biên $P(X)$*** là hằng số chung cho mọi lớp, nên ta có thể so sánh trực tiếp:

<br>

$$\hat{Y} = argmax_{Y}P(Y)P(X|Y)$$

<br>

#### 2. Giả thiết độc lập có điều kiện trong Naive Bayes
Điểm đặc trưng của Naive Bayes là **giả định các đặc trưng đầu vào là độc lập có điều kiện** khi biết lớp $Y$:

<br>

$$P(X|Y) = \prod_{i=1}^{n}P(X_i|Y)$$

<br>

Giả định này đơn giản hóa việc tính toán xác suất, giúp mô hình hoạt động nhanh và dễ huấn luyện, ngay cả với dữ liệu có nhiều thuộc tính.

<br>

Mặc dù trong thực tế các đặc trưng thường có quan hệ phụ thuộc lẫn nhau, Naive Bayes vẫn cho kết quả đáng tin cậy trong nhiều ứng dụng, đặc biệt là ***phân loại văn bản, phân loại email spam, và phân tích rủi ro***.

<br>

#### 3. Các biến thể của Naive Bayes
**a. Gaussian Naive Bayes**
<br>
Gaussian Naive Bayes là một dạng của phương pháp Naive Bayes hoạt động với ***các thuộc tính liên tục (continuous attributes)*** và các đặc trưng dữ liệu (data features) ***tuân theo phân phối Gaussian (Gaussian distribution) trong toàn bộ tập dữ liệu***. Giả định “ngây thơ” (naive) này giúp đơn giản hóa các phép tính và làm cho mô hình trở nên nhanh chóng và hiệu quả. Gaussian Naive Bayes được sử dụng rộng rãi vì nó hoạt động tốt ngay cả với các tập dữ liệu nhỏ và dễ dàng để triển khai và giải thích.
<br>

**b. Multinominal Naive Bayes**
<br>

Multinomial Naive Bayes (MNB) là một trong những biến thể của thuật toán Naive Bayes lý tưởng cho ***dữ liệu rời rạc (discrete data)*** và thường được ***sử dụng trong các bài toán phân loại văn bản (text classification)***. Nó ***mô hình hóa tần suất xuất hiện của các từ (frequency of words)*** dưới dạng **số đếm (counts)** và giả định rằng ***mỗi đặc trưng hoặc mỗi từ được phân phối đa thức (multinomially distributed)***. MNB được sử dụng rộng rãi cho các tác vụ như phân loại tài liệu dựa trên tần suất từ, ví dụ điển hình là trong việc phát hiện thư rác (spam email detection).
<br>

**Cách Multinominal Naive Bayes thực hiện**

<br>
Trong Multinomial Naive Bayes, từ "Naive" (Ngây thơ) có nghĩa là phương pháp này giả định rằng tất cả các đặc trưng, chẳng hạn như các từ trong một câu, là độc lập với nhau; và "Multinomial" (Đa thức) đề cập đến số lần một từ xuất hiện hoặc tần suất một danh mục xảy ra. Nó hoạt động bằng cách sử dụng số lần đếm từ để phân loại văn bản. Ý tưởng chính là nó giả định mỗi từ trong một tin nhắn hoặc một đặc trưng là độc lập với những từ khác. Điều này có nghĩa là sự hiện diện của một từ không ảnh hưởng đến sự hiện diện của một từ khác, làm cho mô hình dễ sử dụng.
<br>

Mô hình xem xét số lần mỗi từ xuất hiện trong các tin nhắn từ các danh mục khác nhau (chẳng hạn như "thư rác" (spam) hoặc "không phải thư rác" (not spam)). Ví dụ, nếu từ "free" (miễn phí) xuất hiện thường xuyên trong các tin nhắn rác, điều đó sẽ được sử dụng để giúp dự đoán liệu một tin nhắn mới có phải là thư rác hay không.
<br>

Để tính toán xác suất một tin nhắn thuộc về một danh mục nhất định, Multinomial Naive Bayes sử dụng phân phối đa thức:
<br>

$$P(X) = \frac{n!}{n_1!n_2!n_3!...n_m!}p_1^{n_1}p_2^{n_2}p_3^{n_3}...p_m^{n_m}$$

<br>

Trong đó:
- $n$ là tổng số lượt thử nghiệm
- $n_i$ là **số đếm** sự xuất hiện của đầu ra $i$
- $p_i$ là xác suất xảy ra đầu ra $i$

<br>

Để ước tính mức độ có khả năng xuất hiện của mỗi từ trong một lớp cụ thể, chẳng hạn như "thư rác" (spam) hoặc "không phải thư rác" (not spam), chúng ta sử dụng một phương pháp gọi là **Ước lượng Khả năng Hợp lý Tối đa (Maximum Likelihood Estimation - MLE)**. Điều này giúp tìm ra các xác suất dựa trên số lần đếm thực tế từ dữ liệu của chúng ta. Công thức là:
<br>

$$\theta_{c,i} = \frac{count(w_i,c)+1}{N+v}$$
<br>

Trong đó:
- $count(w_i,c)$ là số lần từ $w_i$ xuất hiện trong tài liệu thuộc lớp $c$
- $N$ là tổng số từ trong các tài liệu thuộc lớp $c$
- $v$ là kích thước từ vựng, tổng số từ duy nhất (unique words) trong toàn bộ tập tài liệu (từ ấy có thể xuất hiện lần hai nhưng vẫn chỉ tính là 1)
<br>

**Ví dụ**

<br>

| Message ID | Message Text | Class |
| :----------: | :------------: | :-----: |
|M1|"buy cheap now"|Spam|
|M2|"limited offer buy"|Spam|
|M3|"meet me now"|Not Spam|
|M4|"let's catch up"|Not Spam|

<br>

**B1:** Tập từ vựng $V$ = **{buy, cheap, now, limited, offer, meet, me, let's, catch, up}** 

<br>

$\Rightarrow v = 10$

**B2:** 

Tần số từ theo lớp:

**Lớp Spam (M1, M2)**
- buy: 2
- cheap: 1
- now: 1
- limited: 1
- offer: 1

$\Rightarrow$ Tổng số từ: 6

**Lớp Not Spam (M3, M4)**
- meet: 1
- me: 1
- now: 1
- let's: 1
- catch: 1
- up: 1

$\Rightarrow$ Tổng số từ: 6

**B3:** Message mẫu: *"Buy now"*

**B4:** Áp dụng công thức MNB:







