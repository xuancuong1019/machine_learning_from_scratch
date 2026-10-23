# Machine Learning Pipeline từ A đến Z — Bản tổng ôn

*Ví dụ xuyên suốt: phân loại email Spam / Not Spam.*

## 1. Xác định bài toán (Problem Definition)

- **Học có giám sát (Supervised Learning):** mô hình học từ dữ liệu đã có nhãn.
- **Phân loại (Classification):** dự đoán nhãn, ví dụ Spam / Not Spam.
- **Hồi quy (Regression):** dự đoán giá trị số, ví dụ thời gian giao hàng.
- **Mẫu dữ liệu (Sample):** một đối tượng, thường biểu diễn bằng một hàng.
- **Đặc trưng (Feature):** thông tin đầu vào dùng để dự đoán.
- **Nhãn / biến mục tiêu (Label / Target):** kết quả cần dự đoán.
- \(X\): ma trận đặc trưng; \(y\): giá trị mục tiêu thực tế; \(\hat y\): giá trị dự đoán.
- **Mô hình (Model):** hàm ánh xạ dữ liệu đầu vào thành dự đoán, \(\hat y=f_\theta(X)\).

## 2. Thu thập dữ liệu và phân tích khám phá (EDA)

- Kiểm tra số lượng mẫu, số lượng đặc trưng, kiểu dữ liệu và phân phối dữ liệu.
- Phát hiện **giá trị thiếu (Missing Values)**, dữ liệu trùng lặp, giá trị bất thường và lỗi nhập liệu.
- Kiểm tra phân phối nhãn và mức độ mất cân bằng lớp.
- Xem xét mối quan hệ giữa các đặc trưng và biến mục tiêu.
- **Rò rỉ dữ liệu (Data Leakage):** mô hình sử dụng thông tin không được phép biết tại thời điểm dự đoán, khiến kết quả đánh giá thiếu tin cậy.

## 3. Chia dữ liệu (Train / Validation / Test)

Ví dụ minh họa với 10.000 mẫu, chia theo tỷ lệ 70/15/15:

- **Training Set:** 7.000 mẫu, dùng để học tham số mô hình.
- **Validation Set:** 1.500 mẫu, dùng để lựa chọn mô hình, siêu tham số, ngưỡng phân loại và theo dõi dừng sớm.
- **Test Set:** 1.500 mẫu, dùng để đánh giá cuối cùng trên dữ liệu chưa được sử dụng trong quá trình lựa chọn mô hình.

Các cách chia dữ liệu:

- **Random Split:** chia ngẫu nhiên.
- **Stratified Split:** giữ tỷ lệ các lớp gần với dữ liệu ban đầu.
- **Group Split:** giữ các nhóm liên quan trong cùng một tập, ví dụ các email cùng tên miền.
- **Time-based Split:** chia theo thời gian khi cần mô phỏng dự đoán tương lai.
- **Cross-Validation (CV):** chia dữ liệu thành nhiều phần để huấn luyện và đánh giá lặp lại.

**Nguyên tắc:** không sử dụng Test Set để lựa chọn mô hình hoặc điều chỉnh siêu tham số. Mọi bước tiền xử lý cần học thông tin từ dữ liệu phải được thực hiện trên phần huấn luyện tương ứng.

## 4. Tiền xử lý dữ liệu (Data Preprocessing)

### 4.1. Xử lý giá trị thiếu (Missing Values)

- **MCAR:** thiếu hoàn toàn ngẫu nhiên.
- **MAR:** việc thiếu liên quan đến những biến đã quan sát.
- **MNAR:** việc thiếu liên quan đến chính giá trị bị thiếu hoặc thông tin chưa quan sát được.
- Các cách xử lý: xóa hàng/cột, điền bằng trung bình, trung vị, giá trị phổ biến, hằng số hoặc tạo đặc trưng chỉ báo thiếu.
- **Trung vị (Median)** thường ít bị ảnh hưởng bởi giá trị ngoại lệ hơn trung bình.
- Giá trị thiếu không đồng nghĩa với 0 hoặc “Không”.
- Chỉ chọn cách xử lý sau khi xem xét dữ liệu và bối cảnh.

### 4.2. Mã hóa dữ liệu phân loại (Categorical Encoding)

- **One-Hot Encoding:** tạo các cột nhị phân cho những nhóm không có thứ tự tự nhiên.
- **Ordinal Encoding:** ánh xạ các nhóm có thứ tự thành số, ví dụ Thấp = 0, Trung bình = 1, Cao = 2.
- Không tự ý gán số nguyên cho các nhóm không có thứ tự vì có thể tạo ra thứ tự hoặc khoảng cách giả.
- One-Hot Encoding có thể làm tăng số lượng đặc trưng khi biến có quá nhiều giá trị khác nhau.

### 4.3. Chuẩn hóa thang đo (Feature Scaling)

**Chuẩn hóa theo trung bình và độ lệch chuẩn (Standardization):**

\[
z=\frac{x-\mu}{\sigma}
\]

Trong đó \(\mu\) là trung bình và \(\sigma\) là độ lệch chuẩn được tính từ tập huấn luyện. Dữ liệu sau biến đổi có trung bình gần 0 và độ lệch chuẩn gần 1 khi áp dụng trên chính tập huấn luyện với cách tính tương ứng.

**Đưa về khoảng (Min-Max Scaling):**

\[
x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}
\]

Giá trị trên tập huấn luyện thường nằm trong khoảng \([0,1]\). Dữ liệu mới có thể nằm ngoài khoảng này nếu vượt quá giá trị nhỏ nhất hoặc lớn nhất của tập huấn luyện.

- Thường hữu ích với KNN, SVM, Logistic Regression dùng tối ưu dựa trên gradient và mạng nơ-ron.
- Thường không bắt buộc với Decision Tree và Random Forest.
- Cả hai phương pháp có thể bị ảnh hưởng bởi giá trị ngoại lệ; Min-Max đặc biệt nhạy với các giá trị cực đoan.

### 4.4. Biến đổi dữ liệu (Data Transformation)

Biến đổi dữ liệu làm thay đổi cách biểu diễn giá trị.

Ví dụ, với dữ liệu không âm:

\[
x'=\ln(1+x)
\]

- Phép biến đổi log có thể nén các giá trị lớn và giảm độ lệch phải.
- Biến đổi dữ liệu không tự động loại bỏ ngoại lệ hoặc đảm bảo phân phối chuẩn.
- **Scaling** điều chỉnh thang đo; **Transformation** thay đổi cách biểu diễn dữ liệu. Hai thao tác có thể được kết hợp.
- Không bắt buộc biến đổi mọi đặc trưng. Cần lựa chọn dựa trên dữ liệu và mô hình.

### 4.5. Quy trình tiền xử lý (Preprocessing Pipeline)

Quy trình minh họa:

**Kiểm tra dữ liệu → Chia tập → Điền giá trị thiếu → Mã hóa → Biến đổi / chuẩn hóa khi cần → Huấn luyện mô hình.**

- **Fit:** học thông tin từ dữ liệu, ví dụ trung vị, trung bình, độ lệch chuẩn hoặc danh sách nhóm.
- **Transform:** áp dụng thông tin đã học để biến đổi dữ liệu.
- Fit các bước tiền xử lý trên Training Set; dùng cùng các đối tượng đã fit để transform Validation và Test.
- Trong Cross-Validation, fit tiền xử lý riêng bên trong phần huấn luyện của từng lượt chia.
- Chỉ thực hiện những bước thực sự cần thiết, không áp dụng tất cả một cách máy móc.

## 5. Mô hình, tham số và siêu tham số

- **Model:** hàm tạo dự đoán từ dữ liệu.
- **Tham số (Parameters):** giá trị mô hình học được từ dữ liệu, ví dụ trọng số \(w\) và hệ số chệch \(b\) của Linear Regression.
- **Siêu tham số (Hyperparameters):** cấu hình do người xây dựng lựa chọn, ví dụ `max_depth`, `learning_rate`, `batch_size`.
- **Linear Regression:** \(\hat y=wx+b\) trong trường hợp một đặc trưng.
- **Logistic Regression:** biến tổ hợp tuyến tính thành xác suất thông qua hàm sigmoid.

## 6. Hàm mất mát và tối ưu hóa (Loss & Optimization)

### 6.1. Hàm mất mát (Loss Function)

Đo mức độ sai khác giữa dự đoán và giá trị thực tế.

**Sai số bình phương trung bình (Mean Squared Error — MSE):**

\[
MSE=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2
\]

Bình phương sai số giúp tránh triệt tiêu giữa sai số dương và âm, đồng thời phạt mạnh hơn các sai số lớn.

**Entropy chéo nhị phân (Binary Cross-Entropy — BCE):**

\[
L=-[y\log p+(1-y)\log(1-p)]
\]

Trong đó \(y\in\{0,1\}\) là nhãn thực tế và \(p\) là xác suất dự đoán cho lớp 1.

- **Loss:** thường chỉ mức sai số của một mẫu hoặc một nhóm mẫu.
- **Cost / Objective Function:** thường chỉ hàm mục tiêu được tối ưu trên toàn bộ dữ liệu hoặc một tập mẫu.
- Cách dùng thuật ngữ có thể khác nhau giữa các tài liệu.

### 6.2. Đạo hàm và gradient

- **Đạo hàm (Derivative):** biểu thị tốc độ thay đổi của một hàm theo biến.
- **Đạo hàm riêng (Partial Derivative):** đạo hàm theo một biến khi xem các biến khác là cố định.
- **Gradient:** véc-tơ chứa các đạo hàm riêng của hàm mục tiêu theo các tham số.
- Gradient chỉ hướng tăng nhanh nhất cục bộ; khi muốn giảm hàm mục tiêu, Gradient Descent đi theo hướng ngược lại.

### 6.3. Hạ dốc theo gradient (Gradient Descent)

\[
\theta_{t+1}=\theta_t-\eta\nabla_\theta J(\theta_t)
\]

- \(\theta\): các tham số cần học.
- \(J\): hàm mục tiêu.
- \(\nabla_\theta J\): gradient của hàm mục tiêu.
- \(\eta\): tốc độ học (**Learning Rate**).

Learning Rate quá nhỏ có thể khiến quá trình học chậm; quá lớn có thể khiến quá trình dao động hoặc không hội tụ.

**Lan truyền ngược (Backpropagation)** tính gradient trong mạng nơ-ron; **Optimizer** sử dụng gradient để cập nhật tham số. Đây là hai nhiệm vụ liên quan nhưng không đồng nghĩa.

## 7. Vòng lặp huấn luyện (Training Loop)

Một vòng huấn luyện điển hình:

1. Khởi tạo tham số.
2. **Forward Pass:** tính dự đoán.
3. Tính Loss.
4. Tính Gradient / Backpropagation khi cần.
5. Optimizer cập nhật tham số.
6. Lặp lại cho đến điều kiện dừng.

Các khái niệm:

- **Epoch:** một lượt đi qua toàn bộ Training Set.
- **Batch:** nhóm mẫu được xử lý trong một bước.
- **Iteration / Training Step:** một lần cập nhật tham số trong huấn luyện theo batch thông thường.

Số iteration mỗi epoch:

\[
\text{Iterations per epoch}=\left\lceil\frac{N}{B}\right\rceil
\]

Trong đó \(N\) là số mẫu huấn luyện và \(B\) là kích thước batch.

Ví dụ, 7.000 mẫu với `batch_size = 100` tạo ra 70 iteration mỗi epoch. Sau 10 epoch sẽ có 700 iteration nếu mỗi batch đều cập nhật tham số một lần.

## 8. Khả năng tổng quát hóa (Generalization)

**Generalization** là khả năng dự đoán tốt trên dữ liệu mới chưa được sử dụng để huấn luyện.

- **Underfitting:** mô hình chưa học đủ quy luật; lỗi huấn luyện và lỗi kiểm định thường đều cao.
- **Overfitting:** mô hình học quá sát dữ liệu huấn luyện; lỗi huấn luyện thấp nhưng lỗi kiểm định có thể cao.
- **Bias–Variance Tradeoff:** sự đánh đổi giữa độ đơn giản của mô hình và mức độ nhạy với dữ liệu huấn luyện.
- **Learning Curve:** đồ thị theo dõi Loss hoặc Metric qua các epoch hay theo kích thước tập huấn luyện.
- **Regularization:** thêm thành phần phạt vào hàm mục tiêu để hạn chế độ phức tạp của mô hình.

\[
J_{\text{new}}=J_{\text{data}}+\lambda R(\theta)
\]

- \(\lambda\) điều chỉnh mức độ regularization.
- \(\lambda\) quá lớn có thể dẫn đến underfitting.
- **Early Stopping:** dừng huấn luyện dựa trên hiệu suất Validation để hạn chế overfitting.
- Các cách giảm overfitting khác gồm giới hạn độ phức tạp, cải thiện dữ liệu và lựa chọn đặc trưng phù hợp.

## 9. Đánh giá mô hình (Model Evaluation)

### 9.1. Ma trận nhầm lẫn (Confusion Matrix)

Với lớp dương là Spam:

- **TP (True Positive):** Spam được dự đoán là Spam.
- **TN (True Negative):** Not Spam được dự đoán là Not Spam.
- **FP (False Positive):** Not Spam bị dự đoán nhầm thành Spam.
- **FN (False Negative):** Spam bị dự đoán nhầm thành Not Spam.

### 9.2. Các chỉ số đánh giá

**Độ chính xác tổng thể (Accuracy):**

\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

**Độ chính xác của các dự đoán dương (Precision):**

\[
Precision=\frac{TP}{TP+FP}
\]

**Độ bao phủ lớp dương (Recall / TPR):**

\[
Recall=\frac{TP}{TP+FN}
\]

**Trung bình điều hòa Precision và Recall (F1-score):**

\[
F1=2\frac{Precision\cdot Recall}{Precision+Recall}
\]

**Tỷ lệ dương tính giả (False Positive Rate — FPR):**

\[
FPR=\frac{FP}{FP+TN}
\]

- **ROC Curve:** biểu diễn TPR theo FPR qua nhiều ngưỡng.
- **ROC-AUC:** đánh giá khả năng phân biệt và xếp hạng giữa hai lớp qua các ngưỡng.
- **Precision–Recall Curve / Average Precision:** hữu ích khi cần chú trọng lớp dương trong dữ liệu mất cân bằng.

Accuracy có thể gây hiểu lầm nếu 95% dữ liệu là Not Spam và chỉ 5% là Spam: mô hình luôn dự đoán Not Spam vẫn có Accuracy 95% nhưng Recall của Spam bằng 0.

### 9.3. Ngưỡng phân loại (Classification Threshold)

Với xác suất dự đoán \(p\) và ngưỡng \(t\):

- Nếu \(p\ge t\), dự đoán lớp dương.
- Nếu \(p<t\), dự đoán lớp âm.

Giảm ngưỡng thường làm Recall và FPR tăng, nhưng Precision không nhất thiết thay đổi đơn điệu. Chọn ngưỡng dựa trên Validation Set và chi phí của FP/FN, không dựa trên Test Set.

## 10. Triển khai và giám sát (Deployment & Monitoring)

- Đưa mô hình vào hệ thống sử dụng thực tế.
- Theo dõi chất lượng dự đoán và dữ liệu đầu vào.
- **Data Drift:** phân phối dữ liệu đầu vào thay đổi.
- **Concept Drift:** mối quan hệ giữa đặc trưng và biến mục tiêu thay đổi.
- Đánh giá lại và huấn luyện lại khi có đủ bằng chứng cho thấy mô hình không còn phù hợp.

---

# PHẦN B — TOÁN HỌC VÀ NUMPY CHO MACHINE LEARNING FROM SCRATCH

## 11. Số, véc-tơ và ma trận (Scalar, Vector, Matrix)

- **Số (Scalar):** một giá trị đơn lẻ, ví dụ \(x=5\).
- **Véc-tơ (Vector):** dãy giá trị, ví dụ \(\mathbf{x}=[2,3,5]\).
- **Ma trận (Matrix):** bảng gồm nhiều hàng và cột.

Ví dụ:

\[
X=\begin{bmatrix}2&3&5\\1&4&6\end{bmatrix}
\]

Ma trận này có 2 hàng và 3 cột.

Với dữ liệu Machine Learning:

- \(N\): số mẫu.
- \(d\): số đặc trưng.
- \(X\) thường có kích thước \(N\times d\).
- \(y\) chứa các giá trị mục tiêu; trong NumPy có thể có dạng `(N,)` hoặc `(N, 1)` tùy cách biểu diễn.

Kích thước ma trận thường được viết theo thứ tự **số hàng × số cột**.

## 12. Các phép toán trên véc-tơ

### 12.1. Cộng véc-tơ

\[
[2,3]+[4,1]=[6,4]
\]

Cộng các phần tử cùng vị trí. Hai véc-tơ cần có cùng số phần tử.

### 12.2. Nhân véc-tơ với một số

\[
2[2,3]=[4,6]
\]

Nhân từng phần tử với cùng một số.

### 12.3. Nhân từng phần tử và tích vô hướng

Nhân từng phần tử:

\[
[2,3]*[4,5]=[8,15]
\]

**Tích vô hướng (Dot Product):** nhân các phần tử cùng vị trí rồi cộng các kết quả:

\[
[2,3]\cdot[4,5]=2(4)+3(5)=23
\]

Tích vô hướng giúp kết hợp nhiều đặc trưng với trọng số để tạo ra một giá trị.

## 13. Phép nhân ma trận với véc-tơ

Giả sử:

\[
X=\begin{bmatrix}2&3\\1&4\\3&2\end{bmatrix},
\quad
\mathbf{w}=\begin{bmatrix}4\\5\end{bmatrix}
\]

Khi đó:

\[
X\mathbf{w}=\begin{bmatrix}23\\24\\22\end{bmatrix}
\]

Quy tắc kích thước:

\[
(N\times d)(d\times1)=(N\times1)
\]

Số cột của ma trận thứ nhất phải bằng số hàng của ma trận thứ hai. Kết quả có số hàng của ma trận thứ nhất và số cột của ma trận thứ hai.

## 14. Phép chuyển vị ma trận (Transpose)

Phép chuyển vị đổi hàng thành cột và cột thành hàng.

\[
X=\begin{bmatrix}2&3&5\\1&4&6\end{bmatrix}
\]

\[
X^T=\begin{bmatrix}2&1\\3&4\\5&6\end{bmatrix}
\]

Nếu \(X\) có kích thước \(N\times d\), thì \(X^T\) có kích thước \(d\times N\).

Chuyển vị không làm mất các giá trị; nó thay đổi cách sắp xếp hàng và cột. Phép này thường cần thiết để căn chỉnh kích thước khi thực hiện phép nhân ma trận.

## 15. NumPy: Biểu diễn và tính toán dữ liệu

**NumPy (Numerical Python)** là thư viện Python hỗ trợ tính toán số học trên mảng nhiều chiều.

```python
import numpy as np

x = np.array([2, 3, 5])

X = np.array([
    [2, 3, 5],
    [1, 4, 6]
])
```

Khác với phép `+` trên danh sách Python, phép `+` giữa các mảng NumPy cùng kích thước cộng các phần tử tương ứng.

### 15.1. Các thuộc tính cần biết

```python
X.shape  # (2, 3): số hàng và cột
X.ndim   # 2: số chiều
X.size   # 6: tổng số phần tử
```

- **`shape`:** kích thước theo từng chiều.
- **`ndim`:** số chiều của mảng.
- **`size`:** tổng số phần tử.

### 15.2. Truy cập phần tử

NumPy đánh số vị trí bắt đầu từ 0.

```python
X[0, 0]  # 2
X[0, 1]  # 3
X[1, 2]  # 6
```

Cú pháp `X[hàng, cột]` lấy một phần tử theo chỉ số hàng và cột.

### 15.3. Phân biệt `*` và `@`

- `*`: nhân từng phần tử tương ứng.
- `@`: thực hiện tích vô hướng với hai véc-tơ một chiều, hoặc phép nhân ma trận phù hợp với kích thước các mảng.

```python
a = np.array([2, 3])
w = np.array([4, 5])

a * w  # [8, 15]
a @ w  # 23
```

Cần kiểm tra kích thước các mảng khi thực hiện phép nhân.

## 16. Dùng NumPy tính dự đoán

Mô hình hồi quy tuyến tính với nhiều đặc trưng:

\[
\hat y=X\mathbf{w}+b
\]

Trong đó:

- \(X\): ma trận đặc trưng.
- \(\mathbf{w}\): véc-tơ trọng số.
- \(b\): hệ số chệch, được cộng vào từng dự đoán.
- \(\hat y\): véc-tơ dự đoán cho các mẫu.

Ví dụ:

```python
import numpy as np

X = np.array([
    [2, 3],
    [1, 4],
    [3, 2]
])

w = np.array([4, 5])
b = 1

y_pred = X @ w + b

# [24, 25, 23]
```

Phép nhân ma trận giúp tính dự đoán cho nhiều mẫu cùng lúc, thay vì phải viết phép tính riêng cho từng mẫu.

Với phân loại nhị phân bằng Logistic Regression, đầu ra tuyến tính sẽ được đưa qua hàm sigmoid để tạo xác suất; không thể mặc định mọi đầu ra của công thức tuyến tính đều là xác suất.

## 17. Thống kê cơ bản (Statistics)

### 17.1. Trung bình (Mean)

\[
\bar{x}=\frac{1}{N}\sum_{i=1}^{N}x_i
\]

Trung bình bằng tổng các giá trị chia cho số lượng giá trị. Nó dùng để tóm tắt dữ liệu và được sử dụng trong một số phương pháp điền giá trị thiếu, chuẩn hóa và tính toán thống kê.

### 17.2. Phương sai (Variance)

Phương sai đo mức độ phân tán của dữ liệu quanh trung bình.

Với toàn bộ tập dữ liệu:

\[
\sigma^2=\frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2
\]

Trong đó \(\mu\) là trung bình và \(N\) là số giá trị.

Các bước tính:

1. Trừ trung bình khỏi từng giá trị.
2. Bình phương từng độ lệch.
3. Tính trung bình các bình phương đó.

Phương sai càng lớn thường biểu thị dữ liệu càng phân tán. Khi tính phương sai mẫu để ước lượng từ một mẫu quan sát, công thức thường dùng mẫu số \(N-1\) thay vì \(N\).

### 17.3. Độ lệch chuẩn (Standard Deviation)

\[
\sigma=\sqrt{\sigma^2}
\]

Độ lệch chuẩn là căn bậc hai của phương sai, nên có cùng đơn vị với dữ liệu gốc. Độ lệch chuẩn bằng 0 khi tất cả giá trị giống nhau.

Trong chuẩn hóa Standardization, trung bình và độ lệch chuẩn được tính từ tập huấn luyện để chuyển các giá trị về thang đo phù hợp.

## 18. Trung vị và giá trị ngoại lệ (Median & Outliers)

### 18.1. Trung vị (Median)

Trung vị là giá trị ở giữa sau khi sắp xếp dữ liệu.

- Số lượng giá trị lẻ: lấy giá trị chính giữa.
- Số lượng giá trị chẵn: lấy trung bình cộng của hai giá trị chính giữa.

Ví dụ:

\[
[2,4,5,7,9]\Rightarrow Median=5
\]

\[
[2,4,6,8]\Rightarrow Median=\frac{4+6}{2}=5
\]

### 18.2. Giá trị ngoại lệ (Outlier)

Giá trị ngoại lệ là giá trị khác biệt đáng kể so với phần lớn dữ liệu còn lại. Nó có thể đến từ lỗi nhập liệu, lỗi đo lường hoặc một trường hợp hiếm nhưng có thật.

Không nên tự động xóa mọi ngoại lệ; cần kiểm tra nguyên nhân và ý nghĩa thực tế.

### 18.3. Trung bình so với trung vị

Ví dụ:

\[
X=[2,3,3,4,100]
\]

- Trung bình: \(22.4\).
- Trung vị: \(3\).

Giá trị 100 kéo trung bình lên mạnh nhưng ít ảnh hưởng đến trung vị.

- Dữ liệu tương đối cân đối, ít ngoại lệ đáng kể: trung bình có thể phù hợp.
- Dữ liệu lệch hoặc có giá trị cực đoan: trung vị thường bền vững hơn.
- Khi điền dữ liệu số bị thiếu, có thể cân nhắc trung bình hoặc trung vị tùy phân phối và bối cảnh.
- Thống kê dùng để điền giá trị thiếu phải được học từ Training Set, tránh rò rỉ dữ liệu.

---

# Toàn bộ quy trình Machine Learning

**Xác định bài toán → Thu thập dữ liệu → EDA & kiểm tra chất lượng → Chia Train/Validation/Test → Tiền xử lý (Missing Values → Encoding → Transformation/Scaling khi cần) → Chọn mô hình → Huấn luyện (Forward Pass → Loss → Gradient → Update) → Kiểm tra Generalization → Chọn Metrics & Threshold bằng Validation → Đánh giá cuối cùng trên Test → Triển khai & giám sát.**

## Những nguyên tắc cần nhớ

1. **Tránh Data Leakage:** không để thông tin từ Validation/Test ảnh hưởng đến việc học tham số tiền xử lý hoặc lựa chọn mô hình.
2. **Chọn cách tiền xử lý phù hợp:** không phải dữ liệu nào cũng cần scaling, transformation hoặc điền giá trị thiếu theo cùng một cách.
3. **Hiểu phép toán nền tảng:** véc-tơ, ma trận, kích thước, tích vô hướng và phép nhân ma trận là cơ sở để tự xây dựng mô hình.
4. **Đánh giá đúng mục tiêu:** lựa chọn chỉ số theo bài toán và chi phí của các loại sai lầm.
5. **Quan tâm khả năng tổng quát hóa:** mô hình cần dự đoán tốt trên dữ liệu mới, không chỉ trên dữ liệu huấn luyện.

**Bước học tiếp theo:** xác suất cơ bản và xác suất có điều kiện; sau đó học đạo hàm, đạo hàm riêng, gradient và quy tắc dây chuyền. Khi đã nắm các kiến thức NumPy cần thiết, bắt đầu triển khai Linear Regression từ đầu, rồi chuyển sang Logistic Regression và các mô hình khác.