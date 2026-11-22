# M2 – Spline Regression: Dự đoán khối lượng chim cánh cụt

**Thành viên phụ trách:** Khang Hy
**Mô hình:** M2 – Cubic Spline Basis Regression  
**Ngày thực hiện:** 21/11/2026

## 1. Mục tiêu

Mục tiêu của M2 là dự đoán khối lượng cơ thể chim cánh cụt (`body_mass_g`, đơn vị gram) dựa trên các đặc trưng hình thái và thông tin phân loại.

Mô hình sử dụng chung bộ đặc trưng của nhóm:

- `flipper_length_mm`
- `bill_length_mm`
- `bill_depth_mm`
- `species`
- `sex`

M2 sử dụng spline basis để mô hình hóa các quan hệ phi tuyến giữa ba đặc trưng số và khối lượng cơ thể, trong khi vẫn sử dụng hồi quy tuyến tính ở tầng ước lượng cuối cùng.

## 2. Dữ liệu và tiền xử lý

### 2.1. Tổng quan dữ liệu

Dữ liệu được lấy từ bộ Palmer Penguins.

- Số dòng ban đầu: 344.
- Số cột: 8.
- Các cột gồm: `species`, `island`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `sex`, `year`.
- Biến mục tiêu `body_mass_g` có 2 giá trị thiếu.
- Các biến `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm` mỗi biến có 2 giá trị thiếu.
- Biến `sex` có 11 giá trị thiếu.
- Các cột `species`, `island`, `year` không có giá trị thiếu.

Hai dòng thiếu nhãn `body_mass_g` được loại bỏ trước khi chia dữ liệu. Sau bước này còn 342 dòng có nhãn.

### 2.2. Chia tập dữ liệu

Dữ liệu được chia thành:

- Training set: 273 dòng.
- Test set: 69 dòng.
- Tỷ lệ: 80% training và 20% testing.
- `random_state=42`.

Cách chia này được dùng chung trong mã nguồn của nhóm để đảm bảo các mô hình được đánh giá trên cùng tập kiểm tra.

### 2.3. Tiền xử lý

Pipeline sử dụng các bước sau:

**Đối với ba đặc trưng số:**

1. Điền giá trị thiếu bằng median.
2. Áp dụng `SplineTransformer` để chuyển mỗi biến số thành các spline basis features.

**Đối với hai đặc trưng phân loại:**

1. Điền giá trị thiếu bằng giá trị xuất hiện nhiều nhất.
2. Sử dụng `OneHotEncoder(drop="first", handle_unknown="ignore")`.

Các bước tiền xử lý được đặt trong `Pipeline` và chỉ được fit trên dữ liệu huấn luyện nhằm hạn chế rò rỉ dữ liệu từ tập kiểm tra.

## 3. Thiết kế mô hình

### 3.1. Thuật toán

M2 sử dụng Cubic Spline Basis Regression, được xây dựng bằng:

- `SplineTransformer` của scikit-learn để tạo các cơ sở spline bậc ba.
- `LinearRegression` để ước lượng hệ số trên các đặc trưng sau biến đổi.

Khác với hồi quy tuyến tính thông thường, spline basis cho phép mô hình biểu diễn các quan hệ phi tuyến theo từng khoảng của biến đầu vào.

### 3.2. Cấu hình

Cấu hình cuối cùng:

| Siêu tham số | Giá trị |
| --- | --- |
| `degree` | 3 |
| `n_knots` | 3 |
| `knots` | `uniform` |
| `include_bias` | `False` |
| `extrapolation` | `constant` |
| Estimator cuối | `LinearRegression()` |

### 3.3. Lựa chọn số knots

Thực hiện 5-fold cross-validation trên tập training với `KFold(n_splits=5, shuffle=True, random_state=42)` để so sánh bốn cấu hình:

| n_knots | MAE (g) | RMSE (g) | R² |
| ---: | ---: | ---: | ---: |
| 3 | **234.404** | **295.874** | **0.859** |
| 4 | 236.070 | 296.866 | 0.857 |
| 5 | 239.801 | 300.163 | 0.854 |
| 6 | 241.045 | 303.807 | 0.851 |

Cấu hình `n_knots=3` có MAE và RMSE trung bình thấp nhất, đồng thời có R² trung bình cao nhất trong các cấu hình đã thử. Vì vậy, cấu hình này được chọn để đánh giá cuối cùng trên tập test.

### 3.4. Kiểm tra đặc trưng sau biến đổi

Sau khi fit pipeline trên tập training, dữ liệu có kích thước `(273, 15)`.

Tổng số 15 đặc trưng gồm:

- 4 spline basis features từ `flipper_length_mm`.
- 4 spline basis features từ `bill_length_mm`.
- 4 spline basis features từ `bill_depth_mm`.
- 2 biến giả của `species`.
- 1 biến giả của `sex`.

Kết quả này xác nhận ba biến số đã được biến đổi bằng spline bậc ba và các biến phân loại vẫn được mã hóa One-Hot.

## 4. Đánh giá mô hình

### 4.1. Các chỉ số đánh giá

Mô hình được đánh giá trên 69 dòng của tập test bằng ba chỉ số:

- **MAE:** Sai số tuyệt đối trung bình, đơn vị gram; càng thấp càng tốt.
- **RMSE:** Căn bậc hai của sai số bình phương trung bình, đơn vị gram; nhạy hơn với các lỗi dự đoán lớn.
- **R²:** Mức độ giải thích biến thiên của biến mục tiêu; càng gần 1 càng tốt.

### 4.2. Kết quả test

| Mô hình | MAE (g) | RMSE (g) | R² |
| --- | ---: | ---: | ---: |
| M2 – Spline Regression | **252.934** | **308.237** | **0.832** |

### 4.3. So sánh ban đầu với M1

Kết quả M1 Ridge đã được ghi nhận:

| Mô hình | MAE (g) | RMSE (g) | R² |
| --- | ---: | ---: | ---: |
| M1 – Ridge | 254.103 | 301.776 | 0.839 |
| M2 – Spline | 252.934 | 308.237 | 0.832 |

So với M1, M2 có MAE thấp hơn 1.169 g, nhưng RMSE cao hơn 6.461 g và R² thấp hơn 0.007. Như vậy, M2 cải thiện nhẹ sai số tuyệt đối trung bình nhưng chưa vượt M1 ở RMSE và R².

Kết quả cho thấy việc sử dụng spline basis chưa tạo ra cải thiện đồng đều trên tất cả các chỉ số. Cần tiếp tục so sánh với M3 và M4 trên cùng tập test trước khi đưa ra kết luận chung.

## 5. Kết luận

M2 đã triển khai thành công Cubic Spline Basis Regression với ba biến số được biến đổi bằng spline bậc ba và hai biến phân loại được mã hóa One-Hot.

Thông qua 5-fold cross-validation, cấu hình `n_knots=3` được lựa chọn trong bốn cấu hình thử nghiệm. Trên tập test, mô hình đạt MAE 252.934 g, RMSE 308.237 g và R² 0.832.

Mô hình cho thấy khả năng dự đoán tương đối tốt, tuy nhiên chưa vượt trội hơn M1 trên tất cả các chỉ số. Kết quả cần được đặt trong bối cảnh so sánh đầy đủ với các mô hình còn lại của nhóm.

## 6. Ứng dụng dự đoán

[Chèn ảnh giao diện notebook/app sau khi hoàn thiện.]

Ứng dụng dự đoán sử dụng pipeline đã huấn luyện để nhận các đặc trưng đầu vào gồm chiều dài vây, chiều dài mỏ, chiều sâu mỏ, loài và giới tính, sau đó trả về giá trị dự đoán `body_mass_g` theo đơn vị gram.

[Điền ví dụ dự đoán thực tế và mô tả biểu đồ tương tác sau khi kiểm tra notebook.]

## 7. Tài liệu tham khảo

1. Horst, A. M., Hill, A. P., & Gorman, K. B. (2020). *palmerpenguins: Palmer Archipelago (Antarctica) penguin data*. https://doi.org/10.5281/zenodo.3960218
2. Scikit-learn documentation: `SplineTransformer`.
3. Scikit-learn documentation: `Pipeline`, `ColumnTransformer`, `LinearRegression` và `OneHotEncoder`.