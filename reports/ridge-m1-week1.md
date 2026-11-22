# Lab 01 – Dự đoán khối lượng chim cánh cụt

**Phần đóng góp:** M1 · Ridge Regression · Tuần 1  
**Phụ trách:** TV1 · Tiết Trọng Khiêm · MSSV 24127058  
**Ngày chạy:** 21/11/2026 · 

| Thành viên | Công việc thực hiện | Mô hình phụ trách |
| --- | --- | --- |
| Tiết Trọng Khiêm | Tìm hiểu pipeline chung, khai báo Ridge với cấu hình ban đầu, chạy thử và ghi nhận kết quả | M1 |

## 1. Mô tả bài toán và động cơ

Mục tiêu là dự đoán khối lượng chim cánh cụt `body_mass_g` (g) từ chiều dài cánh bơi, chiều dài mỏ, độ sâu mỏ, loài và giới tính. M1 dùng bộ đầu vào và cách chia tập trong [pipeline chung](../source/modeling.py) để có thể so sánh với các mô hình khác của nhóm.

Bài toán cho phép khảo sát mối liên hệ giữa đặc điểm cơ thể và khối lượng. Phần M1 sử dụng Ridge Regression, một mô hình hồi quy tuyến tính có điều chuẩn L2, và ghi nhận kết quả với cấu hình ban đầu.

## 2. Dữ liệu, tiền xử lý và trực quan hóa

Dữ liệu lấy từ bộ Palmer Penguins, bản CSV tại [data/penguins.csv](../data/penguins.csv), gồm 344 dòng và 8 cột. Các số đo cơ thể dùng đơn vị millimet, khối lượng dùng gram; loài, đảo và giới tính là biến phân loại. Đảo và năm nằm ngoài bộ đầu vào hiện tại.

Bỏ 2 dòng thiếu biến đích rồi chia train/test 80/20 với `random_state=42`, thu được **273 dòng train và 69 dòng test**. Tập train có 5 dòng thiếu giới tính. Pipeline chung thực hiện:

- Biến số: điền thiếu bằng trung vị rồi chuẩn hóa bằng `StandardScaler`.
- Biến phân loại: điền thiếu bằng mức phổ biến rồi one-hot, bỏ mức tham chiếu đầu tiên và cho phép xử lý mức chưa gặp.

Các bước tiền xử lý được fit trên train và áp dụng lại cho test. Phần EDA, hình phân bố khối lượng và quan hệ giữa khối lượng với đầu vào được ghép từ kết quả TV4. Việc sử dụng loài và giới tính giả định người dùng biết các thông tin này lúc dự đoán.

## 3. Thiết kế mô hình

Ridge thêm phạt L2 vào hồi quy tuyến tính để thu nhỏ hệ số. Cách này phù hợp để khảo sát khi các số đo cơ thể có thể tương quan với nhau, vì giúp hạn chế hệ số lớn. Hiệu quả thực tế cần được đánh giá bằng số liệu.

Hàm mục tiêu của Ridge là:

\[
\sum_{i=1}^{n}(y_i-\beta_0-\mathbf{x}_i^\top\boldsymbol{\beta})^2
+\alpha\sum_{j=1}^{p}\beta_j^2.
\]

Alpha quyết định mức điều chuẩn; alpha lớn làm tăng mức phạt. Hệ số chặn không bị phạt. Cấu hình ban đầu dùng **alpha = 1.0**, trùng giá trị mặc định của scikit-learn Ridge, làm mốc chạy thử có thể tái hiện. Xem tài liệu [1].

| Mô hình | Thuật toán tự chọn | Siêu tham số | Giả thuyết và lý do chọn |
| --- | --- | --- | --- |
| M1 | Ridge Regression | `alpha=1.0`, các tham số khác giữ mặc định | Điều chuẩn L2 giúp hạn chế hệ số lớn khi đầu vào có tương quan |

Trong [source/models/m1.py](../source/models/m1.py), `ESTIMATOR = Ridge(alpha=1.0)` khai báo mô hình. `build_model()` ghép estimator vào pipeline tiền xử lý chung. Khi chạy tệp như module, `run_model()` tải dữ liệu, chia tập, fit pipeline trên train và tính chỉ số trên test.

## 4. Đánh giá và so sánh

MAE là sai số tuyệt đối trung bình theo gram. RMSE cũng tính bằng gram và nhấn mạnh hơn các sai số lớn. Hai chỉ số này càng thấp càng tốt. R² so sánh tổng bình phương sai số với mốc dự đoán trung bình trên tập đánh giá; giá trị càng cao càng tốt và không phải tỷ lệ phần trăm dự đoán chính xác.

Kết quả chạy M1 với alpha 1.0 trên 69 dòng test:

| Mô hình | MAE (g) | RMSE (g) | R² |
| --- | ---: | ---: | ---: |
| M1 · Ridge, alpha = 1.0 | 254.103 | 301.776 | 0.839 |

Trên tập kiểm tra này, sai số tuyệt đối trung bình khoảng 254 g. RMSE lớn hơn MAE, phản ánh ảnh hưởng mạnh hơn của những dự đoán lệch nhiều. Đây là kết quả chạy thử của cấu hình ban đầu.

Khi ghép báo cáo nhóm, bổ sung chỉ số M2–M4 trên cùng tập test và đồ thị phần dư của M1 từ notebook. Với các biến số đã chuẩn hóa, hệ số được diễn giải theo gram trên một độ lệch chuẩn khi giữ các đầu vào khác cố định. Với biến phân loại, cần ghi mức tham chiếu thực tế từ encoder. Phần tuần 1 chưa kết luận về mô hình tốt nhất của nhóm hoặc dạng phân bố phần dư.

## 5. Kết luận

M1 đã khai báo Ridge Regression với alpha 1.0 và chạy được bằng pipeline chung. Kết quả thử là MAE 254.103 g, RMSE 301.776 g và R² 0.839 trên tập test.

Kết quả đáp ứng phần bàn giao tuần 1 về mô hình chạy được và ghi chú cấu hình ban đầu. Nhóm cần tổng hợp kết quả các mô hình còn lại và chẩn đoán sai số trước khi chọn mô hình minh họa. Các nhận xét hiện tại giới hạn trong bộ biến và cách chia dữ liệu đang dùng.

## 6. Ứng dụng dự đoán

Giao diện trong [source/lab01.ipynb](../source/lab01.ipynb) nhận các đầu vào thuộc `FEATURES`, cho phép chọn M1 và hiển thị khối lượng dự đoán theo gram. Pipeline xử lý đầu vào rồi dự đoán bằng Ridge đã fit. Biểu đồ kịch bản thay đổi một đầu vào trong khi giữ các đầu vào còn lại cố định.

Ảnh giao diện và ví dụ trong video demo được bổ sung khi ghép phần ứng dụng của nhóm. Để chạy riêng M1, mở terminal tại thư mục dự án, dùng Python của môi trường đã cài `requirements.txt` và chạy:

```powershell
python -m source.models.m1
```

Để sử dụng cấu hình mới trong notebook, restart kernel và chạy lại các cell.

## 7. Tài liệu tham khảo

1. Scikit-learn 1.8.0. [Ridge](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.Ridge.html).
2. Horst, A. M., Hill, A. P., & Gorman, K. B. (2020). *palmerpenguins: Palmer Archipelago (Antarctica) penguin data*. [DOI 10.5281/zenodo.3960218](https://doi.org/10.5281/zenodo.3960218).
