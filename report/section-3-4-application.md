## 3.4 Ứng dụng dự đoán

Giao diện ở mục 5 của notebook dùng widget Jupyter. Chạy lại toàn bộ notebook theo README, chọn mô hình, điền ba số đo bằng mm, chọn loài và giới tính rồi bấm **Dự đoán**. Slider sử dụng khoảng quan sát trên train và dropdown chỉ liệt kê các mức hợp lệ. Kết quả là `body_mass_g` theo gram. OLS và M3 hiện chạy được. M1, M2 và M4 hiện được ghi là **Chưa sẵn sàng** trong bảng huấn luyện và sẽ xuất hiện trong bộ chọn khi estimator của từng mô hình được khai báo. [Hướng dẫn notebook](../README.md#chạy-notebook) · [Mở notebook](../source/lab01.ipynb)

Giao diện gọi `predict` trên các pipeline đã fit ở mục 3 của notebook. Một dự đoán dùng hàng đầu vào hiện tại. Biểu đồ tạo một dãy hàng tương tự rồi chỉ thay cột đang chọn, giữ nguyên bốn đầu vào còn lại. Với biến phân loại, mô hình thử lần lượt các mức trong dropdown. Khi bật **So sánh OLS và M3**, biểu đồ số đo có thêm ô chênh lệch dự đoán M3 − OLS theo gram để thấy khác biệt nhỏ giữa hai đường. Các điểm nền là quan sát train với những đầu vào khác nhau, còn dấu sao xác định giá trị đang nhập và dự đoán của mô hình chính. Bảng chỉ số ở mục 4 dùng tập test. [Mã giao diện](../source/prediction_app.py)

### Kịch bản demo

Giữ **So sánh OLS và M3** bật. Sau mỗi thay đổi, bấm **Dự đoán** để cập nhật số và hình.

| Demo | Đầu vào ban đầu | Giá trị ban đầu | Thao tác và giá trị cập nhật |
| --- | --- | ---: | --- |
| OLS, xu hướng theo cánh bơi | Adelie cái, cánh bơi 190 mm, mỏ dài 39 mm, mỏ sâu 18 mm | 3.485,6 g | Chọn biểu đồ cánh bơi, tăng lên 205 mm. OLS dự đoán 3.680,2 g và M3 dự đoán 3.680,6 g. |
| M3, xu hướng theo độ sâu mỏ | Gentoo đực, cánh bơi 215 mm, mỏ dài 49 mm, mỏ sâu 15 mm | 5.281,2 g | Chọn biểu đồ độ sâu mỏ, tăng lên 16 mm. M3 dự đoán 5.366,6 g và OLS dự đoán 5.360,1 g. |
| Tiếp tục demo M3 với biến phân loại | Giữ các số đo sau thay đổi ở dòng trên | 5.366,6 g | Chọn biểu đồ giới tính và đổi sang female. M3 dự đoán 4.957,8 g. |

Ở demo OLS, khi thay duy nhất chiều dài cánh bơi, dự đoán tăng theo đường thẳng với độ dốc cố định. Ở demo M3, thay độ sâu mỏ đánh giá spline tại một điểm mới. Đổi giới tính thay hiệu ứng phân loại trong M3 trong khi các đóng góp của ba spline giữ nguyên. Các kịch bản là biến thiên có điều kiện theo mô hình, với các đầu vào do người dùng giữ cố định.
