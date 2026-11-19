# Lab 01 – Dự đoán khối lượng chim cánh cụt

**Nhóm** [Mã nhóm]  
**Học phần** [Tên học phần, lớp]  
**Thành viên** [Họ tên và MSSV của 4 thành viên]  
**Ngày nộp** [Ngày/tháng/năm]

| Thành viên | Công việc thực hiện | Mô hình phụ trách |
| --- | --- | --- |
| [Thành viên 1 – MSSV] | [Công việc] | M1 |
| [Thành viên 2 – MSSV] | [Công việc] | M2 |
| [Thành viên 3 – MSSV] | [Công việc] | M3 |
| [Thành viên 4 – MSSV] | [Công việc] | M4 |

## 1. Mô tả bài toán và động cơ

- Mục tiêu đề xuất là dự đoán `body_mass_g` (g) từ các đặc trưng quan sát được. [Viết lại mục tiêu theo phạm vi bài làm và liệt kê các đầu vào nhóm thực sự dùng.]
- [Vì sao nhóm chọn bài toán này? Nhóm muốn kiểm tra mối quan hệ nào qua các mô hình?]

## 2. Dữ liệu, tiền xử lý và trực quan hóa

- Nguồn dữ liệu là bộ `penguins` của Palmer Penguins. [Điền số dòng, số cột, kiểu biến, đơn vị và số giá trị thiếu sau khi kiểm tra CSV.]
- [Mô tả các bước làm sạch, xử lý thiếu, mã hóa, tỷ lệ chia tập và seed. Giải thích quyết định nào được học từ tập huấn luyện.]
- [Chèn biểu đồ phân bố biến đích và biểu đồ quan hệ giữa đầu vào với biến đích. Viết nhận xét trực tiếp từ hình.]

## 3. Thiết kế mô hình

[Trình bày dạng phương trình hồi quy tuyến tính và cách đọc hệ số của biến số hoặc biến phân loại theo nhóm tham chiếu.]

| Mô hình | Đầu vào thực tế | Biến đổi hoặc mã hóa | Giả thuyết và lý do chọn |
| --- | --- | --- | --- |
| M1 | [Danh sách cột] | [Các bước] | [Lý do] |
| M2 | [Danh sách cột] | [Các bước] | [Lý do] |
| M3 | [Danh sách cột] | [Các bước] | [Lý do] |
| M4 | [Danh sách cột] | [Các bước] | [Lý do] |

## 4. Đánh giá và so sánh

[Nêu cách tính và cách đọc MAE, RMSE và R². Các chỉ số test cần được tính trên cùng tập kiểm tra.]

| Mô hình | MAE (g) | RMSE (g) | R² |
| --- | ---: | ---: | ---: |
| M1 | [ ] | [ ] | [ ] |
| M2 | [ ] | [ ] | [ ] |
| M3 | [ ] | [ ] | [ ] |
| M4 | [ ] | [ ] | [ ] |

- [So sánh mức chênh lệch chỉ số giữa các mô hình. Chèn đồ thị phần dư hoặc giá trị thật và dự đoán, rồi mô tả những gì quan sát được.]
- [Giải thích dấu, độ lớn và đơn vị của các hệ số tiêu biểu. Với mô hình nhiều biến, giữ cố định các biến còn lại khi diễn giải.]

## 5. Kết luận

[Trả lời lại câu hỏi nghiên cứu từ các kết quả trên. Nêu mô hình nhóm chọn để minh họa, chỉ số liên quan và giới hạn quan sát được.]

## 6. Ứng dụng dự đoán

- [Chèn ảnh giao diện. Mô tả đầu vào, nút dự đoán và đầu ra theo đơn vị g.]
- [Giải thích pipeline nhận giá trị từ giao diện và tạo dự đoán. Diễn giải cách biểu đồ thay đổi một biến trong khi giữ các đầu vào khác cố định.]
- [Ghi ví dụ đã chạy trong video demo và kết quả hiển thị.]

## 7. Tài liệu tham khảo

1. Horst, A. M., Hill, A. P., & Gorman, K. B. (2020). *palmerpenguins: Palmer Archipelago (Antarctica) penguin data*. [DOI 10.5281/zenodo.3960218](https://doi.org/10.5281/zenodo.3960218).
2. [Bổ sung tài liệu và các công cụ tham khảo nhóm đã sử dụng.]
