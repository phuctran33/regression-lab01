# Kế hoạch Lab 01 · 19/11–08/12/2026

Nhóm bắt đầu **19/11/2026** và nộp bài **08/12/2026**. Chốt mô hình ban đầu vào 22/11, kết quả và ứng dụng vào 29/11, báo cáo và video vào 06/12. Dành 07–08/12 để kiểm tra lần cuối và nộp.

## Phân công

| Thành viên | Tệp phụ trách | Thuật toán | Việc chung | Người rà soát |
| --- | --- | --- | --- | --- |
| TV1 · Tiết Trọng Khiêm | [M1](source/models/m1.py) | Ridge Regression | Viết mô tả bài toán, nguồn dữ liệu, cách chia tập và gom gói nộp | TV2 |
| TV2 · Nguyễn Khang Hy | [M2](source/models/m2.py) | Cubic Spline Basis Regression | Tổng hợp MAE, RMSE, R² và biểu đồ phần dư để so sánh bốn mô hình | TV1 |
| TV3 · Nguyễn Đức Hưng | [M3](source/models/m3.py) | Additive Natural Cubic Smoothing Spline | Tổng hợp kết quả EDA do TV4 bàn giao, ghép báo cáo và xuất PDF. [Mẫu báo cáo](report/report-template.md) | TV4 |
| TV4 · Trần Thiên Phúc | [M4](source/models/m4.py) | Huber Regression | Làm EDA, tích hợp notebook và ứng dụng, quay video. [Notebook chung](source/lab01.ipynb) | TV3 |

Bốn người tự chọn bốn thuật toán hồi quy tuyến tính khác nhau để dự đoán `body_mass_g`. Mỗi tệp có vị trí `ESTIMATOR` dành cho lựa chọn của người phụ trách. Cả nhóm chốt bộ biến đầu vào chung sau khi khám phá dữ liệu và dùng cùng pipeline, tập kiểm tra để so sánh kết quả.

Đề xuất TV1 làm đại diện nộp. Ngày 19/11, nhóm xác nhận đại diện, mã nhóm và giờ đóng bài trên Moodle.

## Cách phối hợp

1. **Một người một tệp mô hình.** Chọn và tìm hiểu thuật toán song song với EDA. Sau khi nhóm chốt bộ biến, điền `ESTIMATOR` rồi chạy riêng tệp của mình. Để xem hệ số và phần dư ở mục 4, khởi động lại kernel và chạy notebook. [Hướng dẫn chạy](README.md#phân-công-mô-hình)
2. **Giữ phần dùng chung có người ghép.** TV4 phụ trách EDA, gửi đề xuất bộ biến và ghép các thay đổi vào notebook hoặc pipeline chung. Mỗi người gửi đoạn viết và hình của mình cho TV3 để ghép vào một bản báo cáo. [Pipeline chung](source/modeling.py)
3. **Mỗi mô hình đi qua một lượt rà soát.** Người rà soát đọc tên thuật toán và cấu hình, kiểm tra log chạy và xem nhận xét có dẫn số liệu hay hình tương ứng. Hai cặp TV1–TV2 và TV3–TV4 rà chéo trước khi ghép. Khi vướng, gửi tên tệp, thông báo lỗi và bước tái hiện để người liên quan xử lý.
4. **Dùng cùng cách đánh giá.** Notebook chia train/test 80/20 với seed 42. Dùng dữ liệu train để khám phá mối quan hệ với biến đích và chọn đầu vào. Các bước điền thiếu, chuẩn hóa và mã hóa được fit trên tập huấn luyện. Nếu thử nhiều cấu hình, chọn bằng đánh giá chéo trên train. Cả bốn mô hình dùng cùng tập test ở lượt so sánh cuối.

### Luồng Git cho một thành viên

Sau khi TV4 cập nhật bộ biến chung, TV2 có thể lấy phiên bản mới nhất từ nhánh chính rồi tạo nhánh riêng theo ví dụ dưới đây. Thay `main` và `origin` nếu repo dùng tên khác.

```powershell
git switch main
git pull --ff-only
git switch -c model/m2
```

Sau khi sửa và chạy thử notebook, stage tệp mô hình rồi kiểm tra danh sách tệp trước khi commit. Output sinh ra khi chạy notebook có thể giữ ở máy cá nhân.

```powershell
git add source/models/m2.py
git diff --cached --name-only
git commit -m "Cập nhật mô hình M2"
git push -u origin model/m2
```

Mở pull request với ba dòng gồm thay đổi chính, kết quả chạy thử và phần cần góp ý. Sau khi rà chéo ngày 26–27/11, TV4 ghép các thay đổi vào nhánh chính và chạy lại notebook.

## 19–22/11 · EDA và triển khai ban đầu

**19/11 · TV4 phụ trách EDA.** Dùng `train_data` ở mục 2 của notebook để kiểm tra giá trị thiếu và phân bố khối lượng. Xem quan hệ với số đo cánh bơi, mỏ và so sánh theo loài, giới tính. Rà cột nào có sẵn lúc dự đoán. TV4 gửi bảng hoặc hình chính cùng đề xuất biến giữ hoặc loại trước buổi chốt ngày 20/11.

**19/11 · Chuẩn bị thuật toán song song.** TV1, TV2 và TV3 kiểm tra môi trường và chuẩn bị triển khai. TV2 chọn thuật toán M2 và báo nhóm trước khi bắt đầu huấn luyện. TV4 chuẩn bị M4 sau khi gửi kết quả EDA.

**20/11 · Chốt bộ biến chung.** TV4 trình bày đề xuất từ EDA. Cả nhóm thống nhất bộ biến và ghi lý do chọn. TV4 cập nhật `FEATURES` để cả bốn người lấy cùng cấu hình. [Cấu hình đầu vào](source/modeling.py)

**20–22/11 · Triển khai song song sau khi chốt biến.** Mỗi người điền `ESTIMATOR` trong tệp mình phụ trách và chạy thử từ terminal với cấu hình ban đầu. Ghi tên thuật toán, lý do chọn và các tham số có thể tinh chỉnh.

**Bàn giao 22/11.** Kết quả EDA, bộ biến chung và bốn tệp mô hình chạy được với cấu hình ban đầu.

## 23–29/11 · Tinh chỉnh mô hình và làm ứng dụng

**23–25/11 · Tinh chỉnh trên tập train.** Cả nhóm thống nhất cách chia fold và chỉ số chính, chẳng hạn MAE. Mỗi người đánh giá cấu hình ban đầu bằng cross-validation và thử siêu tham số phù hợp với thuật toán của mình. Ghi cấu hình đề xuất cùng kết quả trên train. Dùng tập test cho lượt so sánh sau khi chốt cấu hình.

**23–25/11 · TV4 làm ứng dụng song song.** Dùng các mô hình ban đầu để kiểm tra mục 5. Giao diện cần cho phép chọn mô hình, nhập các biến đã chốt, xem khối lượng dự đoán theo gram và xem biểu đồ khi thay đổi đầu vào. [Giao diện trong notebook](source/lab01.ipynb)

| Người | Rà soát và bàn giao 26–29/11 |
| --- | --- |
| TV1 | Rà M2, gửi cấu hình chốt của M1 và ghi lại nguồn dữ liệu, cách chia tập để đưa vào báo cáo. |
| TV2 | Rà M1, gửi cấu hình chốt của M2. Từ phiên bản đã ghép, lấy bảng MAE, RMSE, R² và biểu đồ phần dư. |
| TV3 | Rà M4, gửi cấu hình chốt của M3. Nhận bảng EDA và ghi chú chọn biến từ TV4, rồi tập hợp ghi chú chọn cấu hình của bốn người. |
| TV4 | Rà M3, chốt M4 và ghép bốn cấu hình vào notebook. Chạy **Restart Kernel and Run All Cells**, chuyển kết quả cho TV2 và thử chọn mô hình, đổi đầu vào, dự đoán, xem biểu đồ. |

**26–27/11.** Rà chéo và chốt cấu hình của bốn mô hình. **28/11.** Ghép code, chạy trên cùng tập test rồi kiểm thử giao diện với các mô hình đã chốt. **Bàn giao 29/11.** Bảng test, bảng hệ số của bốn mô hình và ứng dụng chạy được từ nhập liệu đến dự đoán và biểu đồ. Ghi mã commit hoặc ngày chạy cạnh bảng để báo cáo và video dùng cùng phiên bản.

Mỗi người gửi TV3 hướng tiếp cận, lý do chọn thuật toán, siêu tham số và một nhận xét có số liệu hoặc hình đi kèm. Với biến số đã chuẩn hóa, ghi đơn vị hệ số theo gram trên một độ lệch chuẩn. Với biến phân loại, nêu mức tham chiếu. Giữ các biến còn lại cố định khi diễn giải từng hệ số.

## 30/11–06/12 · Báo cáo và video

| Người | 30/11–02/12 | 03–06/12 |
| --- | --- | --- |
| TV1 | Viết mô tả bài toán, nguồn dữ liệu và cách chia tập. Chuẩn bị thư mục nộp đúng cấu trúc. | Gom PDF, video, mã nguồn và dữ liệu vào ZIP. |
| TV2 | Viết phần so sánh mô hình. Đối chiếu bảng và biểu đồ với notebook ở phiên bản đã chốt. | Đọc soát PDF lần cuối. So lại các số chính được nhắc trong video với bảng kết quả. |
| TV3 | Viết phần EDA, ghép báo cáo, chèn hình ứng dụng và viết kết luận dựa trên kết quả. Đề gợi ý 10–20 trang. | Xuất PDF, kiểm tra hình, bảng và danh mục tài liệu tham khảo. Gửi bản chốt cho TV1. |
| TV4 | Thử lại ứng dụng, sửa lỗi nếu có và tập demo. Gửi TV3 ảnh giao diện kèm mô tả luồng dự đoán. | Quay video, xem lại hình và tiếng rồi gửi TV1. |

**Kịch bản video.** Mở notebook và giới thiệu biến đích. Chọn M1, đổi một đầu vào nhóm đã giữ sau EDA rồi bấm **Dự đoán** để xem kết quả tính bằng gram. Chọn M4, đổi một đầu vào khác và xem dự đoán hoặc biểu đồ tương ứng. Cuối video, hiện bảng MAE, RMSE và R² của bốn mô hình.

**Bàn giao 06/12.** Báo cáo PDF, video demo và gói ZIP đã gom đủ tệp.

## 07–08/12 · Kiểm tra và nộp

- **07/12.** TV1 cùng TV4 giải nén `Group [Group-ID].zip` vào thư mục mới, chạy lại notebook, mở PDF và video, rồi xử lý lỗi phát hiện được.
- **08/12.** Đại diện nộp trước giờ đóng bài trên Moodle và báo nhóm khi trạng thái nộp đã được xác nhận.

### Kiểm tra gói nộp

- [ ] `data/` có CSV Palmer Penguins.
- [ ] `source/` có notebook, bốn tệp mô hình và pipeline dùng chung.
- [ ] `requirements.txt` có trong ZIP và hướng dẫn chạy dùng đúng đường dẫn.
- [ ] Video mở được, có thao tác nhập dữ liệu, dự đoán và xem biểu đồ.
- [ ] Báo cáo PDF có mã nhóm, bốn thành viên, mô tả bài toán, dữ liệu, bốn mô hình, đánh giá, kết luận, ứng dụng và tài liệu tham khảo.
- [ ] Notebook trong bản giải nén chạy từ đầu và hiển thị đủ bốn kết quả.
