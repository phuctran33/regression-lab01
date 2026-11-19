# Kế hoạch 3 tuần · Lab 01 hồi quy tuyến tính

**Ngày 1** là ngày nhóm bắt đầu. Điền tên thật, mã nhóm và hạn Moodle ở buổi đầu. Kế hoạch dành Ngày 20 để kiểm tra gói nộp và Ngày 21 để nộp. Căn hai mốc này theo hạn thực tế.

## Phân công

| Thành viên | Tệp phụ trách từ Ngày 1 | Thuật toán tự chọn | Việc chung từ Tuần 2 | Người rà soát |
| --- | --- | --- | --- | --- |
| TV1 · [Tên, MSSV] | [M1](source/models/m1.py) | [Điền lựa chọn của TV1] | Viết mô tả bài toán, nguồn dữ liệu, cách chia tập và gom gói nộp | TV2 |
| TV2 · [Tên, MSSV] | [M2](source/models/m2.py) | [Điền lựa chọn của TV2] | Tổng hợp MAE, RMSE, R² và biểu đồ phần dư để so sánh bốn mô hình | TV1 |
| TV3 · [Tên, MSSV] | [M3](source/models/m3.py) | [Điền lựa chọn của TV3] | Viết phần EDA từ kết quả mình làm ở tuần 1, ghép báo cáo và xuất PDF. [Mẫu báo cáo](report/report-template.md) | TV4 |
| TV4 · [Tên, MSSV] | [M4](source/models/m4.py) | [Điền lựa chọn của TV4] | Tích hợp notebook và ứng dụng ở tuần 2, quay video ở tuần 3. [Notebook chung](source/lab01.ipynb) | TV3 |

Bốn người tự chọn bốn thuật toán hồi quy tuyến tính khác nhau để dự đoán `body_mass_g`. Mỗi tệp có vị trí `ESTIMATOR` dành cho lựa chọn của người phụ trách. Cả nhóm chốt bộ biến đầu vào chung sau khi khám phá dữ liệu và dùng cùng pipeline, tập kiểm tra để so sánh kết quả.

Đề xuất TV1 làm đại diện nộp. Nhóm có thể đổi người ở Ngày 1 và ghi tên đại diện cùng hạn nộp.

## Cách làm việc hằng tuần

1. **Một người một tệp mô hình.** Chọn và tìm hiểu thuật toán song song với EDA. Sau khi nhóm chốt bộ biến, điền `ESTIMATOR` rồi chạy riêng tệp của mình. Để xem hệ số và phần dư ở mục 4, khởi động lại kernel và chạy notebook. [Hướng dẫn chạy](README.md#phân-công-mô-hình)
2. **Giữ phần dùng chung có người ghép.** TV3 phụ trách EDA và gửi đề xuất bộ biến. TV4 ghép các thay đổi vào notebook hoặc pipeline chung. Mỗi người gửi đoạn viết và hình của mình cho TV3 để ghép vào một bản báo cáo. [Pipeline chung](source/modeling.py)
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

Mở pull request với ba dòng gồm thay đổi chính, kết quả chạy thử và phần cần góp ý. Từ tuần 2, người rà soát góp ý trên pull request, sau đó TV4 ghép các thay đổi vào nhánh chính và chạy lại notebook.

## Tuần 1 · EDA, chốt đầu vào và làm bốn mô hình

**Ngày 1–2 · TV3 phụ trách toàn bộ EDA.** Dùng `train_data` ở mục 2 của notebook để kiểm tra giá trị thiếu và phân bố khối lượng. Xem quan hệ giữa khối lượng với các số đo cánh bơi, mỏ và so sánh theo loài, giới tính. Rà cột nào có sẵn lúc dự đoán. Cuối Ngày 2, TV3 gửi bảng hoặc hình chính cùng đề xuất biến giữ hoặc loại và lý do.

**Chuẩn bị thuật toán song song.** TV1, TV2 và TV4 kiểm tra môi trường và tìm hiểu thuật toán cho M1, M2 và M4. TV3 chọn thuật toán cho M3 trong cùng giai đoạn. Cả bốn báo lựa chọn trong nhóm để tránh trùng.

**Ngày 3 · Chốt bộ biến chung.** TV3 trình bày đề xuất dựa trên EDA. Cả nhóm thống nhất bộ biến và lưu lý do chọn vào ghi chú. Năm biến trong template là điểm xuất phát. TV4 cập nhật `FEATURES` để cả bốn người lấy cùng cấu hình. [Cấu hình đầu vào](source/modeling.py)

**Ngày 4–7 · Triển khai song song.** Mỗi người lấy bộ biến chung, điền `ESTIMATOR` trong tệp đã nhận và chạy thử từ terminal với cấu hình ban đầu. Ghi tên thuật toán, lý do chọn và các tham số có thể tinh chỉnh trong tuần 2.

**Bàn giao Ngày 7.** Nhóm có kết quả EDA và bộ biến đã chốt. Bốn tệp mô hình chạy được với cấu hình ban đầu và sẵn sàng cho lượt tinh chỉnh.

## Tuần 2 · Tinh chỉnh mô hình và làm ứng dụng

**Ngày 8–10 · Tinh chỉnh trên tập train.** Cả nhóm thống nhất cách chia fold và chỉ số chính, chẳng hạn MAE. Mỗi người đánh giá cấu hình ban đầu bằng cross-validation và thử siêu tham số phù hợp với thuật toán của mình. Ghi cấu hình đề xuất cùng kết quả trên train. Dùng tập test cho lượt so sánh sau khi chốt cấu hình.

**Ngày 8–10 · TV4 chuẩn bị ứng dụng song song.** Dùng các mô hình chạy được từ tuần 1 để kiểm tra mục 5. Giao diện cần cho phép chọn mô hình, nhập các biến đã chốt, xem khối lượng dự đoán theo gram và xem biểu đồ khi thay đổi đầu vào. [Giao diện trong notebook](source/lab01.ipynb)

| Người | Rà soát và bàn giao Ngày 11–14 |
| --- | --- |
| TV1 | Rà M2, gửi cấu hình chốt của M1 và ghi lại nguồn dữ liệu, cách chia tập để tuần 3 đưa vào báo cáo. |
| TV2 | Rà M1, gửi cấu hình chốt của M2. Từ phiên bản đã ghép, lấy bảng MAE, RMSE, R² và biểu đồ phần dư. |
| TV3 | Rà M4, gửi cấu hình chốt của M3. Gom bảng EDA và ghi chú chọn biến, chọn cấu hình của bốn người. |
| TV4 | Rà M3, chốt M4 và ghép bốn cấu hình vào notebook. Chạy **Restart Kernel and Run All Cells**, chuyển kết quả cho TV2 và thử chọn mô hình, đổi đầu vào, dự đoán, xem biểu đồ. |

**Ngày 12.** Rà chéo xong và chốt cấu hình của bốn mô hình. **Ngày 13.** Ghép code, chạy trên cùng tập test rồi kiểm thử giao diện với các mô hình đã chốt. **Bàn giao Ngày 14.** Có bảng test, bảng hệ số của bốn mô hình và ứng dụng chạy được từ nhập liệu đến dự đoán và biểu đồ. Ghi mã commit hoặc ngày chạy cạnh bảng để tuần 3 dùng cùng phiên bản.

Mỗi người gửi TV3 hướng tiếp cận, lý do chọn thuật toán, siêu tham số và một nhận xét có số liệu hoặc hình đi kèm. Với biến số đã chuẩn hóa, ghi đơn vị hệ số theo gram trên một độ lệch chuẩn. Với biến phân loại, nêu mức tham chiếu. Giữ các biến còn lại cố định khi diễn giải từng hệ số.

## Tuần 3 · Báo cáo, video và nộp

| Người | Ngày 15–17 | Ngày 18–20 |
| --- | --- | --- |
| TV1 | Viết mô tả bài toán, nguồn dữ liệu và cách chia tập. Chuẩn bị thư mục nộp đúng cấu trúc. | Tạo ZIP, giải nén sang thư mục mới và kiểm tra đủ tệp. Cùng TV4 chạy lại từ bản giải nén. |
| TV2 | Viết phần so sánh mô hình. Đối chiếu bảng và biểu đồ với notebook ở phiên bản đã chốt. | Đọc soát PDF lần cuối. So lại các số chính được nhắc trong video với bảng kết quả. |
| TV3 | Viết phần EDA, ghép báo cáo, chèn hình ứng dụng và viết kết luận dựa trên kết quả. Đề gợi ý 10–20 trang. | Xuất PDF, kiểm tra hình, bảng và danh mục tài liệu tham khảo. Gửi bản chốt cho TV1. |
| TV4 | Thử lại ứng dụng, sửa lỗi nếu có và tập demo. Gửi TV3 ảnh giao diện kèm mô tả luồng dự đoán. | Quay video, xem lại hình và tiếng rồi gửi TV1. Kiểm tra từ gói giải nén rằng cả bốn mô hình hiện trong notebook và ứng dụng dự đoán được. |

**Kịch bản video.** Mở notebook và giới thiệu biến đích. Chọn M1, đổi một đầu vào nhóm đã giữ sau EDA rồi bấm **Dự đoán** để xem kết quả tính bằng gram. Chọn M4, đổi một đầu vào khác và xem dự đoán hoặc biểu đồ tương ứng. Cuối video, hiện bảng MAE, RMSE và R² của bốn mô hình.

**Bàn giao Ngày 20.** TV1 kiểm tra `Group [Group-ID].zip` theo danh sách bên dưới. **Ngày 21.** Đại diện nộp trước hạn Moodle và báo nhóm khi đã thấy trạng thái nộp thành công.

### Kiểm tra gói nộp

- [ ] `data/` có CSV Palmer Penguins.
- [ ] `source/` có notebook, bốn tệp mô hình và pipeline dùng chung.
- [ ] `requirements.txt` có trong ZIP và hướng dẫn chạy dùng đúng đường dẫn.
- [ ] Video mở được, có thao tác nhập dữ liệu, dự đoán và xem biểu đồ.
- [ ] Báo cáo PDF có mã nhóm, bốn thành viên, mô tả bài toán, dữ liệu, bốn mô hình, đánh giá, kết luận, ứng dụng và tài liệu tham khảo.
- [ ] Notebook trong bản giải nén chạy từ đầu và hiển thị đủ bốn kết quả.
