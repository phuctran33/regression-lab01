# Kế hoạch 3 tuần · Lab 01 hồi quy tuyến tính

**Ngày 1** là ngày nhóm bắt đầu. Điền tên thật, mã nhóm và hạn Moodle ở buổi đầu. Kế hoạch dành Ngày 20 để kiểm tra gói nộp và Ngày 21 để nộp. Căn hai mốc này theo hạn thực tế.

## Phân công

| Thành viên | Mô hình phụ trách | Việc chung phụ trách | Người rà soát |
| --- | --- | --- | --- |
| TV1 · [Tên, MSSV] | M1 dùng chiều dài cánh bơi. [m1.py](source/models/m1.py) | Kiểm tra dữ liệu, chuẩn bị phần mô tả và biểu đồ EDA, gom gói nộp | TV2 |
| TV2 · [Tên, MSSV] | M2 thêm chiều dài và độ sâu mỏ. [m2.py](source/models/m2.py) | Tổng hợp MAE, RMSE, R² và biểu đồ phần dư để so sánh bốn mô hình | TV1 |
| TV3 · [Tên, MSSV] | M3 thêm loài. [m3.py](source/models/m3.py) | Ghép báo cáo, rà trích dẫn và xuất PDF. [Mẫu báo cáo](report/report-template.md) | TV4 |
| TV4 · [Tên, MSSV] | M4 thêm giới tính. [m4.py](source/models/m4.py) | Tích hợp notebook, kiểm tra giao diện và phụ trách video. [Notebook chung](source/lab01.ipynb) | TV3 |

Đề xuất TV1 làm đại diện nộp. Nhóm có thể đổi người ở Ngày 1 và ghi tên đại diện cùng hạn nộp.

## Cách làm việc hằng tuần

1. **Một người một tệp mô hình.** Sửa `FEATURES` hoặc `build_model()` trong tệp được giao. Sau khi sửa, khởi động lại kernel và chạy toàn bộ notebook. Ở mục 4, chọn đúng mô hình để xem hệ số và phần dư. [Hướng dẫn chạy](README.md#chạy-notebook)
2. **Giữ phần dùng chung có người ghép.** Gửi yêu cầu sửa notebook hoặc tiền xử lý chung cho TV4. TV1 rà các thay đổi liên quan đến dữ liệu. Mỗi người gửi đoạn viết và hình của mình cho TV3 để ghép vào một bản báo cáo. [Pipeline chung](source/modeling.py)
3. **Mỗi mô hình đi qua một lượt rà soát.** Người rà soát đọc danh sách biến, kiểm tra log chạy và xem nhận xét có dẫn số liệu hay hình tương ứng. Hai cặp TV1–TV2 và TV3–TV4 rà chéo trước khi ghép. Khi vướng, gửi tên tệp, thông báo lỗi và bước tái hiện để người liên quan xử lý.
4. **Dùng cùng cách đánh giá.** Notebook hiện chia train/test 80/20 với seed 42. Dùng cùng tập kiểm tra cho cả bốn mô hình. Các bước điền thiếu và mã hóa được fit trên tập huấn luyện. Nếu thử nhiều cấu hình cho cùng một mô hình, dùng dữ liệu train để chọn cấu hình rồi lấy kết quả test ở lượt so sánh cuối.

### Luồng Git cho một thành viên

Ví dụ TV2 bắt đầu từ nhánh chính của repo nhóm rồi tạo nhánh riêng. Thay `main` và `origin` nếu repo dùng tên khác.

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

Mở pull request với ba dòng gồm thay đổi chính, kết quả chạy thử và phần cần góp ý. Người rà soát góp ý trên pull request, sau đó TV4 ghép các thay đổi vào nhánh chính và chạy lại notebook.

## Tuần 1 · Chạy được và hiểu dữ liệu

**Ngày 1.** Cả nhóm chốt mã nhóm, hạn nộp, người phụ trách và repo chung. Từng người mở notebook, chọn môi trường Python theo README và chạy từ đầu để có cùng kết quả mẫu.

| Người | Ngày 2–4 | Ngày 5–7 |
| --- | --- | --- |
| TV1 | Kiểm tra số dòng, kiểu biến, giá trị thiếu và đơn vị đo. Chạy M1, xem phân bố khối lượng và đồ thị cánh bơi so với khối lượng. | Gửi bảng tóm tắt dữ liệu, hai biểu đồ EDA và đoạn mô tả cách chia tập. Rà biến đầu vào M2. |
| TV2 | Chạy M2 với ba số đo. Làm mẫu bảng MAE, RMSE, R² và ghi lại đơn vị của từng chỉ số. | Viết giả thuyết vì sao hai số đo mỏ có thể bổ sung thông tin. Rà cách đọc hệ số g/mm của M1. |
| TV3 | Chạy M3, kiểm tra các mức của `species` và mức tham chiếu notebook hiển thị. Chuẩn bị khung báo cáo có đầy đủ đề mục. | Ghi nguồn Palmer Penguins và nhận xét ban đầu về biến loài. Rà cách M4 dùng biến phân loại. |
| TV4 | Chạy M4, kiểm tra giá trị thiếu của `sex`. Thử chọn M1–M4 trong mục 5 và dự đoán ở cả biến số lẫn biến phân loại. | Ghi hai tình huống demo, chia sẻ các lỗi chạy và kiểm tra nhánh chính nạp được đủ bốn mô hình. |

**Bàn giao Ngày 7.** Bốn người chạy được template. Nhóm có bảng dữ liệu thiếu, hai biểu đồ EDA, danh sách đầu vào từng mô hình và sườn báo cáo. Chốt cấu hình nào sẽ được thử trong tuần 2.

## Tuần 2 · Hoàn thiện mô hình và chốt kết quả

| Người | Ngày 8–10 | Ngày 11–14 |
| --- | --- | --- |
| TV1 | Hoàn thiện M1 và giải thích chiều dài cánh bơi liên hệ thế nào với khối lượng dự đoán. Ghi hệ số theo g/mm và xem phần dư M1. | Gửi mục mô tả bài toán, dữ liệu, tiền xử lý và trực quan hóa cho TV3. Rà pull request M2. |
| TV2 | Hoàn thiện M2. Xem hệ số của hai biến số đo mỏ và chuẩn bị bảng để so sánh với M1. | Rà M1. Từ phiên bản đã ghép, xuất bảng MAE, RMSE, R² và biểu đồ phần dư tiêu biểu. Ghi mức chênh lệch M1–M2. |
| TV3 | Hoàn thiện M3. Kiểm tra cách mã hóa loài và diễn giải hệ số theo mức tham chiếu. | Rà M4. Dùng bảng đã chốt để so sánh M2–M3. Ghép phần thiết kế và đánh giá, thống nhất tên mô hình, đơn vị và chú thích hình. |
| TV4 | Hoàn thiện M4. Kiểm tra cách xử lý thiếu của giới tính và thử giao diện khi chuyển qua các mô hình. | Rà M3, ghép các pull request đã duyệt và chạy **Restart Kernel and Run All Cells**. Chuyển kết quả cho TV2, xem chênh lệch M3–M4 rồi kiểm thử lại hai tình huống demo. |

**Ngày 12.** Hoàn thành rà soát. **Ngày 13.** TV4 chạy notebook từ nhánh chính đã ghép và chuyển kết quả cho TV2. **Bàn giao Ngày 14.** Có bảng test, bảng hệ số của bốn mô hình và bản nháp báo cáo. Ghi ngày chạy hoặc mã commit cạnh bảng để tuần 3 thống nhất phiên bản số liệu.

Mỗi người gửi TV3 một đoạn mô tả đầu vào, lý do chọn biến và cách xử lý thiếu hoặc mã hóa nếu có. Kèm một hệ số hoặc biểu đồ đáng chú ý, một nhận xét dẫn số liệu và một giới hạn quan sát được. Khi diễn giải mô hình nhiều biến, ghi rõ các biến còn lại được giữ cố định. Với biến phân loại, nêu mức tham chiếu.

## Tuần 3 · Báo cáo, video và nộp

| Người | Ngày 15–17 | Ngày 18–20 |
| --- | --- | --- |
| TV1 | Rà phần nguồn dữ liệu, cách chia tập, trích dẫn và thông tin thành viên. Chuẩn bị thư mục nộp đúng cấu trúc. | Tạo ZIP, giải nén sang thư mục mới và kiểm tra đủ tệp. Cùng TV4 chạy lại từ bản giải nén. |
| TV2 | Đối chiếu bảng và biểu đồ với notebook ở phiên bản đã chốt. Kiểm tra đơn vị, tên mô hình và nhận xét so sánh. | Đọc soát PDF lần cuối. So lại các số chính được nhắc trong video với bảng kết quả. |
| TV3 | Hoàn thiện báo cáo, chèn hình ứng dụng, cập nhật kết luận dựa trên kết quả và xuất PDF. Đề gợi ý 10–20 trang. | Mở PDF sau khi xuất để kiểm tra hình, bảng, số trang và danh mục tài liệu tham khảo. Gửi bản chốt cho TV1. |
| TV4 | Tập demo từ đầu, chuẩn bị đầu vào và cách chuyển biểu đồ. Gửi TV3 một ảnh giao diện kèm mô tả luồng dự đoán. | Quay video, xem lại hình và tiếng rồi gửi TV1. Kiểm tra từ gói giải nén rằng cả bốn mô hình hiện trong notebook và ứng dụng dự đoán được. |

**Kịch bản video.** Mở notebook và giới thiệu biến đích. Chọn M1, đổi chiều dài cánh bơi từ 195 mm lên 205 mm và bấm **Dự đoán** để xem kết quả tính bằng gram. Chọn M4, giữ các số đo, đổi một đầu vào phân loại và xem dự đoán hoặc biểu đồ tương ứng. Cuối video, hiện bảng MAE, RMSE và R² của bốn mô hình.

**Bàn giao Ngày 20.** TV1 kiểm tra `Group [Group-ID].zip` theo danh sách bên dưới. **Ngày 21.** Đại diện nộp trước hạn Moodle và báo nhóm khi đã thấy trạng thái nộp thành công.

### Kiểm tra gói nộp

- [ ] `data/` có CSV Palmer Penguins.
- [ ] `source/` có notebook, bốn tệp mô hình và pipeline dùng chung.
- [ ] `requirements.txt` có trong ZIP và hướng dẫn chạy dùng đúng đường dẫn.
- [ ] Video mở được, có thao tác nhập dữ liệu, dự đoán và xem biểu đồ.
- [ ] Báo cáo PDF có mã nhóm, bốn thành viên, mô tả bài toán, dữ liệu, bốn mô hình, đánh giá, kết luận, ứng dụng và tài liệu tham khảo.
- [ ] Notebook trong bản giải nén chạy từ đầu và hiển thị đủ bốn kết quả.
