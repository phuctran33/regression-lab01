# Lab 01 – Dự đoán khối lượng chim cánh cụt

Template hồi quy tuyến tính với bộ dữ liệu Palmer Penguins cho nhóm 4 người. Biến đích là `body_mass_g` (g). Notebook có 4 cấu hình mẫu, bắt đầu từ chiều dài cánh bơi và lần lượt bổ sung số đo mỏ, loài và giới tính. Nhóm điền các mục TODO và viết nhận xét dựa trên kết quả thực nghiệm.

| Tệp | Nội dung |
| --- | --- |
| [source/lab01.ipynb](source/lab01.ipynb) | Khám phá dữ liệu, huấn luyện, so sánh mô hình và giao diện dự đoán |
| [data/penguins.csv](data/penguins.csv) | Bản CSV của bộ `penguins` đã đơn giản hóa, dùng được ngoại tuyến |
| [report/report-template.md](report/report-template.md) | Khung báo cáo và câu hỏi gợi ý cho từng phần |
| [requirements.txt](requirements.txt) | Thư viện Python cần cài |

## Chạy notebook

Dùng Python 3.11 hoặc 3.12. Mở PowerShell tại thư mục dự án và chạy các lệnh dưới đây. Nếu lệnh `python -m pip --version` báo thiếu pip, đặt `$labPython` thành đường dẫn tới bản Python có pip rồi chạy tiếp từ dòng tạo môi trường.

```powershell
$labPython = "python"
# Ví dụ đường dẫn cho Python cài theo tài khoản trên Windows
# $labPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
& $labPython -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab source/lab01.ipynb
```

Trong JupyterLab, chọn kernel Python của môi trường vừa tạo và chạy **Kernel → Restart Kernel and Run All Cells**. Trong VS Code, mở notebook và chọn `.venv\Scripts\python.exe` làm kernel. Giao diện dự đoán nằm ở mục 5 của notebook.

## Phân công mô hình

| Thành viên | Tệp phụ trách | Biến đầu vào ban đầu |
| --- | --- | --- |
| 1 | [M1](source/models/m1.py) | Chiều dài cánh bơi |
| 2 | [M2](source/models/m2.py) | Chiều dài cánh bơi, chiều dài và độ sâu mỏ |
| 3 | [M3](source/models/m3.py) | Số đo của M2 và loài |
| 4 | [M4](source/models/m4.py) | Đầu vào của M3 và giới tính |

Mỗi người chỉnh `FEATURES` hoặc `build_model()` trong tệp của mình. Khởi động lại kernel và chạy toàn bộ notebook để nạp lần sửa mới. Mục 4 có bảng so sánh chung và danh sách chọn mô hình để xem hệ số, phần dư.

Nếu dùng Git, mỗi người làm trên nhánh riêng và đưa tệp mô hình mình phụ trách vào commit. Ví dụ, người làm M2 có thể dùng `git add source/models/m2.py`. Phối hợp trước khi sửa notebook hoặc [pipeline chung](source/modeling.py). Khi gửi thay đổi cho một mô hình, giữ output notebook ở máy cá nhân để tránh xung đột.

## Hoàn thiện bài nhóm

1. Điền mục tiêu, động cơ, mã nhóm, thành viên và phân công.
2. Giữ ít nhất 4 mô hình hồi quy tuyến tính. Giải thích cách tiền xử lý và lý do chọn biến cho từng mô hình.
3. Chạy lại toàn bộ notebook, đưa bảng chỉ số và biểu đồ vào báo cáo. Liên hệ nhận xét với hệ số, sai số và dữ liệu quan sát được.
4. Quay video demo giao diện nhập dữ liệu, dự đoán và xem biểu đồ. Xuất báo cáo hoàn chỉnh sang PDF (đề gợi ý 10–20 trang).
5. Cử một đại diện nộp `Group [Group-ID].zip`, thay `[Group-ID]` bằng mã nhóm. Gói nộp gồm thư mục `data/`, thư mục `source/`, `requirements.txt`, video demo và báo cáo PDF. Kiểm tra notebook chạy được sau khi giải nén.

## Nguồn dữ liệu

Horst, A. M., Hill, A. P., & Gorman, K. B. (2020). *palmerpenguins: Palmer Archipelago (Antarctica) penguin data*. Dataset phát hành theo CC0. [Trang dự án](https://allisonhorst.github.io/palmerpenguins/) · [CSV gốc](https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv) · [DOI 10.5281/zenodo.3960218](https://doi.org/10.5281/zenodo.3960218)
