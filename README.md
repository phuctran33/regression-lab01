# Lab 01 – Dự đoán khối lượng chim cánh cụt

Template hồi quy tuyến tính với bộ dữ liệu Palmer Penguins cho nhóm 4 người. Biến đích là `body_mass_g` (g). Mỗi người tự chọn một thuật toán hồi quy tuyến tính và triển khai trong tệp mình phụ trách. Notebook dùng chung bộ đầu vào và tập kiểm tra để so sánh.

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

| Thành viên | Tệp phụ trách | Thuật toán |
| --- | --- | --- |
| TV1 · Tiết Trọng Khiêm | [M1](source/models/m1.py) | Ridge Regression |
| TV2 · Nguyễn Khang Hy | [M2](source/models/m2.py) | Chưa chọn |
| TV3 · Nguyễn Đức Hưng | [M3](source/models/m3.py) | Smoothing Linear Regression |
| TV4 · Trần Thiên Phúc | [M4](source/models/m4.py) | Huber Regression |

TV4 phụ trách EDA tuần 1 và bàn giao kết quả cho cả nhóm để chốt bộ biến đầu vào. [Kế hoạch tuần 1](KE-HOACH-3-TUAN.md#tuần-1--eda-chốt-đầu-vào-và-làm-bốn-mô-hình)

Mỗi người báo lựa chọn cho nhóm để bốn thuật toán khác nhau. Import estimator đã chọn, khởi tạo rồi gán vào `ESTIMATOR` trong tệp của mình. Notebook hiện hệ số nếu estimator đã huấn luyện cung cấp `coef_` và `intercept_`. Khi `ESTIMATOR = None`, tệp chạy cấu hình mẫu để thử pipeline và notebook hiển thị trạng thái **Chưa chọn (mẫu)**. [Tài liệu hồi quy tuyến tính của scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html)

Bộ biến mẫu gồm chiều dài cánh bơi, chiều dài và độ sâu mỏ, loài và giới tính. Chạy mục 2 của notebook để bắt đầu khám phá dữ liệu train rồi chốt bộ biến chung trong `FEATURES`. Sau đó, mở terminal tại thư mục dự án và dùng lệnh dưới đây để chạy riêng M2. Đổi `m2` thành tệp mình phụ trách.

```powershell
.\.venv\Scripts\python.exe -m source.models.m2
```

Khởi động lại kernel và chạy toàn bộ notebook để nạp lựa chọn mới. Mục 4 có bảng so sánh và danh sách chọn mô hình để xem hệ số, phần dư. [Kế hoạch 3 tuần](KE-HOACH-3-TUAN.md)

Nếu dùng Git, mỗi người làm trên nhánh riêng và đưa tệp mô hình mình phụ trách vào commit. Ví dụ, người làm M2 có thể dùng `git add source/models/m2.py`. Phối hợp trước khi sửa notebook hoặc [pipeline chung](source/modeling.py). Khi gửi thay đổi cho một mô hình, giữ output notebook ở máy cá nhân để tránh xung đột.

## Hoàn thiện bài nhóm

1. Điền mục tiêu, động cơ, mã nhóm, thành viên và phân công.
2. Chốt bộ biến và cách tiền xử lý dựa trên EDA. Giải thích lý do chọn thuật toán và siêu tham số của từng mô hình.
3. Chạy lại toàn bộ notebook, đưa bảng chỉ số và biểu đồ vào báo cáo. Liên hệ nhận xét với hệ số, sai số và dữ liệu quan sát được.
4. Quay video demo giao diện nhập dữ liệu, dự đoán và xem biểu đồ. Xuất báo cáo hoàn chỉnh sang PDF (đề gợi ý 10–20 trang).
5. Cử một đại diện nộp `Group [Group-ID].zip`, thay `[Group-ID]` bằng mã nhóm. Gói nộp gồm thư mục `data/`, thư mục `source/`, `requirements.txt`, video demo và báo cáo PDF. Kiểm tra notebook chạy được sau khi giải nén.

## Nguồn dữ liệu

Horst, A. M., Hill, A. P., & Gorman, K. B. (2020). *palmerpenguins: Palmer Archipelago (Antarctica) penguin data*. Dataset phát hành theo CC0. [Trang dự án](https://allisonhorst.github.io/palmerpenguins/) · [CSV gốc](https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv) · [DOI 10.5281/zenodo.3960218](https://doi.org/10.5281/zenodo.3960218)
