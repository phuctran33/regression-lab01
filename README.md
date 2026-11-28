# Lab 01 – Dự đoán khối lượng chim cánh cụt

Đồ án hồi quy tuyến tính với bộ dữ liệu Palmer Penguins. Biến đích là `body_mass_g` (g). Bản tích hợp có OLS, M1 Ridge, M2 spline cơ sở và M3 spline làm mượt. EDA tuần 2 và M4 còn chờ Phúc. Notebook dùng chung bộ đầu vào và tập kiểm tra để so sánh.

| Tệp | Nội dung |
| --- | --- |
| [source/lab01.ipynb](source/lab01.ipynb) | Khám phá dữ liệu, huấn luyện, so sánh mô hình và giao diện dự đoán |
| [data/penguins.csv](data/penguins.csv) | Bản CSV của bộ `penguins` đã đơn giản hóa, dùng được ngoại tuyến |
| [report/report-template.md](report/report-template.md) | Khung báo cáo và câu hỏi gợi ý cho từng phần |
| [requirements.txt](requirements.txt) | Thư viện Python cần cài |

## Kết quả tích hợp

[Báo cáo khoa học](report/integration.md) · [Bản PDF dự thảo](report/integration.pdf) · [Nghiệm thu và tái lập](report/handoff.md) · [Số liệu theo fold, ID và nhóm](report/comparison.json)

Từ môi trường đã cài requirements, chạy `python -m source.analysis.comparison` để tái tạo benchmark, nested CV và split conformal. Ứng dụng mặc định dùng predictor fit 218 mẫu cùng 55 mẫu hiệu chỉnh. Chế độ benchmark dùng 273 mẫu. Mỗi chế độ có bộ dự đoán riêng.

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

Trong JupyterLab, chọn kernel Python của môi trường vừa tạo và chạy **Kernel → Restart Kernel and Run All Cells**. Trong VS Code, mở notebook và chọn `.venv\Scripts\python.exe` làm kernel. Giao diện ở mục 5 cho phép dự đoán bằng các mô hình sẵn sàng, chỉnh đầu vào và xem xu hướng có điều kiện. Ô nhập số đi kèm từng thanh trượt. Dải kết quả hiển thị dự đoán và hai cận trên một thước gram. Chọn **Xu hướng**, **Phần dư** hoặc **Sai số theo loài** để đổi hình. Mỗi hình có bảng số liệu. Chế độ so sánh hiển thị các mô hình trên cùng đầu vào. Ở chế độ benchmark, chọn OLS để xem CI cho trung bình và PI cho một cá thể. [Kịch bản demo Section 3.4](report/section-3-4-application.md) · [Ảnh và kiểm thử giao diện](report/ui-audit.md)

## Phân công mô hình

| Thành viên | Tệp phụ trách | Thuật toán |
| --- | --- | --- |
| TV1 · Tiết Trọng Khiêm | [M1](source/models/m1.py) | Ridge Regression |
| TV2 · Nguyễn Khang Hy | [M2](source/models/m2.py) | Cubic Spline Basis Regression |
| TV3 · Nguyễn Đức Hưng | [M3](source/models/m3.py) | Additive Natural Cubic Smoothing Spline |
| TV4 · Trần Thiên Phúc | [M4](source/models/m4.py) | Huber Regression (dự kiến) |

TV4 phụ trách EDA từ 19/11 và bàn giao kết quả để cả nhóm chốt bộ biến đầu vào ngày 20/11.

Notebook hiện hệ số nếu estimator đã huấn luyện cung cấp `coef_` và `intercept_`. Khi `ESTIMATOR = None`, lệnh chạy riêng tệp vẫn dùng cấu hình mẫu để thử pipeline. Notebook hiển thị **Chưa sẵn sàng** và chỉ đưa các mô hình đã khai báo vào bảng đánh giá và giao diện dự đoán. [Tài liệu hồi quy tuyến tính của scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html)

Bộ biến đã chọn gồm chiều dài cánh bơi, chiều dài và độ sâu mỏ, loài và giới tính. Chạy mục 2 của notebook để xem cơ sở lựa chọn bộ biến chung trong `FEATURES`. Sau đó, mở terminal tại thư mục dự án và dùng lệnh dưới đây để chạy riêng M2. Đổi `m2` thành tệp mình phụ trách.

```powershell
.\.venv\Scripts\python.exe -m source.models.m2
```

Khởi động lại kernel và chạy toàn bộ notebook để nạp lựa chọn mới. Mục 4 có bảng so sánh và danh sách chọn mô hình để xem hệ số, phần dư. Chạy `python -m source.evaluation` để tái lập bảng chẩn đoán OLS, đánh giá M3 và bootstrap theo cặp. [Phân tích Section 3.3](report/section-3-3-evaluation.md) · [Kế hoạch 19/11–08/12/2026](KE-HOACH-3-TUAN.md)

Nếu dùng Git, mỗi người làm trên nhánh riêng và đưa tệp mô hình mình phụ trách vào commit. Ví dụ, người làm M2 có thể dùng `git add source/models/m2.py`. Phối hợp trước khi sửa notebook hoặc [pipeline chung](source/modeling.py). Khi gửi thay đổi cho một mô hình, giữ output notebook ở máy cá nhân để tránh xung đột.

## Hoàn thiện bài nhóm

1. Xác nhận mã nhóm và thông tin thành viên cho bản nộp.
2. Ghép EDA tuần 2 và M4 của Phúc, đối chiếu với kết quả bốn predictor trong báo cáo tích hợp.
3. Chạy lại toàn bộ notebook, đưa bảng chỉ số và biểu đồ vào báo cáo. Liên hệ nhận xét với hệ số, sai số và dữ liệu quan sát được.
4. Quay video demo giao diện nhập dữ liệu, dự đoán và xem biểu đồ. Xuất báo cáo hoàn chỉnh sang PDF (đề gợi ý 10–20 trang).
5. Cử một đại diện nộp `Group [Group-ID].zip`, thay `[Group-ID]` bằng mã nhóm. Gói nộp gồm thư mục `data/`, thư mục `source/`, `requirements.txt`, video demo và báo cáo PDF. Kiểm tra notebook chạy được sau khi giải nén.

## Nguồn dữ liệu

Horst, A. M., Hill, A. P., & Gorman, K. B. (2020). *palmerpenguins: Palmer Archipelago (Antarctica) penguin data*. Dataset phát hành theo CC0. [Trang dự án](https://allisonhorst.github.io/palmerpenguins/) · [CSV gốc](https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv) · [DOI 10.5281/zenodo.3960218](https://doi.org/10.5281/zenodo.3960218)
