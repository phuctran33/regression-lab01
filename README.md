# Lab 01 – Dự đoán khối lượng chim cánh cụt

Đồ án hồi quy tuyến tính với bộ dữ liệu Palmer Penguins. Biến đích là `body_mass_g` (g). Bản tích hợp có OLS, M1 Ridge, M2 spline cơ sở, M3 spline làm mượt và M4 Huber. EDA thống kê phân tích tương quan, khác biệt giữa nhóm và hình dạng quan hệ trên train. Notebook dùng chung bộ đầu vào và tập kiểm tra để so sánh.

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

Trong JupyterLab, chọn kernel Python của môi trường vừa tạo và chạy **Kernel → Restart Kernel and Run All Cells**. Trong VS Code, mở notebook và chọn `.venv\Scripts\python.exe` làm kernel. Ở mục 5, chọn **Cá thể mẫu** hoặc nhập ba số đo, loài và giới tính, rồi bấm **Dự đoán**. Bốn khu vực dùng chung đầu vào và mô hình.

| Khu vực | Có thể đọc gì? |
| --- | --- |
| Dự đoán cá thể | Khối lượng và khoảng dự đoán, xu hướng có điều kiện, phần dư, sai số theo loài và thực tế so với dự đoán. |
| So sánh mô hình | Năm khoảng conformal trên cùng thước gram. Đổi sang Benchmark để xem ΔMAE, CI bootstrap, chỉ số test và CV. |
| Kiểm chứng khoảng | Cận, dự đoán và khối lượng thật của 69 cá thể. So sánh độ bao phủ và độ rộng ở hai mức 90% và 95%. |
| Phương pháp | Bảy bước tách tập, fit, hiệu chỉnh và kiểm chứng. Mở bảng để đọc 55 sai số đã xếp hạng. |

**Khôi phục mặc định** đưa toàn bộ ứng dụng về ví dụ OLS ban đầu. Các hình có mô tả và bảng đọc được bằng bàn phím. Giữ `report/comparison.json` trong bản bàn giao để hiện CV và bootstrap đã đối chiếu. [Demo hai phút](report/section-3-4-application.md#demo-hai-phút) · [Ảnh và nghiệm thu](report/ui-audit.md)

## Trình diễn độc lập

Từ thư mục dự án, trong môi trường đã cài [requirements.txt](requirements.txt), chạy một lệnh.

```bash
python -m source.present
```

Mở địa chỉ localhost mà Voilà in ra (mặc định `http://127.0.0.1:8866`). Chờ kết quả OLS đầu tiên rồi chọn **Toàn màn hình** nếu muốn chiếu. Mỗi phiên dùng cùng dữ liệu, mô hình và widget như mục 5 của notebook. Sau khi cài môi trường, ứng dụng chạy ngoại tuyến. Dùng **Ctrl+C** trong terminal để dừng.

Chọn một đường đo trên sơ đồ để sửa số đo tương ứng. Trên biểu đồ, trỏ, chạm hoặc chọn **Điểm dữ liệu** để đọc quan sát. **Mở rộng biểu đồ** giữ điểm đang xem. **Tải CSV** lưu một dòng gồm đầu vào, mô hình, dự đoán, khoảng và kết quả kiểm chứng hiện hành. [Hướng dẫn đọc và xuất kết quả](report/section-3-4-application.md)

## Phân công mô hình

| Thành viên | Tệp phụ trách | Thuật toán |
| --- | --- | --- |
| TV1 · Tiết Trọng Khiêm | [M1](source/models/m1.py) | Ridge Regression |
| TV2 · Nguyễn Khang Hy | [M2](source/models/m2.py) | Cubic Spline Basis Regression |
| TV3 · Nguyễn Đức Hưng | [M3](source/models/m3.py) | Additive Natural Cubic Smoothing Spline |
| TV4 · Trần Thiên Phúc | [M4](source/models/m4.py) | Huber Regression |

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
2. Đối chiếu kết quả của năm mô hình và duyệt [EDA thống kê](report/eda-week1.md) cùng [phân tích M4](report/huber-m4-week2.md).
3. Chạy lại toàn bộ notebook, đưa bảng chỉ số và biểu đồ vào báo cáo. Liên hệ nhận xét với hệ số, sai số và dữ liệu quan sát được.
4. Quay video demo giao diện nhập dữ liệu, dự đoán và xem biểu đồ. Xuất báo cáo hoàn chỉnh sang PDF (đề gợi ý 10–20 trang).
5. Cử một đại diện nộp `Group [Group-ID].zip`, thay `[Group-ID]` bằng mã nhóm. Gói nộp gồm thư mục `data/`, thư mục `source/`, `requirements.txt`, video demo và báo cáo PDF. Kiểm tra notebook chạy được sau khi giải nén.

## Nguồn dữ liệu

Horst, A. M., Hill, A. P., & Gorman, K. B. (2020). *palmerpenguins: Palmer Archipelago (Antarctica) penguin data*. Dataset phát hành theo CC0. [Trang dự án](https://allisonhorst.github.io/palmerpenguins/) · [CSV gốc](https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv) · [DOI 10.5281/zenodo.3960218](https://doi.org/10.5281/zenodo.3960218)
