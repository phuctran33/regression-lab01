# Nghiệm thu tích hợp Lab 01

**Phạm vi của bản tích hợp ngày 28/11/2026.** PASS trong bảng là kết quả kỹ thuật của bản tích hợp. Mỗi thành viên vẫn cần tự đọc, kiểm tra và điều chỉnh phần của mình theo quy định học thuật. Chưa có đánh giá điểm của giảng viên. [Bản thảo khoa học](integration.md) · [PDF dự thảo](integration.pdf)

| Yêu cầu | Bằng chứng | Trạng thái | Việc còn thiếu |
|---|---|---|---|
| Đề Lab 01 §3.1, xác định mục tiêu, động cơ, dữ liệu và đầu vào | [Bản thảo §3.1](integration.md), [EDA tuần 1 và tuần 2](eda-week1.md), [CSV](../data/penguins.csv) | PASS | — |
| §3.2, giải thích lựa chọn và cách xây dựng mô hình | [Năm mô hình và giới hạn](integration.md), [OLS/M3](ols-m3-week1.md), [M1](../reports/ridge-m1-week2.md), [M2](m2_week2/M2_week2_report.md) · [M4](huber-m4-week2.md) | PASS | — |
| §3.3, đánh giá, hệ số, kiểm định và diễn giải từ dữ liệu | [Chẩn đoán OLS](section-3-3-evaluation.md), [đường hệ số Ridge](../reports/m1-week2/coefficient-path.png), [đường tác động spline](m2_week2/tables/C_curvature_summary.csv), [tổng hợp](integration.md) · [Độ nhạy và sai số M4](huber-m4-week2.md) | PASS | — |
| §3.4, thay năm đầu vào, dự đoán và biểu đồ | [Notebook mục 5](../source/lab01.ipynb), [hướng dẫn demo](section-3-4-application.md), [ảnh thật trước](figures/prediction-before.png) và [sau](figures/prediction-dashboard.png) | PASS | — |
| Lời dặn về độ tin cậy, sai số và trực quan | [CI/PI cổ điển](section-3-3-evaluation.md), [split conformal và coverage](integration.md), [chi tiết nhóm](comparison.json) | PASS | Nhóm nhỏ và test đã dùng trước hạn chế suy rộng |
| Tái lập, nguồn và đối chiếu tài liệu | Các lệnh bên dưới, [thiết lập](../requirements.txt), [tài liệu tham khảo](integration.md) | PASS cho code và nguồn đã đối chiếu | Chờ bản toàn văn của Bùi Tiến Lên để đối chiếu trang |
| §4, gói nộp của cả nhóm | CSV, source, môi trường và PDF dự thảo đã có | PENDING | Mã nhóm, PDF cuối 10–20 trang theo gợi ý, video và ZIP |
| Tích hợp | [PR #17](https://github.com/phuctran33/regression-lab01/pull/17) và PR gốc #5–#8, #13–#16 | SHA và trạng thái trên GitHub | — |

### Đối chiếu các checkbox trong issue

| Issue và nhóm tiêu chí | Bằng chứng | Trạng thái kỹ thuật | Việc còn thiếu |
|---|---|---|---|
| #9 A, bốn mục về hội tụ, mở rộng λ, chọn cấu hình và đường tác động | [Lưới 7 giá trị, 35 fold-fit, độ cong](section-3-3-evaluation.md) | PASS | Hưng rà diễn giải |
| #9 B, năm mục về hệ số, giả thuyết, F/Wald chung và ví dụ CI/PI | [Bảng chẩn đoán, ảnh hưởng và dự đoán](section-3-3-evaluation.md) | PASS | Hưng rà giả định |
| #9 C, bốn mục về metrics, phần dư nhóm, bootstrap và ý nghĩa thực tế | [OLS/M3](section-3-3-evaluation.md), [so sánh chung](integration.md) | PASS | — |
| #9 D và sáu sản phẩm bàn giao, chọn mô hình, năm input, biểu đồ, §3.3–3.4, lệnh tái chạy | [App](../source/prediction_app.py), [demo](section-3-4-application.md), [test](../tests/test_conformal.py) | PASS cho năm mô hình | — |
| #11 A và B, cấu hình, 12 + 3 cột, fit trong fold, bốn metrics, phần dư, lỗi nhóm và CI | [Report M2](m2_week2/M2_week2_report.md), [bảng từng fold](m2_week2/tables/A_leakage_check_per_fold.csv) | PASS | Hy rà kết quả |
| #11 C, ba đường có điều kiện, vai trò cơ sở, đối chiếu OLS/M3 và độ cong | [Bảng 9 đường](m2_week2/tables/C_curvature_summary.csv), [log có M3](m2_week2/run_log.txt) | PASS | — |
| #11 D và năm sản phẩm, H0 tuyến tính, tính lồng nhau, F/HC3, giới hạn sau chọn knots, kết luận và lệnh | [Phân tích M2](m2_week2/M2_week2_report.md), [mã chạy](../source/analysis/m2_week2.py) | PASS | — |
| #12 A và B, dùng lại M1, năm alpha, CV train, test, phần dư, sai số nhóm và bootstrap | [Bảng và hình Ridge](../reports/m1-week2), [báo cáo](../reports/ridge-m1-week2.md) | PASS | Khiêm rà diễn giải |
| #12 C, đường hệ số, cùng thang đo, Pearson/VIF, dao động qua fold, cách suy luận | [Bảng hệ số](../reports/m1-week2/coefficients.csv), [VIF](../reports/m1-week2/vif_train.csv), [đường alpha](../reports/m1-week2/coefficient-path.png) | PASS | — |
| #12 D và năm sản phẩm, §3.1, báo cáo Ridge, môi trường sạch, dependencies, source/notebook, leakage và ghi lỗi | [§3.1–3.3](integration.md), [kiểm tra toàn nhóm](../source/check_week2.py), kết quả bên dưới | PASS | — |
| #10 A–E, M4, độ nhạy và EDA thống kê | [M4](huber-m4-week2.md), [EDA thống kê](eda-week1.md), [test Huber](../tests/test_huber.py) | PASS | — |

Nhóm đã dùng nested CV, hai mốc dự đoán hằng và split conformal như phân tích mở rộng. Chế độ benchmark giữ nguyên bộ biến, seed, split và metric. Bỏ qua hiệu chỉnh conformal theo từng nhóm vì chỉ có 55 quan sát calibration. Chế độ trình diễn localhost dùng cùng mô hình và giao diện qua `python -m source.present`.

### Tái lập từ thư mục gốc

Dùng Python 3.12 và môi trường mới, cài `python -m pip install -r requirements.txt`. Chạy các lệnh trong cùng môi trường.

```bash
python -m pip check
python -m compileall -q source tests
python -m unittest discover -s tests -v
python -m source.models.ols
python -m source.models.m1 --select-alpha
python -m source.models.m1
python -m source.models.m2
python -m source.models.m3 --select-lam
python -m source.models.m3
python -m source.models.m4
python -m source.evaluation
python -m source.ridge_evaluation --details --output-dir /tmp/lab01-ridge
python -m source.analysis.m2_week2
python -m source.eda_statistics
python -m source.huber_evaluation
python -m source.analysis.comparison
python -m source.check_week2 --notebook
```

Python 3.12.15, mười dependency ghim trong requirements, gồm anywidget cho giao diện và Voilà cho chế độ trình diễn cục bộ. Kết quả đối chiếu dùng checkout tách biệt và môi trường sạch. Các phép so sánh giữ ID hàng và thứ tự. Tệp JSON lưu toàn bộ ID test, proper train, calibration và năm validation fold ngoài. Kiểm thử theo dõi cả các fit lồng trong GridSearchCV, kiểm tra đúng predictor cho hai cận, định hạng 51/54 và trường hợp phải trả bán kính vô hạn. Cờ `--details` cũng thực thi diagnostics của Ridge. [Runner thống nhất](../source/analysis/comparison.py)

**Kiểm tra bản tích hợp.** `source.check_week2 --notebook` tạo dự đoán hữu hạn, kiểm đối chiếu OLS/M1 và chạy notebook từ kernel mới. Phép kiểm đối chiếu khối lượng và biểu đồ với predictor của từng chế độ và yêu cầu cả năm mô hình. Bảng kết quả chung dùng lại đúng ID train, calibration và test của bản trước. Hai cận của M4 lấy từ 55 sai số hiệu chỉnh sau khi chọn cấu hình trong 218 proper train. [Số liệu chung](comparison.json)

Để xuất lại PDF dự thảo từ Markdown, dùng Pandoc và XeLaTeX trong thư mục `report/` (hai công cụ soạn thảo nằm ngoài môi trường chạy notebook).

```bash
pandoc integration.md -o integration.pdf --pdf-engine=xelatex -V mainfont='Times New Roman' -V geometry:margin=23mm -V fontsize=10pt -V papersize=a4 -V colorlinks=true
```

### Kiểm thử giao diện và giới hạn

Hồ sơ thiết kế ở lần bàn giao bốn mô hình đã ghi Chromium và Firefox trong JupyterLab, WebKit cho bản trình diễn và luồng bàn phím với Orca trên Linux. Nghiệm thu bao gồm bốn vùng dùng chung đầu vào, so sánh benchmark/conformal, 69 hàng test và quy trình hiệu chỉnh. Báo cáo giao diện ghi ảnh trước/sau, kết quả axe theo trạng thái, thao tác bàn phím, kích thước đã thử và các giới hạn. Bản trình diễn dùng chung mô hình và widget, xuất CSV của đúng predictor. [Nghiệm thu giao diện](ui-audit.md) · [Kết quả QA](ui-checks.json)

### Gói nộp của nhóm

Mỗi thành viên đọc và xác nhận phần mình phụ trách. Điền mã nhóm, đối chiếu thông tin thành viên và nguồn, hoàn thiện PDF theo đề, quay video demo, đóng ZIP và chạy lại từ bản giải nén. [Hướng dẫn demo](section-3-4-application.md#demo-hai-phút)
