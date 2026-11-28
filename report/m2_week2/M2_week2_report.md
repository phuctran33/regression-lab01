# M2 – Cubic Spline Basis Regression · Nội dung báo cáo tuần 2

**Phụ trách:** Nguyễn Khang Hy (TV2) · **Issue:** #11 · **Số liệu:** do `source/analysis/m2_week2.py` sinh ra (xem `run_log.txt`, `tables/`, `results.json`).

Quy ước chung: 273 mẫu train, 69 mẫu test, seed 42; CV 5-fold `shuffle=True`, `random_state=42`; test set chỉ dùng để báo cáo kết quả, không dùng để lựa chọn siêu tham số.

## 3.2 Đánh giá thống kê M2 và so sánh với OLS

### 3.2.1 Kiểm tra cấu hình và rò rỉ dữ liệu

- Cấu hình giữ nguyên tuần 1: `degree=3`, `n_knots=3`, `knots="uniform"`, `include_bias=False`, `extrapolation="constant"`, estimator cuối `LinearRegression`.
- Ma trận đặc trưng sau biến đổi có kích thước `(273, 15)`: gồm 12 spline basis (4 hàm cơ sở cho mỗi trong 3 biến số) và 3 biến giả (`species_Chinstrap`, `species_Gentoo`, `sex_male`).
- Chạy lại cấu hình trên train/test chung cho kết quả trùng tuần 1 (MAE 252.934 g, RMSE 308.237 g, R² 0.832).
- Trong từng fold, median của imputer, mốc biên (min/max) của `SplineTransformer` và mức phổ biến nhất của biến phân loại đều khớp với dữ liệu fit của fold đó (`tables/A_leakage_check_per_fold.csv`). `cross_validate` clone cả `Pipeline`, nên bước tiền xử lý được fit lại trong mỗi fold, không học từ fold kiểm tra hay từ test.
- Không có quan sát test nào nằm ngoài miền giá trị của train. Mỗi fold CV có 1–2 giá trị kiểm tra nằm ngoài miền fold-train, nên extrapolation hằng số ảnh hưởng rất ít.

### 3.2.2 Chỉ số trên tập kiểm tra (n = 69)

| Mô hình | MAE (g) | MSE (g²) | RMSE (g) | R² |
| --- | ---: | ---: | ---: | ---: |
| M2 – Spline | 252.934 | 95 009.7 | 308.237 | 0.832 |
| OLS (cùng pipeline, cùng split) | 256.683 | 94 252.1 | 307.005 | 0.833 |

M2 có MAE thấp hơn OLS 3.7 g nhưng RMSE và MSE cao hơn nhẹ; R² gần như bằng nhau. Vì vậy, kết quả trên tập test chưa cho thấy M2 vượt trội OLS một cách nhất quán.

### 3.2.3 Cross-validation 5-fold trên train

Giá trị là trung bình ± độ lệch chuẩn giữa 5 fold (ddof = 1). RMSE là trung bình của RMSE từng fold.

| Mô hình | MAE (g) | MSE (g²) | RMSE (g) | R² |
| --- | ---: | ---: | ---: | ---: |
| M2 – Spline | 234.4 ± 24.9 | 88 278 ± 17 200 | 295.9 ± 30.3 | 0.859 ± 0.037 |
| OLS | 225.6 ± 17.2 | 84 043 ± 11 683 | 289.3 ± 21.1 | 0.866 ± 0.023 |

M2 có MAE thấp hơn OLS ở 2/5 fold (fold 3 và 4); ở 3 fold còn lại M2 kém hơn, fold 2 kém nhất (+26.8 g). Điểm đáng chú ý là tuần 1 chọn `n_knots = 3` bằng CV, và chính CV này cho thấy mô hình tuyến tính không kém hơn spline.

Vì `n_knots` được chọn trên cùng các fold, ước lượng CV của M2 có thể hơi lạc quan. Nested CV (vòng trong chọn `n_knots` ∈ {3, 4, 5, 6}) cho MAE = **239.0 ± 28.3 g**; `n_knots = 3` được chọn ở 4/5 fold (fold 5 chọn 6). Con số này còn cao hơn OLS (225.6 g).

### 3.2.4 Phần dư và sai số theo nhóm

- `figures/fig01_actual_vs_predicted.png`: điểm dự đoán của M2 và OLS bám đường chéo tương tự nhau.
- `figures/fig02_residuals.png`: phần dư không có xu hướng rõ theo giá trị dự đoán. Phần dư out-of-fold của M2 (n = 273) không cho thấy bằng chứng đủ mạnh để bác bỏ giả thuyết phân phối chuẩn (Shapiro p = 0.345). Kiểm định Levene cũng chưa phát hiện khác biệt phương sai có ý nghĩa thống kê giữa ba loài (p = 0.875).

Sai số M2 trên test (bias = trung bình của thực tế − dự đoán). Nhóm có n nhỏ chỉ nên đọc để tham khảo.

| Nhóm | n | MAE (g) | RMSE (g) | bias (g) |
| --- | ---: | ---: | ---: | ---: |
| Adelie | 35 | 263.0 | 317.0 | −24.8 |
| Chinstrap | 11 | 253.1 | 293.0 | +31.0 |
| Gentoo | 23 | 237.5 | 301.8 | −7.3 |
| Cái | 29 | 214.1 | 256.2 | +40.9 |
| Đực | 36 | 279.4 | 333.2 | −19.3 |
| Thiếu `sex` | 4 | 296.7 | 404.5 | −296.7 |

Nhóm đực có sai số lớn hơn nhóm cái ở cả test (279 so với 214 g) lẫn out-of-fold trên train (261 so với 207 g, n = 132 và 136), nên mẫu hình này cũng xuất hiện trên tập lớn hơn. Bốn mẫu test thiếu `sex` được điền bằng mức phổ biến nhất của train (`female`) và có sai số lớn nhất; với n = 4 chỉ nên coi đây là quan sát mô tả. Bảng đầy đủ theo loài × giới tính (kể cả số mẫu từng ô) nằm ở `tables/B_group_errors_test.csv` và `tables/B_group_errors_oof_train.csv`. Một số ô rất nhỏ, ví dụ Chinstrap cái chỉ có 3 mẫu test.

### 3.2.5 Paired bootstrap M2 so với OLS

Resample cùng chỉ số test cho cả hai mô hình (2.000 lần, `default_rng(42)`), không refit. ΔMAE = MAE(M2) − MAE(OLS); âm nghĩa là M2 tốt hơn.

| Chỉ số | Quan sát | CI 95% percentile | CI chứa 0 |
| --- | ---: | ---: | --- |
| **ΔMAE (g)** | **−3.75** | **[−16.51; +8.41]** | **Có** |
| ΔRMSE (g), phụ | +1.23 | [−10.85; +13.23] | Có |
| ΔR², phụ | −0.0013 | [−0.0154; +0.0124] | Có |

Có 73.5% lần bootstrap cho ΔMAE < 0; đây chỉ là tần suất, không phải p-value. M2 có sai số tuyệt đối nhỏ hơn OLS ở 35/69 mẫu. Hai dòng phụ thuộc loại khám phá, không dùng để kết luận.

Ý nghĩa thống kê: khoảng tin cậy chứa 0 nên chưa đủ bằng chứng về khác biệt MAE giữa M2 và OLS trên tập test này. Ý nghĩa thực tế: chênh lệch quan sát 3.7 g bằng khoảng 1.5% MAE (≈ 253 g), và cận trên của CI (+8.4 g) cho thấy M2 cũng có thể kém hơn một chút. Xem `figures/fig03_paired_bootstrap_delta_mae.png`. Giới hạn: n = 69 nên CI rộng, và bootstrap không refit nên chỉ phản ánh biến thiên của tập test, không phản ánh biến thiên khi huấn luyện lại.

## 3.3 Diễn giải đường cong spline và nhu cầu phi tuyến

### 3.3.1 Cách spline tạo đường cong (ISLP, Chương 7, §7.3–7.4)

Với mỗi biến số, `SplineTransformer` thay một cột x bằng 4 hàm cơ sở B-spline bậc 3 trải trên các mốc uniform (3 mốc). Hồi quy tuyến tính ước lượng một hệ số cho mỗi hàm cơ sở, nên đường dự đoán là tổ hợp tuyến tính của các hàm cơ sở: từng đoạn là đa thức bậc 3, nối trơn tại mốc. Vì mô hình cộng các thành phần của từng biến (additive, tương tự GAM ở ISLP §7.7), hình dạng đường cong của mỗi biến không phụ thuộc vào giá trị cố định của các biến còn lại; đổi giá trị tham chiếu chỉ dịch cả đường lên hoặc xuống.

### 3.3.2 Ba biểu đồ conditional prediction

Giá trị tham chiếu lấy từ train: cánh bơi 197 mm, mỏ dài 44.5 mm, mỏ sâu 17.3 mm, loài Adelie, giới tính cái. Mỗi biểu đồ thay đổi một biến, vẽ đường dự đoán M2 và OLS, kèm dải CI 95% bootstrap (500 lần refit) cho M2 và vùng ngoài miền train (xám): `figures/fig04_conditional_flipper_length_mm.png`, `figures/fig04_conditional_bill_length_mm.png`, `figures/fig04_conditional_bill_depth_mm.png`. Bản bám miền dữ liệu từng loài là `figures/fig05_conditional_by_species.png`. Đây là dự đoán có điều kiện, không phải quan hệ nhân quả; tổ hợp tham chiếu (Adelie với cánh bơi 230 mm chẳng hạn) có thể không tồn tại trong dữ liệu, nên hình 05 đáng tin hơn khi đọc theo từng loài.

Độ dốc trung bình trong khoảng p5–p95 của train và biên độ lệch khỏi đường thẳng bình phương tối thiểu của chính đường cong (`tables/C_curvature_summary.csv`):

| Biến | Độ dốc OLS (g/mm) | Độ dốc M2 đoạn thấp → giữa → cao | Biên độ cong M2 (g) |
| --- | ---: | --- | ---: |
| Cánh bơi (181–225 mm) | 13.0 | 7.1 → 19.9 → 20.6 | 94.7 |
| Mỏ dài (35.6–52.0 mm) | 16.2 | 29.1 → 23.3 → −6.1 | 82.0 |
| Mỏ sâu (13.9–20.0 mm) | 75.6 | 139.3 → 94.3 → 50.9 | 65.6 |

- **Cánh bơi:** M2 phẳng hơn ở khoảng 181–195 mm rồi dốc hơn từ khoảng 205 mm trở lên (dạng chữ S nhẹ). Phía trên 225 mm đường phẳng ra, nơi dữ liệu thưa.
- **Mỏ dài:** dốc ở đầu thấp, rồi hạ độ dốc và gần như phẳng hoặc giảm ở đoạn trên 50 mm. Đoạn này chỉ có ít quan sát (Chinstrap và vài Gentoo), nên khó tin đây là một quan hệ thật.
- **Mỏ sâu:** độ dốc giảm dần (đường lõm), đồng thời độ dốc ở đầu thấp (139 g/mm) lớn gần gấp đôi OLS.
- **Cong có đáng kể không?** Biên độ cong của M2 là 66–95 g, tức khoảng 26–37% MAE test của M2 (253 g). Về mặt thống kê, đường OLS nằm trọn trong dải CI 95% của M2 tại 100% điểm trong khoảng p5–p95 ở cả ba biến, và dải này rộng (nửa bề rộng trung bình 181–240 g, `tables/C_band_coverage.csv`). Residual bootstrap dưới H0 tuyến tính cho biên độ quan sát nằm trong vùng bình thường của phân phối null: cánh bơi p = 0.145, mỏ dài p = 0.104, mỏ sâu p = 0.350 (`figures/fig06_nonlinearity_null.png`). Độ cong có thể thấy bằng mắt, nhưng chưa phân biệt được với dao động do mẫu.
- **Overfitting và extrapolation:** dải bootstrap loe rộng ở hai đầu, đoạn mỏ dài trên 50 mm đảo chiều độ dốc, độ lệch chuẩn giữa các fold của M2 lớn hơn OLS, và CV của M2 tệ hơn OLS; tất cả cùng hướng về một mô hình hơi quá linh hoạt so với 273 mẫu. Ngoài miền train, `extrapolation="constant"` giữ đường phẳng (vùng xám ở hình 04), trong khi OLS tiếp tục tăng tuyến tính; hai cách này khác nhau ở ngoài miền nhưng không ảnh hưởng đến test hiện tại vì không có mẫu test ngoài miền.

### 3.3.3 Kiểm tra nhu cầu thành phần phi tuyến

H0: không cần thành phần phi tuyến so với mô hình tuyến tính (cùng các biến giả).

Có thể dùng F-test có điều kiện vì hai mô hình lồng nhau: cột của OLS biểu diễn chính xác được bằng cơ sở spline (sai số số học 4.6e-13), M2 có 16 tham số so với 7 của OLS (thêm 9), và mốc uniform chỉ phụ thuộc vào X chứ không phụ thuộc y. Các p-value dưới đây vẫn chỉ mang tính mô tả: `n_knots = 3` được chọn bằng CV trên cùng dữ liệu, các mốc phụ thuộc vào miền train, và phép kiểm định dùng 273 mẫu train (không dùng test).

| Phương pháp | Thống kê | p |
| --- | ---: | ---: |
| F-test lồng nhau, chung ba biến (9 df) | F(9, 257) = 1.502 | 0.147 |
| Wald–F bền vững HC3 (9 df) | F = 1.596 | 0.117 |
| F từng biến (3 df): cánh bơi / mỏ dài / mỏ sâu | 1.177 / 1.455 / 0.900 | 0.319 / 0.227 / 0.442 (Holm: 0.682) |

AIC trong mẫu: OLS 3093.6 so với M2 3097.6; BIC: OLS 3118.9 so với M2 3155.4 (thấp hơn là tốt hơn, cả hai thiên về OLS). Phần dư train của M2 chưa cho thấy bằng chứng đủ mạnh để bác bỏ giả thuyết phân phối chuẩn (Shapiro p = 0.259).

### 3.3.4 Kết luận về tính phi tuyến

Kết quả từ các góc nhìn đều cùng hướng. **Chưa đủ bằng chứng để bác bỏ H0**: F-test (p = 0.147), kiểm định bền vững (p = 0.117), kiểm định từng biến, residual bootstrap, paired bootstrap (CI ΔMAE chứa 0), CV và nested CV (OLS tốt hơn hoặc bằng M2) và AIC/BIC (thiên về OLS) không cho thấy lợi ích rõ ràng của spline. Chưa bác bỏ H0 không có nghĩa là quan hệ chắc chắn tuyến tính: với n = 273 và 9 tham số thêm, phép kiểm định có thể thiếu công suất để phát hiện độ cong nhỏ, và các đường cong có hình dạng nhất quán với một ít độ cong ở cánh bơi và độ sâu mỏ. Về mặt thực tế, M2 không cải thiện MAE hay RMSE trên test một cách đáng tin cậy và phải trả giá bằng 9 tham số thêm, ít dễ diễn giải hơn; do đó, với bộ dữ liệu này, tăng độ phức tạp từ OLS lên spline chưa cho thấy lợi ích rõ ràng. Kết luận này chỉ áp dụng cho M2 so với OLS; so sánh với M1 và M4 thuộc phần so sánh chung của nhóm.

## Giới hạn

- Test chỉ có 69 mẫu; nhiều nhóm loài × giới tính chỉ có vài mẫu.
- Median/mode imputation được dùng cho các giá trị thiếu (2 dòng thiếu số đo trong dữ liệu gốc, 11 dòng thiếu giới tính); các dòng thiếu `sex` có sai số lớn hơn.
- Các p-value ở 3.3.3 là mô tả vì `n_knots` được chọn bằng CV; residual bootstrap giả định phần dư hoán đổi được (Levene chưa phát hiện khác biệt phương sai theo loài).
- Kết quả chính của M2 đã được chạy lại trên môi trường local với scikit-learn 1.8.0 theo phiên bản ghim trong `requirements.txt`. Trên nhánh main tại thời điểm thực hiện phân tích, source/models/m3.py chưa có estimator (ESTIMATOR = None), nên báo cáo không vẽ hoặc sử dụng đường M3. Phần so sánh với M3 được tạm để lại cho phần đánh giá chung của nhóm sau khi mã M3 được merge và có thể chạy lại.

## Tài liệu tham khảo

1. James, Witten, Hastie, Tibshirani, Taylor (2023). *An Introduction to Statistical Learning with Applications in Python* – Chương 3 (F-statistic cho mô hình lồng nhau), Chương 5 (cross-validation, bootstrap), Chương 7 (§7.3–7.4 regression splines, chọn số mốc; §7.7 GAM). Số mục theo mục lục bản Python, nên đối chiếu lại với bản bạn đang dùng.
2. Horst, A. M., Hill, A. P., & Gorman, K. B. (2020). *palmerpenguins*. https://doi.org/10.5281/zenodo.3960218
3. Scikit-learn documentation: `SplineTransformer`, `cross_validate`, `GridSearchCV`.
