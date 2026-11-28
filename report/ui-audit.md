# Nghiệm thu ứng dụng Palmer Penguins

Ở ví dụ mặc định, OLS dự đoán **3.571,7 g**. Hai cận của khoảng cá thể 95% là **2.935,5 g và 4.208,0 g**. Người xem có thể mở **Kiểm chứng khoảng** và tự kiểm đếm 67/69 khối lượng thật nằm trong dải. Con số 95% là mục tiêu của quy trình hiệu chỉnh, còn 67/69 là kết quả quan sát trên tập test.

| Trước, commit `9936072` | Sau, cùng đầu vào và mức bao phủ |
| --- | --- |
| ![Giao diện trước khi mở rộng các khu vực phân tích](figures/prediction-before.png) | ![Giao diện sau, ví dụ OLS và thước khoảng dự đoán](figures/prediction-dashboard.png) |

[Khung JupyterLab 1440 × 900](figures/application/viewport-1440.png) · [1366 × 768](figures/application/viewport-1366.png) · [Widget ở cửa sổ 390 px](figures/prediction-narrow.png) · [Audit của bản trước](https://github.com/phuctran33/regression-lab01/blob/9936072d305b97bc8d794da7fbcb92485bf0a647/report/ui-audit.md)

## Nghiệm thu từng chức năng

| Chức năng | Bằng chứng thực chạy | Trạng thái | Giới hạn |
| --- | --- | --- | --- |
| 1. Bốn khoảng cho cùng cá thể | [Cùng trục gram](figures/application/interval-comparison.png). Bảng đối chiếu từng điểm, cận, độ rộng, coverage và MAE với bốn predictor ở 90% và 95%. | PASS | Áp dụng cho bốn predictor conformal đã hiệu chỉnh. |
| 2. Bao phủ và độ rộng | [Tám tổ hợp](figures/application/coverage-width.png). M2 tăng từ 1.016,15 lên 1.360,08 g, số ngoài từ 7 xuống 1. Đường, marker và bảng nhận diện mô hình. | PASS | Mỗi cá thể tương ứng 1/69 trên tập này. |
| 3. Khoảng trên test | [Dải của M2 ở 90%](figures/application/test-intervals.png). Kiểm lại 69 ID, dự đoán, cận và trạng thái với predictor ở cả hai mức. Dấu × là ngoài, vòng tròn là trong. | PASS | Trục ngang xếp theo dự đoán tăng dần. Đây là tập test nhóm đã từng sử dụng. |
| 4. Nhận xét động | [Thực tế so với dự đoán](figures/application/actual-predicted.png), [sai số theo loài](figures/prediction-errors.png), [histogram](figures/application/residual-histogram.png). Kiểm ID lệch nhất, MAE, sai lệch có dấu và số quan sát. | PASS | Các diễn giải mô tả dữ liệu và predictor đang chọn. |
| 5. Cá thể mẫu | [Kiểm thử các preset](../tests/test_prediction_evidence.py). Năm hàng train có thật, đủ đầu vào. Trình duyệt chọn lần lượt từng hàng và dự đoán lại. | PASS | Ba mẫu gần trung vị của loài, hai mẫu ở biên trong các hàng đầy đủ. |
| 6. Khôi phục mặc định | [Ô số vượt miền](figures/application/input-error.png). Từ ô sai, reset phục hồi 197 mm, OLS, 95%, đầu vào và kết quả gốc. Thử cả bảng đang mở. | PASS | Reset tác động lên widget hiện tại. |
| 7. Thực tế và dự đoán | [Hai trục gram, đường y = x](figures/application/actual-predicted.png). 69 hàng ở cả hai protocol. M3 benchmark RMSE 307,0 g, M3 conformal MAE 258,2 g. | PASS | Mỗi lần đọc một mô hình. |
| 8. Quy trình conformal | [Bảy bước và đường sai số](figures/application/method.png). Tái tính hạng 51 và 54 trên 55 sai số. | PASS | Bao phủ biên dựa trên giả định trao đổi được. Bán kính cố định cho mỗi mô hình và mức. |
| 9. Tra cứu thuật ngữ | 18 mục tiếng Việt, mở và đóng bằng Enter hoặc bấm chạm. [Nội dung](../source/ui/glossary.html) | PASS | Thuật ngữ tập trung ở cuối mỗi vùng. |
| 10. Phản hồi thao tác | [Runner](../tests/browser_prediction.py) kiểm trạng thái chờ, lỗi, khôi phục và cấu hình giảm chuyển động. Số mới xuất hiện khi phép tính hoàn tất. | PASS | Chuyển nhẹ nền/trạng thái. |
| 11. Chức năng cũ | [Xu hướng](figures/prediction-dashboard.png), [boxplot](figures/prediction-errors.png), [histogram](figures/application/residual-histogram.png), CI và PI OLS, bốn mô hình, năm đầu vào, so sánh có điều kiện và cảnh báo. | PASS | CI và PI cổ điển chỉ có ở OLS benchmark. |
| 12. Khả năng tiếp cận | Axe, bàn phím, tên điều khiển, thông báo kết quả, giao diện tối và zoom đã kiểm. Chi tiết bên dưới. | BLOCKED một phần | Chưa xác nhận hành vi đọc thực tế bằng VoiceOver. |
| 13. Bố cục đáp ứng | [Laptop](figures/application/viewport-1366.png), [điện thoại](figures/prediction-narrow.png). Cửa sổ 320 px chỉ còn 202 px cho widget, kể cả khi mở bảng vẫn cuộn trong vùng bảng. | PASS | Đã thử qua trình duyệt desktop. Chưa đo trên thiết bị cảm ứng thật. |
| 14. Tính đúng thống kê | [Test theo từng ID](../tests/test_prediction_evidence.py) và [kết quả gốc](comparison.json). JSON thực nghiệm và ba hình evaluation tái tạo khớp từng byte. | PASS | 69 cá thể test thuộc cùng nguồn quan sát. |
| 15. Trình diễn | [Demo hai phút](section-3-4-application.md#demo-hai-phút). Đã thực hiện chọn mẫu, tính lại, đổi mức, kiểm chứng, so sánh và reset trên trình duyệt. | PASS | Chưa đo thời gian thao tác của người mới. |
| 16. Tái lập | [Lệnh chạy](handoff.md), [runner trình duyệt](../tests/browser_prediction.py). Notebook chạy từ kernel sạch, 32/32 unit test đạt. | PASS cho bốn mô hình | WebKit/Safari chưa xác minh. Hai kiểm tra cấp nhóm đang chờ M4. |

## Đối chiếu với đề và lời dặn

| Nguồn | Yêu cầu hoặc mục đích | Vị trí bàn giao |
| --- | --- | --- |
| Đề Lab 01, §3.4 | Giao diện tương tác để nhập đặc trưng và nhận dự đoán | [Notebook mục 5](../source/lab01.ipynb), [luồng nhập và dự đoán](section-3-4-application.md) |
| Đề Lab 01, §3.1–3.3 và mục nộp | Giải thích lựa chọn, so sánh hiệu năng, tổng hợp báo cáo | [Báo cáo khoa học](integration.md), [PDF](integration.pdf), [đối chiếu toàn nhóm](handoff.md) |
| Lời dặn nhóm ghi nhận | Dùng đơn giản, chỉ rõ độ tin cậy, có nhận xét, histogram và boxplot, chú trọng hình thức. Tỷ trọng định hướng khoảng 90% báo cáo và 10% ứng dụng. | [Kết quả mặc định](figures/prediction-dashboard.png), [69 cá thể test](figures/application/test-intervals.png), [phân bố sai số](figures/application/residual-histogram.png) |
| Nhóm chủ động mở rộng | Preset từ train, forest plot ΔMAE, hai loại CV, tám mức so sánh mô hình–coverage, bảy bước hiệu chỉnh, glossary và kiểm thử UI. | [Bảng và forest plot](figures/application/benchmark.png), [phương pháp](figures/application/method.png), [artifact](comparison.json) |

## Đánh giá thiết kế trước và sau

Rà bản dựng trong JupyterLab, ảnh trước/sau và các màn hình kết quả với cùng dữ liệu. Điểm dưới đây là nhận định thiết kế trên thang 100 cho lần rà này. Các chênh lệch phản ánh cả phạm vi mười chức năng mới. Các phép kiểm có kết quả nhị phân nằm ở bảng nghiệm thu phía trên. Nhóm chưa tổ chức chấm mù với người dùng hoặc giảng viên.

| Tiêu chí | Trước | Sau | Căn cứ cho nhận định |
| --- | ---: | ---: | --- |
| Thẩm mỹ | 86 | 95 | Tiêu đề có phong cảnh vùng cực, khoảng trắng và điểm nhấn thị giác ở khối lượng. |
| Chỉ đạo mỹ thuật | 83 | 95 | Bố cục gợi sổ tay thực địa xuyên suốt trang, đồ thị giữ nền trắng để đọc. |
| Nhận diện | 87 | 96 | Chữ nghiêng ở tên Penguins, minh họa ba loài cạnh biểu mẫu và tranh phong cảnh ở tiêu đề. |
| Typography | 85 | 94 | Ba cấp chữ cho tên ứng dụng, dự đoán và số liệu đối chiếu. Tên trục còn nhỏ tại chiều rộng tối thiểu. |
| Bố cục | 84 | 94 | Lệnh dự đoán và kết quả hiện đồng thời ở hai khổ laptop. |
| Khoảng cách | 87 | 93 | Thước và hai nhóm chỉ số có khoảng tách riêng. Trên mobile hẹp, luồng nhập vẫn dài. |
| Hài hòa màu | 85 | 95 | Nền giấy, xanh nước biển và màu hổ phách phân vai cho cảnh nền, số chính và bằng chứng. |
| Thứ bậc thị giác | 88 | 95 | Khối lượng lớn nhất, hai cận và số mẫu có vị trí cố định. |
| Kiến trúc thông tin | 79 | 96 | Bốn khu vực dùng chung trạng thái, dữ liệu hàng chỉ mở khi cần. |
| Điều hướng | 81 | 94 | Chọn vùng bằng bàn phím, hai lối chuyển tiêu điểm trên khổ hẹp. |
| Dễ sử dụng | 86 | 93 | Năm preset, dự đoán, reset, lỗi tại trường. Đổi tham số còn cần bấm Dự đoán. |
| Thiết kế tương tác | 87 | 94 | Khi sửa số đo, kết quả và hình cũ được bỏ cùng nhau. |
| Trực quan hóa khoa học | 82 | 96 | Trục gram chung, dải 69 cá thể, đường chéo y = x, mốc ΔMAE = 0. |
| Truyền đạt thống kê | 84 | 96 | Nêu mẫu số, độ rộng, số ngoài khoảng, bậc phân vị và giới hạn suy rộng. |
| Khả năng tiếp cận | 88 | 92 | Có bảng, mô tả, trạng thái đọc và phím tắt theo điều khiển native. Chờ thử VoiceOver. |
| Hiển thị đáp ứng | 88 | 93 | Chuyển sang hình khổ hẹp, bảng cuộn cục bộ và kiểm ở zoom trình duyệt 200%. |
| Đầy đủ chức năng | 67 | 97 | Đã thử cả mười nhóm chức năng cho bốn predictor. |
| Nhất quán thiết kế | 89 | 95 | Nút, thước, tiêu đề, chú giải và định dạng gram lặp lại giữa các vùng. |
| Hiệu năng cảm nhận | 85 | 90 | Dùng lại các hình theo tập test và predictor đã fit. Kernel khởi động và lần vẽ Matplotlib đầu vẫn cần thời gian. Chưa benchmark độ trễ. |
| Mức hoàn thiện | 85 | 94 | Tài nguyên cục bộ, ảnh thật, các trạng thái bàn phím, lỗi và phục hồi. |
| Trải nghiệm demo | 82 | 95 | Có luồng từ một cá thể cụ thể đến 69 quan sát và reset. |
| Đáp ứng rubric Application | 90 | 96 | Nhập, dự đoán, giải thích và hình tương tác tái lập từ notebook. |
| Đáp ứng lời dặn giảng viên | 86 | 96 | Người xem kiểm đếm được các trường hợp dự đoán trượt ngay trong giao diện. |

## Kiểm thử trình duyệt và số học

- **Chromium 159 và Firefox 159.** Mỗi trình duyệt chạy notebook riêng, hoàn tất luồng bốn mô hình, hai protocol, năm đặc trưng, năm preset, đổi mức, chuyển khu vực, đọc glossary, hồi phục sau nhập sai và dự đoán lặp lại. Axe-core 4.14.1 ghi nhận **0 vi phạm, 0 kết quả cần xem thêm** ở mỗi trong 10 trạng thái trên mỗi trình duyệt. Các trạng thái gồm mặc định, ô số sai, bốn khoảng, dải test, coverage–width, phương pháp, benchmark, thực tế–dự đoán, sai số theo loài và mobile.
- **Chrome 158.** Rà lại cây accessibility, phím mũi tên trong điều hướng, thứ tự focus, mở bảng dữ liệu ở 320 px, reset bảng thuật ngữ và JupyterLab Dark. Ở hai khổ 1440 × 900 và 1366 × 768, sau khi cuộn đến đầu widget, nút dự đoán bắt đầu ở khoảng y = 298 px, số dự đoán ở khoảng y = 397 px. Đã thử thêm 768, 480, 390 và 320 px. Ở Chromium, zoom thật qua API trình duyệt từ 100% lên 200% làm viewport nội dung từ 1366 còn 683 CSS px, chuyển được sang 69 cá thể test. Runner còn thử đổi kích thước, forced colors và reduced motion.
- **Tương phản và cách đọc.** Chữ chính trên trắng đạt 12,69:1, nút chính đạt 7,79:1 và focus trên nền biểu mẫu đạt 5,53:1 theo luminance sRGB. Đường nền thước đạt ít nhất 3,10:1 so với nền kết quả, phần khoảng đạt 3,21:1 so với đường nền. Hai cận còn hiện bằng chữ số. Arrow chọn radio trong nhóm, Enter mở glossary, cập nhật live region đọc khối lượng và hai cận.
- **Giới hạn kiểm thử.** Playwright WebKit dừng tại bước tạo trang trên macOS 14.6.1. SafariDriver cũng chưa tạo được phiên trong lần thử. Chưa xác nhận Safari, VoiceOver hoặc thiết bị cảm ứng thật. Axe chỉ đánh giá phần widget theo các tag WCAG A/AA được runner chọn.
- **Đối chiếu số học.** 32/32 unit test, kiểm tra dependency, compile/import và notebook kernel sạch đạt. Bốn mô hình fit/predict trên cùng 69 ID. JSON chứa benchmark, nested CV, bootstrap, conformal tái tạo giống từng byte, ba PNG của evaluation cũng khớp. Hai dòng FAIL của kiểm tra toàn nhóm là `m4_configured` và `five_real_models_in_app`. [Mã kiểm tra](../source/check_week2.py) · [Test dữ liệu của các hình](../tests/test_prediction_evidence.py)

Để tái lập các luồng trình duyệt, mở **bản sao notebook** bằng JupyterLab với cờ `--expose-app-in-browser`. Trong môi trường QA đã cài Playwright, Chromium/Firefox và axe-core cục bộ, chạy lệnh bên dưới với URL của bản sao. Thêm `--browser firefox` cho Firefox. `--evidence-dir` xuất ảnh của từng khu vực và hai viewport thật. Ảnh widget toàn chiều cao dùng viewport kéo dài để JupyterLab giữ cell trong DOM. [Runner trình duyệt](../tests/browser_prediction.py)

```bash
python tests/browser_prediction.py "<URL notebook cục bộ>" --axe /duong/dan/axe.min.js --evidence-dir /tmp/penguin-screens
```

## Quyết định thiết kế, nguồn và công cụ

- Giữ một ứng dụng trong notebook. Đã xem xét mở rộng trang cuộn dọc và tách khu vực theo câu hỏi khoa học. Bốn khu vực cho phép chuyển từ cá thể sang bằng chứng trên test mà vẫn giữ predictor, mức bao phủ và đầu vào. Ở các khổ hẹp, thước và số liệu xếp lại theo dòng, từng bảng có thể cuộn riêng.
- Giữ nền đồ thị sáng, hiển thị đơn vị và dùng cả marker/nét vẽ. Đặt mô tả và bảng ngay sau đồ thị để người đọc kiểm chứng từng điểm. Thiết kế nhóm radio dùng hành vi bàn phím native. [WAI – Complex Images](https://www.w3.org/WAI/tutorials/images/complex/) · [WAI – Radio Group](https://www.w3.org/WAI/ARIA/apg/patterns/radio/) · [Observable Plot – Rule](https://observablehq.com/plot/marks/rule)
- Đã dùng **Frontend Design** của Anthropic, **UI/UX Pro Max** và **Design System**. Skill mới cài là Frontend Design, từ commit chính thức `e4bb5d9` qua Skill Installer. Dùng ImageGen để tạo phong cảnh minh họa mang tính trang trí, đóng gói JPEG cục bộ. Ảnh về kết quả thực nghiệm đều chụp từ notebook. Playwright và axe-core chỉ phục vụ QA. [Frontend Design](https://github.com/anthropics/skills/tree/e4bb5d960d87126ff3b3a8af1a104b7db67560ba/skills/frontend-design) · [UI/UX Pro Max](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill)
- Minh họa ba loài của Allison Horst, chữ Bricolage Grotesque được đóng gói dưới SIL OFL 1.1. [Artwork by @allison_horst](https://allisonhorst.github.io/palmerpenguins/articles/art.html) · [Giấy phép phông](../source/ui/fonts/OFL.txt)
- Checklist M4 cho lần tích hợp sau gồm tiếp nhận estimator thật của Phúc, kiểm tra fit/hyperparameter selection chỉ dùng train, đối chiếu đúng 69 ID, bổ sung benchmark và bootstrap, fit lại trên 218 proper train, hiệu chỉnh riêng trên 55 mẫu và kiểm hai mức. Sau đó mới mở lựa chọn M4 tương ứng, chạy lại UI, notebook và toàn bộ suite. [Bàn giao của nhóm](handoff.md)
