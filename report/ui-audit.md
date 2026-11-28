# Nghiệm thu thiết kế ứng dụng Palmer Penguins

Trên cùng cá thể Adelie cái (cánh bơi 197 mm, mỏ 44,5 × 17,3 mm), OLS conformal cho **3.571,7 g** ở cả hai phiên bản. Ở mức danh nghĩa 95%, khoảng cá thể là **2.935,5–4.208,0 g**. Có **67/69** cá thể test nằm trong khoảng tương ứng của chúng. Ứng dụng hiển thị kết quả này ngay dưới tiêu đề.

| Trước · commit `5d6b85e` | Sau · cùng đầu vào |
| --- | --- |
| ![Biểu mẫu, dự đoán và hình xu hướng trước lần nâng cấp](figures/prediction-before.png) | ![Kết quả có thước gram, độ bao phủ quan sát được và hình xu hướng](figures/prediction-dashboard.png) |

[Ảnh ở cửa sổ 480 px](figures/prediction-narrow.png) · [Boxplot mở bảng tứ phân vị](figures/prediction-errors.png) · [Báo cáo khoa học PDF](integration.pdf) · [Audit trước đây](https://github.com/phuctran33/regression-lab01/blob/5d6b85eebc32dd85a0afa53b88c5838f244b6ddd/report/ui-audit.md)

## Thiết kế dựa trên lượt dùng thử

- **Lần đầu mở.** Ở bản trước, kết quả, độ rộng và sai số nằm chung trong một khối chữ. Vạch từ cận dưới đến cận trên luôn phủ hết chiều ngang. Nay kết quả đọc theo ba câu hỏi về khối lượng ước tính, độ rộng của khoảng và sai số đã quan sát trên tập test. Trên thước từ 2.500 đến 6.500 g, dải OLS mặc định chiếm 31,81% ở mức 95% và 29,12% ở mức 90%. Hình thoi giữ nguyên vị trí. Thước mở rộng nếu dự đoán hoặc cận vượt miền hiển thị.
- **Lúc trình diễn.** Giữ kết quả, biểu mẫu và hình trên cùng trang để đọc được tác động của một thay đổi. Ba lựa chọn hình luôn hiện, có thể chuyển bằng phím mũi tên. Phần diễn giải dài và bảng số liệu mở theo nhu cầu. Hướng thiết kế dựa trên minh họa ba loài của Allison Horst, chữ Bricolage Grotesque đóng gói cùng notebook và màu xanh của biểu mẫu quan trắc. Màu sáng trên thước dành riêng cho khoảng dự đoán.
- **Khi sửa số đo.** Kết quả cũ được thu hồi. Vùng thông báo giữ chiều cao trong lúc sửa để thanh trượt ở nguyên vị trí trong biểu mẫu. Đo vị trí của cả ba thanh tương đối với đầu ứng dụng trước/sau một bước phím cho chênh lệch 0 px ở bố cục desktop. Ô trống hoặc vượt miền hiển thị hướng dẫn tại trường và khóa nút dự đoán.
- **Khi đọc hình trên điện thoại.** Ảnh cũ thu nhỏ cả chú giải. Ở bản mới, cùng đối tượng đồ thị Matplotlib được vẽ lại theo hai bố cục. Dưới 440 px chiều rộng vùng vẽ, chú giải đổi hàng và trục số bớt vạch. Hai nút **Xem biểu đồ** và **Đọc kết quả dự đoán** chuyển tiêu điểm trực tiếp. Chiều rộng thực tế của ô notebook quyết định bố cục.

## Chấm theo thang 100

Đây là điểm tự đánh giá sau thiết kế. Mức 80–89 ứng với luồng chính dùng được nhưng còn trở ngại rõ. Mức 90–94 cần bằng chứng ở nhiều trạng thái. Mức 95–99 cần thêm đối chiếu định lượng hoặc bằng chứng xuyên suốt cho tiêu chí đó. Đánh giá về thẩm mỹ và nhận diện vẫn là nhận định thiết kế. Chưa có phiên đánh giá với giảng viên hoặc người dùng công nghệ hỗ trợ.

| Khía cạnh | Điểm | Căn cứ và giới hạn |
| --- | ---: | --- |
| Thẩm mỹ | 94 | Ảnh hai phiên bản cho thấy kết quả có khoảng riêng và đọc được trước khi đến biểu mẫu. |
| Định hướng mỹ thuật | 94 | Minh họa đúng bộ dữ liệu, chữ tiêu đề và thước đo tạo nhận diện riêng cho ứng dụng. |
| Typography | 94 | Tiêu đề, khối lượng và nhãn có thứ bậc; chữ và dấu tiếng Việt được lưu ngoại tuyến. |
| Bố cục và khoảng cách | 93 | Khoảng cách theo chiều rộng ô, kết quả nằm đầu trang. Biểu mẫu vẫn cần cuộn trên điện thoại. |
| Hài hòa màu sắc | 94 | Xanh cho điều khiển, màu sáng cho khoảng, nâu cho cảnh báo; các trạng thái có thêm chữ hoặc ký hiệu. |
| Thứ tự đọc | 96 | Khối lượng → khoảng danh nghĩa → độ bao phủ thực nghiệm và MAE. Các giá trị đã đối chiếu với predictor. |
| Nhất quán thiết kế | 95 | Cùng ký pháp gram, thước và điều khiển trong hai chế độ. Các đường so sánh vẫn phân biệt bằng nét hoặc marker. |
| Tổ chức thông tin | 95 | Có thể chuyển qua ba câu hỏi phân tích mà vẫn thấy dự đoán. Bảng chi tiết mở ngay cạnh hình. |
| Usability | 94 | Thử năm đầu vào, bốn mô hình, lỗi và phục hồi. Chưa đo thời gian hoàn thành với người dùng mới. |
| Tương tác | 95 | Bàn phím hoạt động qua biểu mẫu, bộ chọn hình và nút chuyển tiêu điểm. Sửa đầu vào không đẩy thanh trượt. |
| Trực quan hóa khoa học | 97 | Dải và điểm cùng trục có tỷ lệ; phần dư có dấu, boxplot dùng trị tuyệt đối; số lượng trong hai bảng đều cộng thành 69. |
| Khả năng tiếp cận | 94 | axe, bàn phím, kiểm tra tương phản và chế độ màu bắt buộc. Cần thử trực tiếp với trình đọc màn hình. |
| Responsive | 94 | Cửa sổ 320–1440 px, bảng mở không tràn. Ở cửa sổ 320 px, Jupyter chỉ dành 202 px cho ứng dụng. |
| Đầy đủ chức năng | 97 | Bốn predictor, hai chế độ, CI/PI OLS, conformal, so sánh, ba hình, bảng và cảnh báo đã được kiểm tra. |
| Mức độ hoàn thiện | 94 | Font và minh họa cục bộ, kiểm thử hàm và trình duyệt, ảnh hiện hành trong PDF. Chưa thử thiết bị cảm ứng thật. |
| Đáp ứng rubric và lời dặn | 96 | Một luồng nhập, dự đoán và đọc độ bất định; có nhận xét và hình tương tác. Quyết định nghiệm thu học phần thuộc giảng viên. |

## Kiểm tra trình duyệt và accessibility

Hai lượt cuối dùng hai phiên Chrome 158 và hai kernel mới trong JupyterLab 4.5.6. Lượt thứ nhất chạy OLS/M1/M2/M3 qua cả hai chế độ, thử hai mức bao phủ, so sánh mô hình, thay cả năm đầu vào, cảnh báo ngoài miền loài, đổi biến số/phân loại, mở bảng và phục hồi sau nhập sai. Lượt thứ hai dùng bàn phím để dự đoán, chuyển tiêu điểm và chọn hình, thử theme tối, chế độ màu bắt buộc và các cửa sổ 1440, 1366, 768, 480, 390, 320 px. Hai lượt hoàn tất với 0 ngoại lệ trang. Runner bàn giao chạy thêm trên Chromium 159 và Firefox 159. Cả hai khớp số OLS, phục hồi sau bỏ trống ô số và đọc đủ 69 cá thể ở histogram hẹp. Cả hai trạng thái desktop/mobile trên mỗi trình duyệt đều ghi nhận 0 vi phạm axe. Playwright WebKit trên macOS 14.6.1 dừng ở bước tạo trang trình duyệt, trước khi mở notebook. Chưa xác minh Safari.

axe-core 4.14.1 quét vùng ứng dụng với các tag WCAG 2.0 A/AA, 2.1 A/AA và 2.2 AA. Bản đầu vào đã có 0 vi phạm ở trạng thái mặc định. Ở vòng thiết kế trung gian, axe phát hiện phần ghi chú sai cấu trúc danh sách định nghĩa. Rà ảnh cũng phát hiện chú giải thu hẹp vùng vẽ và ghi công minh họa tràn ở 320 px. Số bên dưới lấy sau khi sửa và chạy lại.

| Trạng thái | Quy tắc qua | Vi phạm | Cần xem thêm theo axe |
| --- | ---: | ---: | ---: |
| Bản trước, mặc định | 21 | 0 | 0 |
| Bản mới, mặc định | 23 | 0 | 0 |
| Ô số sai miền | 20 | 0 | 0 |
| Histogram mở bảng | 25 | 0 | 0 |
| Boxplot mở bảng | 25 | 0 | 0 |
| Benchmark OLS | 23 | 0 | 0 |
| Xu hướng, cửa sổ 480 px | 23 | 0 | 0 |
| JupyterLab Dark | 23 | 0 | 0 |
| Boxplot hẹp mở bảng | 25 | 0 | 0 |
| Histogram hẹp mở bảng | 25 | 0 | 0 |

Đối chiếu luminance sRGB cho tỷ lệ tương phản 14,04:1 giữa chữ chính và nền trắng, 8,53:1 cho chữ trên nút dự đoán, 5,28:1 cho viền focus trên nền biểu mẫu. Trên thước gram, đường nền đạt 3,15:1 so với nền kết quả, đoạn khoảng đạt 3,32:1 so với đường nền. Hai cận còn được ghi bằng số. Phạm vi đánh giá là widget trong notebook. Chưa thực hiện thử nghiệm VoiceOver/NVDA hoặc chứng nhận WCAG.

## Tái lập và nguồn thiết kế

- Chạy `python -m unittest discover -s tests -v` và `python -m source.check_week2 --notebook`. Có 29 kiểm thử hàm, gồm kiểm tra vị trí hai cận, dấu điểm và tỷ lệ hai khổ ảnh. Notebook chạy hết với kernel sạch. Bộ kiểm tra chung còn hai mục chờ M4 là `m4_configured` và `five_real_models_in_app`. Benchmark/nested CV/conformal trong JSON và ba hình từ evaluation tái tạo khớp từng byte.
- Để chạy kiểm tra trình duyệt tùy chọn, mở **bản sao notebook** bằng JupyterLab cục bộ với cờ `--expose-app-in-browser`. Trong môi trường kiểm thử có Playwright và Chromium hoặc Firefox đã cài, chạy `python tests/browser_prediction.py "<URL JupyterLab của bản sao>" --axe /duong/dan/axe.min.js`. Runner chạy toàn bộ cell và đối chiếu dự đoán, hai cận, sửa lỗi, bàn phím, số lượng mẫu và axe trên desktop/mobile. Thêm `--browser firefox` để thử bằng Firefox. [Runner](../tests/browser_prediction.py)
- Đã dùng **Frontend Design** của Anthropic để rà bố cục và nhận diện, **UI/UX Pro Max** để đối chiếu cách đọc khoảng và lựa chọn hình, **Design System** để thống nhất màu ngữ nghĩa, khoảng cách và trạng thái focus. Đối chiếu hướng dẫn WAI khi cung cấp số liệu đi kèm hình phức tạp. [Frontend Design](https://github.com/anthropics/skills/tree/e4bb5d960d87126ff3b3a8af1a104b7db67560ba/skills/frontend-design) · [UI/UX Pro Max](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill) · [WAI – Complex Images](https://www.w3.org/WAI/tutorials/images/complex/)
- Minh họa gốc **Artwork by @allison_horst**; chữ Bricolage Grotesque được đóng gói dưới SIL OFL 1.1. [Minh họa bộ dữ liệu](https://allisonhorst.github.io/palmerpenguins/articles/art.html) · [Nguồn phông chữ](https://github.com/google/fonts/tree/main/ofl/bricolagegrotesque) · [Giấy phép](../source/ui/fonts/OFL.txt)
