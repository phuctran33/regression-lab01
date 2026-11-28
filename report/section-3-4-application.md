## 3.4 Ứng dụng dự đoán

Mở JupyterLab bằng `python -m jupyterlab source/lab01.ipynb`, chọn kernel của môi trường đã cài thư viện rồi chạy toàn bộ notebook. Ở mục 5, chọn mô hình, nhập chiều dài cánh bơi, chiều dài và độ sâu mỏ bằng mm, chọn loài và giới tính, sau đó bấm **Dự đoán**. Kết quả chính là `body_mass_g` theo gram. Hai chế độ đều có OLS, M1 Ridge, M2 Cubic Spline Basis, M3 Natural Cubic Smoothing Spline và M4 Huber. Với M4, benchmark dùng cấu hình chọn trên toàn bộ train. Conformal chọn lại epsilon và alpha trên proper train rồi tính cận từ 55 mẫu hiệu chỉnh riêng. [Hướng dẫn notebook](../README.md#chạy-notebook) · [Mở notebook](../source/lab01.ipynb)

Trên laptop, đặt số đo ở cột trái và đọc kết quả ở cột phải. Trên điện thoại, nút **Tính từ số đo** nằm ngay cuối biểu mẫu. Trên thước gram, đoạn sáng nối hai cận và hình thoi đánh dấu dự đoán điểm. Thước lấy hai mốc ngoài từ miền khối lượng train làm tròn ra bậc 500 g và mở rộng nếu cận vượt miền. Các trang phân tích thu kết quả thành một dòng ghi loài, giới tính, mô hình, chế độ và dự đoán. **Chỉnh số đo** đưa người xem trở về biểu mẫu.

Chế độ mặc định dùng split conformal. Khi mở giao diện, năm pipeline được chọn cấu hình và fit trên 218 cá thể rồi hiệu chỉnh trên 55 cá thể. Ở nominal 95%, 67/69 mẫu test nằm trong khoảng OLS, độ rộng 1.273 g. Chuyển xuống 90% cho 66/69 mẫu và khoảng rộng 1.165 g. Cận và MAE thuộc cùng predictor với dự đoán. Các mức bao phủ mô tả hiệu năng trên quần thể trao đổi được, độ rộng cố định theo mô hình và mức. [Coverage và độ rộng theo mô hình](integration.md)

Trong chế độ **Benchmark**, giao diện gọi `predict` trên pipeline đã fit ở mục 3 của notebook. Một dự đoán dùng hàng đầu vào hiện tại. Biểu đồ tạo một dãy hàng chỉ thay biến đang chọn và giữ nguyên bốn đầu vào còn lại. Với biến phân loại, mô hình thử lần lượt các mức hợp lệ. Bật **So sánh các mô hình sẵn sàng** để hiện dự đoán và đường của các mô hình trên cùng đầu vào. Khi OLS và M3 cùng xuất hiện, biểu đồ số đo có thêm ô chênh lệch M3 − OLS theo gram. Các điểm nền là quan sát train với nhiều tổ hợp đầu vào. Dấu sao đánh dấu giá trị đang nhập và dự đoán của mô hình chính. Mục 4 có bảng MAE, MSE, RMSE và R² trên cùng tập test. [Mã giao diện](../source/prediction_app.py)

Thanh trượt và ô nhập số dùng chung một giá trị, giới hạn trong khoảng quan sát gộp của train và dropdown lấy các mức đã gặp trên train. Một giá trị trống hoặc vượt miền sẽ hiện hướng dẫn ngay tại trường và khóa nút dự đoán. Nếu số đo nhập ngoài khoảng đã quan sát trong loài đang chọn, kết quả vẫn hiện với chú thích ghi rõ số đo cần thận trọng. Khoảng từng biến riêng lẻ là một kiểm tra đơn giản về mức hỗ trợ của dữ liệu. Một tổ hợp nằm trong cả ba khoảng vẫn có thể hiếm trên train.

Trong chế độ benchmark, khi chọn OLS làm mô hình chính, giao diện hiển thị CI 95% cho khối lượng trung bình có điều kiện và PI 95% cho một cá thể mới, theo giả định OLS cổ điển. Cả hai được tính từ thiết kế đã tiền xử lý và kết quả OLS trên train. Giao diện chuẩn bị suy luận một lần khi mở. Thay đầu vào chỉ gọi dự đoán và cập nhật khoảng. [Công thức và giới hạn của CI/PI](section-3-3-evaluation.md#khoảng-cho-trung-bình-và-cho-một-cá-thể)

**Phần dư** dùng khối lượng thực trừ khối lượng dự đoán. Giá trị dương tương ứng với dự đoán thấp. **Sai số theo loài** dùng trị tuyệt đối nên biểu thị độ lớn sai lệch. Hộp cho biết khoảng tứ phân vị, vạch giữa là trung vị. Dùng phím mũi tên trong nhóm chọn biểu đồ để chuyển hình. Trên màn hình hẹp, **Xem biểu đồ**, **Đọc kết quả dự đoán** và **Đến nút dự đoán** chuyển tiêu điểm đến đúng vùng theo trạng thái của biểu mẫu. Cả hai hình lấy 69 cá thể test với predictor của chế độ đang chọn. Thay số đo của một cá thể có thể làm đổi dự đoán của cá thể đó, còn các sai số trên tập test cố định vẫn như cũ. Mở **Xem số liệu của biểu đồ** để đọc tần số hoặc các tứ phân vị. [Ảnh và kết quả kiểm thử](ui-audit.md)

### Sơ đồ đo và cách đọc một điểm

Ở bản vẽ bên cạnh kết quả, chọn cánh bơi hoặc một kích thước mỏ để chuyển tới ô nhập. Ngược lại, khi chạm, trỏ hoặc đưa tiêu điểm vào ô nhập, sơ đồ tô đậm đường tương ứng và ghi giá trị đang nhập theo mm. Hình toàn thân chỉ vị trí vai và chóp cánh bơi. Hình đầu phóng lớn tách chiều dài dọc sống mỏ lộ ra khỏi chiều sâu theo phương đứng. Viền mắt Adélie, dải cằm Chinstrap và mảng trắng trên đầu cùng màu mỏ Gentoo giúp nhận diện ba loài. Ba hình dùng chung tỷ lệ minh họa quy ước. [Cách hiểu số đo mỏ](https://allisonhorst.github.io/palmerpenguins/) · [Phép đo cánh bơi trong nghiên cứu thực địa](https://polarresearch.net/index.php/polar/article/download/3702/11088/) · [Nhận diện loài, Australian Antarctic Division](https://www.antarctica.gov.au/site/assets/files/45511/aad-penguin-poster.pdf)

Các đường vector được vẽ riêng cho Lab 01 bằng mã SVG có hỗ trợ của Codex. Tác giả ghi công **Lab 01 Regression** khi dùng lại sơ đồ theo CC BY 4.0. Hình chỉ vị trí đo, không mang thông tin về tỷ lệ cơ thể của cá thể đang dự đoán. Minh họa của Allison Horst là nguồn tham khảo hình thái. Phong cảnh ở đầu trang là ảnh minh họa tạo bằng ImageGen từ phiên bản trước. [SVG gốc](../source/ui/anatomy.svg) · [Giấy phép CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Artwork của Allison Horst](https://allisonhorst.github.io/palmerpenguins/articles/art.html)

Trỏ hoặc chạm một dấu trên biểu đồ, hoặc dùng phím mũi tên trong **Điểm dữ liệu**. Vòng tròn đánh dấu vị trí, nhãn ngắn nhận diện điểm và dòng bên dưới cho biết số liệu. Chẳng hạn, tại **Kiểm chứng khoảng**, chọn một ID có dấu ×. Đối chiếu khối lượng thật, dự đoán và hai cận rồi tìm cùng ID trong bảng. Mở rộng hình để xem rõ hơn và nhấn **Esc** để trở về điểm vừa chọn. Với các đường chồng nhau, danh sách cho phép chọn chính xác từng giá trị. [Điểm nằm ngoài khoảng](figures/application/point-inspector.png) · [Hình mở rộng](figures/application/expanded.png)

**Tải CSV** xuất đúng một dòng UTF-8, kèm năm đầu vào, mô hình, protocol, cỡ tập fit, dự đoán, cận và độ rộng theo gram, các metric của predictor và cảnh báo nếu có. Benchmark OLS có hai cột CI cho trung bình bên cạnh PI cá thể. Các trường không có giá trị theo protocol được để trống. Khi sửa đầu vào, phải dự đoán lại để xuất. **Cách đọc kết quả** và trợ giúp dưới từng biểu đồ giải thích thuật ngữ ngay cạnh dữ liệu.

### Demo hai phút

Chạy `python -m source.present` từ thư mục dự án. Mở địa chỉ localhost của Voilà, chờ khối lượng hiện ra và chọn **Toàn màn hình** nếu cần. Có thể dùng cùng kịch bản ở mục 5 của JupyterLab. [Màn hình trình diễn](figures/application/presentation.png)

| Thời gian | Thao tác trực tiếp | Điều cần cho người xem thấy |
| --- | --- | --- |
| 0:00–0:20 | Chọn **Gentoo · gần trung vị**, chọn đường chiều dài mỏ trên hình rồi bấm **Dự đoán**. | Năm đầu vào thuộc cùng một cá thể train. Đọc khối lượng, hai cận theo gram, số cá thể test trong khoảng và MAE. |
| 0:20–0:40 | Mở **So sánh mô hình**, giữ mức 95%. | Năm ký hiệu dùng một thước gram cho cùng cá thể. Độ rộng và mức chênh dự đoán điểm trả lời hai câu hỏi khác nhau. |
| 0:40–1:05 | Chọn M2, 90%, bấm **Dự đoán**, mở **Kiểm chứng khoảng**. Chạm dấu × hoặc chọn ID trong danh sách. Xem lại ở 95%. | Đối chiếu khối lượng thật với hai cận. Từ 7 xuống 1 cá thể ngoài khoảng trên cùng 69 ID khi khoảng rộng thêm 343,9 g. |
| 1:05–1:25 | Mở **Phương pháp**. Chuyển từ 95% về 90% và bấm **Dự đoán**. | Hạng được chọn đổi từ 54 về 51 trong 55 sai số hiệu chỉnh. Predictor fit trên 218 mẫu. Một tỉ lệ bao phủ trên test mô tả kết quả của cả tập. |
| 1:25–1:45 | Mở **So sánh mô hình**, chọn Benchmark và bấm **Dự đoán**. | Đường đứng là ΔMAE bằng 0. Cả bốn CI bootstrap ghép cặp đi qua đường này. Các chênh lệch MAE quan sát được chỉ khoảng vài gram. |
| 1:45–2:00 | Trở về **Dự đoán cá thể**, mở **Thực tế và dự đoán** hoặc **Sai số theo loài**. Tải CSV rồi chọn **Khôi phục mặc định**. | Điểm xa đường chéo giúp giải thích vì sao MAE trung bình có thể che một sai số cá thể lớn. Đối chiếu dự đoán xuất ra với màn hình. |

Ba preset theo loài là hàng đầy đủ gần trung vị nhất theo ba số đo đã chuẩn hóa trong loài. Hai preset còn lại lấy cá thể nhẹ nhất và cánh bơi dài nhất trong các hàng train đầy đủ. Panel **Phương pháp** có đồ thị hạng sai số, quy trình bảy bước và giới hạn trao đổi được. Phần tra cứu ở cuối mỗi màn hình giải nghĩa 18 thuật ngữ. [Ảnh mọi khu vực và bảng nghiệm thu](ui-audit.md)

### Ví dụ số để tái lập riêng chế độ benchmark

Chọn **Benchmark · toàn bộ train** và bật **So sánh các mô hình sẵn sàng** để tái hiện các giá trị trong bảng. Bấm **Dự đoán** sau khi đổi cá thể, mô hình, chế độ hoặc mức bao phủ. Đổi biểu đồ, biến hiển thị hoặc bật so sánh cập nhật ngay dự đoán đang xem.

| Demo | Đầu vào ban đầu | Giá trị ban đầu | Thao tác và giá trị cập nhật |
| --- | --- | ---: | --- |
| OLS, xu hướng theo cánh bơi | Adelie cái, cánh bơi 190 mm, mỏ dài 38,5 mm, mỏ sâu 17,9 mm | 3.469,9 g | Xem CI [3.403,0 đến 3.536,9] g và PI [2.904,4 đến 4.035,5] g. Chọn biểu đồ cánh bơi, tăng lên 205 mm. OLS dự đoán 3.664,5 g và M3 dự đoán 3.664,7 g. |
| M3, xu hướng theo độ sâu mỏ | Gentoo đực, cánh bơi 215 mm, mỏ dài 49 mm, mỏ sâu 15 mm | 5.281,2 g | Chọn biểu đồ độ sâu mỏ, tăng lên 16 mm. M3 dự đoán 5.366,6 g và OLS dự đoán 5.360,1 g. |
| Tiếp tục demo M3 với biến phân loại | Giữ các số đo sau thay đổi ở dòng trên | 5.366,6 g | Chọn biểu đồ giới tính và đổi sang **Cái**. M3 dự đoán 4.957,8 g. |

Ở demo OLS, khi thay duy nhất chiều dài cánh bơi, dự đoán tăng theo đường thẳng với độ dốc cố định. Ở demo M3, thay độ sâu mỏ đánh giá spline tại một điểm mới. Đổi giới tính thay hiệu ứng phân loại trong M3 trong khi các đóng góp của ba spline giữ nguyên. Các kịch bản là biến thiên có điều kiện theo mô hình, với các đầu vào do người dùng giữ cố định.

### Bàn giao

Cả năm mô hình dùng chung giao diện. Báo cáo Huber giải thích điểm ảnh hưởng và độ nhạy của M4 khi fit lại. [Báo cáo M4](huber-m4-week2.md) Phần đối chiếu hiệu năng OLS/M3 và liên kết đến các bàn giao hiện có nằm ở mục 3.3. [Đánh giá và diễn giải](section-3-3-evaluation.md)
