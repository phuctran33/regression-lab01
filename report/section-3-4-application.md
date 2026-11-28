## 3.4 Ứng dụng dự đoán

Mở JupyterLab bằng `python -m jupyterlab source/lab01.ipynb`, chọn kernel của môi trường đã cài thư viện rồi chạy toàn bộ notebook. Ở mục 5, chọn mô hình, nhập chiều dài cánh bơi, chiều dài và độ sâu mỏ bằng mm, chọn loài và giới tính, sau đó bấm **Dự đoán**. Kết quả chính là `body_mass_g` theo gram. Nhánh bàn giao có OLS, M1 Ridge, M2 Cubic Spline Basis và M3 Natural Cubic Smoothing Spline. M4 Huber được ghi là **Chưa sẵn sàng** ở bảng huấn luyện. Khi M4 có estimator, notebook có thể thêm vào benchmark. Chế độ conformal cần procedure chọn cấu hình của M4 và kiểm chứng coverage riêng. [Hướng dẫn notebook](../README.md#chạy-notebook) · [Mở notebook](../source/lab01.ipynb)

Dải kết quả ngay dưới hàng điều khiển ghi khối lượng dự đoán, loài và giới tính. Trên thước gram, đoạn sáng nối hai cận và hình thoi đánh dấu dự đoán điểm. Thước lấy hai mốc ngoài từ miền khối lượng train làm tròn ra bậc 500 g và mở rộng nếu cận vượt miền. Có thể đối chiếu độ rộng giữa các lần dự đoán bằng hai mốc ghi trên thước.

Chế độ mặc định dùng split conformal. Khi mở giao diện, bốn pipeline được chọn cấu hình và fit trên 218 cá thể rồi hiệu chỉnh trên 55 cá thể. Ở nominal 95%, 67/69 mẫu test nằm trong khoảng OLS, độ rộng 1.273 g. Chuyển xuống 90% cho 66/69 mẫu và khoảng rộng 1.165 g. Cận và MAE thuộc cùng predictor với dự đoán. Các mức bao phủ mô tả hiệu năng trên quần thể trao đổi được, độ rộng cố định theo mô hình và mức. [Coverage và độ rộng theo mô hình](integration.md)

Trong chế độ **Benchmark**, giao diện gọi `predict` trên pipeline đã fit ở mục 3 của notebook. Một dự đoán dùng hàng đầu vào hiện tại. Biểu đồ tạo một dãy hàng chỉ thay biến đang chọn và giữ nguyên bốn đầu vào còn lại. Với biến phân loại, mô hình thử lần lượt các mức hợp lệ. Bật **So sánh các mô hình sẵn sàng** để hiện dự đoán và đường của các mô hình trên cùng đầu vào. Khi OLS và M3 cùng xuất hiện, biểu đồ số đo có thêm ô chênh lệch M3 − OLS theo gram. Các điểm nền là quan sát train với nhiều tổ hợp đầu vào. Dấu sao đánh dấu giá trị đang nhập và dự đoán của mô hình chính. Mục 4 có bảng MAE, MSE, RMSE và R² trên cùng tập test. [Mã giao diện](../source/prediction_app.py)

Thanh trượt và ô nhập số dùng chung một giá trị, giới hạn trong khoảng quan sát gộp của train và dropdown lấy các mức đã gặp trên train. Một giá trị trống hoặc vượt miền sẽ hiện hướng dẫn ngay tại trường và khóa nút dự đoán. Nếu số đo nhập ngoài khoảng đã quan sát trong loài đang chọn, kết quả vẫn hiện với chú thích ghi rõ số đo cần thận trọng. Khoảng từng biến riêng lẻ là một kiểm tra đơn giản về mức hỗ trợ của dữ liệu. Một tổ hợp nằm trong cả ba khoảng vẫn có thể hiếm trên train.

Trong chế độ benchmark, khi chọn OLS làm mô hình chính, giao diện hiển thị CI 95% cho khối lượng trung bình có điều kiện và PI 95% cho một cá thể mới, theo giả định OLS cổ điển. Cả hai được tính từ thiết kế đã tiền xử lý và kết quả OLS trên train. Giao diện chuẩn bị suy luận một lần khi mở. Thay đầu vào chỉ gọi dự đoán và cập nhật khoảng. [Công thức và giới hạn của CI/PI](section-3-3-evaluation.md#khoảng-cho-trung-bình-và-cho-một-cá-thể)

**Phần dư** dùng khối lượng thực trừ khối lượng dự đoán. Giá trị dương tương ứng với dự đoán thấp. **Sai số theo loài** dùng trị tuyệt đối nên biểu thị độ lớn sai lệch. Hộp cho biết khoảng tứ phân vị, vạch giữa là trung vị. Dùng phím mũi tên trong nhóm chọn biểu đồ để chuyển hình. Trên màn hình hẹp, **Xem biểu đồ**, **Đọc kết quả dự đoán** và **Đến nút dự đoán** chuyển tiêu điểm đến đúng vùng theo trạng thái của biểu mẫu. Cả hai hình lấy 69 cá thể test với predictor của chế độ đang chọn. Thay số đo của một cá thể có thể làm đổi dự đoán của cá thể đó, còn các sai số trên tập test cố định vẫn như cũ. Mở **Xem số liệu của biểu đồ** để đọc tần số hoặc các tứ phân vị. [Ảnh và kết quả kiểm thử](ui-audit.md)

### Demo hai phút

| Thời gian | Thao tác trực tiếp | Điều cần cho người xem thấy |
| --- | --- | --- |
| 0:00–0:20 | Ở **Dự đoán cá thể**, chọn **Gentoo · gần trung vị**, bấm **Dự đoán**. | Cả năm đầu vào thuộc một hàng train có thật. Đọc khối lượng, hai cận theo gram và MAE. Chọn mẫu khác sẽ xóa kết quả cũ trong khi chờ tính lại. |
| 0:20–0:40 | Mở **So sánh mô hình**, giữ mức 95%. | Bốn ký hiệu dùng một thước gram cho cùng cá thể. Độ rộng và mức chênh dự đoán điểm trả lời hai câu hỏi khác nhau. |
| 0:40–1:05 | Chọn M2, 90%, bấm **Dự đoán**, mở **Kiểm chứng khoảng**. Sau đó chọn 95% và dự đoán lại. | Dấu × là khối lượng thật nằm ngoài dải. Từ 7 xuống 1 cá thể ngoài khoảng trên cùng 69 ID. Ở **Bao phủ và độ rộng**, cái giá của sự thay đổi là khoảng rộng thêm 343,9 g. |
| 1:05–1:25 | Mở **Phương pháp**. Chuyển từ 95% về 90% và bấm **Dự đoán**. | Hạng được chọn đổi từ 54 về 51 trong 55 sai số hiệu chỉnh. Predictor fit trên 218 mẫu. Một tỉ lệ bao phủ trên test mô tả kết quả của cả tập. |
| 1:25–1:45 | Mở **So sánh mô hình**, chọn Benchmark và bấm **Dự đoán**. | Đường đứng là ΔMAE bằng 0. Cả ba CI bootstrap ghép cặp đi qua đường này. Các chênh lệch MAE quan sát được chỉ khoảng vài gram. |
| 1:45–2:00 | Trở về **Dự đoán cá thể**, chọn **Thực tế và dự đoán** hoặc **Sai số theo loài**. Bấm **Khôi phục mặc định**. | Điểm xa đường chéo giúp giải thích vì sao MAE trung bình có thể che một sai số cá thể lớn. Reset khôi phục OLS conformal 95%, bộ số đo và đồ thị xu hướng. |

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

Các mô hình sẵn sàng đã được thử qua chung giao diện này. Bản demo của nhóm có thể thêm Huber sau khi Phúc bàn giao estimator, cùng nhận xét về kết quả dự đoán của từng chủ mô hình. Phần đối chiếu hiệu năng OLS/M3 và liên kết đến các bàn giao hiện có nằm ở mục 3.3. [Đánh giá và diễn giải](section-3-3-evaluation.md)
