## Phương pháp

### Dữ liệu và kho dữ liệu
Home Credit Default Risk (Kaggle): ~307 nghìn hồ sơ vay có nhãn, dữ liệu tín dụng (bureau) của
khách hàng, các hồ sơ vay trước đó tại Home Credit và ~13,6 triệu dòng thanh toán trả góp.
Các file CSV gốc được mô hình hóa bằng **dbt trên DuckDB** theo ba tầng:

| Tầng | Model | Mục đích |
|---|---|---|
| staging | `stg_applications`, `stg_bureau`, `stg_previous_applications`, `stg_installment_payments` | đổi tên, ép kiểu, làm sạch (ví dụ giá trị giữ chỗ `365243` nghĩa là "không đi làm") |
| intermediate | `int_instalments`, `int_*_by_customer` | dữ liệu theo từng kỳ trả góp, gồm số tiền đã trả trước hạn / trong cửa sổ; tổng hợp theo khách hàng |
| marts | `mart_customer_features`, `mart_collections_events` | bảng sẵn sàng cho mô hình |

26 data test của dbt chạy ở mỗi lần build: khóa duy nhất và không rỗng, giá trị nhãn, tỷ lệ vỡ nợ
hợp lý, và hai kiểm tra point-in-time (không dùng thông tin tương lai) trên bảng thu hồi nợ.

### 1. Scorecard phê duyệt (xác suất vỡ nợ)
* Biến mục tiêu: `TARGET = 1`, khách hàng gặp khó khăn trả nợ (~8% hồ sơ).
* Chia 70/30 có phân tầng. Dữ liệu hồ sơ không có ngày theo lịch nên không chia out-of-time được.
* **WoE binning** tối ưu cho từng biến (optbinning), giữ biến có **IV ≥ 0,02**, sau đó hồi quy
  logistic trên giá trị WoE.
* Quy đổi ra điểm: **600 điểm tại odds 50:1, 20 điểm để odds tăng gấp đôi**. Điểm càng cao rủi ro
  càng thấp. Sáu hạng A–F có số lượng bằng nhau, cắt theo điểm của tập huấn luyện.
* Báo cáo: AUC, Gini, KS, PSI giữa tập huấn luyện và tập kiểm tra, tỷ lệ vỡ nợ theo hạng, đánh đổi
  giữa tỷ lệ phê duyệt và tỷ lệ vỡ nợ theo điểm cắt.
* **Mô hình đối chứng (challenger)**: LightGBM trên cùng bộ biến, giải thích bằng SHAP. Khoảng cách
  giữa hai mô hình cho thấy mô hình minh bạch phải đánh đổi bao nhiêu sức mạnh.

### 2. Thu hồi nợ: mô hình roll rate và ưu tiên hàng đợi
* **Sự kiện (vào thu hồi nợ)** = một kỳ trả góp đã thanh toán dưới 95% số tiền **5 ngày sau hạn
  (DPD 5)**. Khoảng một nửa số kỳ trả trễ được thanh toán trong vòng 3 ngày, nên một khoảng ân hạn
  ngắn giúp hàng đợi chỉ gồm những khoản mà nhân viên thu hồi thực sự cần xử lý.
* **Nhãn** `is_cured` = đã trả ít nhất 95% trong vòng **30 ngày** sau hạn; nếu không, khoản vay
  **chuyển sang DPD 30+ (roll)**. Các sự kiện chưa quan sát đủ 30 ngày trước thời điểm cắt dữ liệu
  bị loại.
* **Biến point-in-time**: các kỳ trả góp trước đó của khách hàng (số lượng, tỷ lệ trễ hạn), các lần
  vào thu hồi trước đó đã biết kết quả 30 ngày tại ngày vào thu hồi (tỷ lệ tự trả), số ngày từ lần
  vào thu hồi gần nhất, lịch sử trên cùng khoản vay, số tiền và tỷ lệ còn thiếu tại DPD 5, kỳ hạn và
  tiến độ khoản vay, thông tin nhân khẩu học ổn định. Các điểm như `EXT_SOURCE_*` bị **loại**: chúng
  được đo tại hồ sơ hiện tại, sau các sự kiện này, nên sẽ làm rò rỉ thông tin tương lai.
* **Kiểm định out-of-time**: huấn luyện trên 70% sự kiện cũ nhất, kiểm định trên 30% mới nhất.
* **Kiểm tra cách ưu tiên**: sự kiện được nhóm theo tuần (hàng đợi mà đội thu hồi nhìn thấy), xếp
  hạng trong mỗi nhóm theo bốn chiến lược, và X% đầu tiên được "xử lý". Thước đo là tỷ lệ số tiền
  quá hạn **không** tự được trả nằm trong phần hàng đợi được xử lý. Chỉ dùng kết quả quan sát thực
  tế, không giả định hiệu quả của cuộc gọi.

### 3. Thiết kế A/B test
Xếp hạng tốt không chứng minh được cuộc gọi làm thay đổi hành vi. Dashboard tính cỡ mẫu cho thử
nghiệm ngẫu nhiên cần thiết để đo mức cải thiện: kiểm định z hai phía cho hai tỷ lệ, alpha 0,05,
power 0,80.

### Hạn chế
* Các trường `DAYS_*` của Home Credit tính tương đối theo ngày nộp hồ sơ của từng khách hàng, không
  phải ngày theo lịch. Vì vậy "out-of-time" ở đây nghĩa là "gần với hồ sơ mới nhất của khách hàng
  hơn", và các nhóm theo tuần trộn lẫn nhiều ngày lịch. Bản triển khai thực tế sẽ dùng ngày thật.
* Dữ liệu không ghi lại các hành động thu hồi nợ, nên không thể ước lượng tác động nhân quả thực sự
  của cuộc gọi. Đó chính là mục đích của A/B test.
* Khối lượng mỗi ngày dùng để đổi cỡ mẫu ra thời gian chạy thử nghiệm là một giả định (2.000 khoản
  quá hạn mới mỗi ngày), có thể chỉnh trong dashboard.
