# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:2A202603583**
- **Họ và tên: Lê Quang Linh**
- **SEED** (in ở ô đầu notebook):2640

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.826      0.904        0.919   
2. BoW + MLP (deep learning)         1.000   0.895      0.945        0.947   
3a. Encoder (MiniLM) + kNN           0.862   0.942      0.900        0.974   
3b. Encoder (MiniLM) + LogReg        0.952   0.930      0.941        0.974   
3c. Encoder (MiniLM) + MLP head      0.952   0.930      0.941        0.974   
4a. LLM nhỏ zero-shot                0.905   0.950      0.927        0.927   
4b. LLM lớn zero-shot                1.000   1.000      1.000        1.000   
4c. LLM nhỏ few-shot                 0.870   1.000      0.930        0.930   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            9.708       7729.507       0.053   
2. BoW + MLP (deep learning)         8.869      33187.235       0.298   
3a. Encoder (MiniLM) + kNN          87.639         24.392      52.744   
3b. Encoder (MiniLM) + LogReg       88.221         34.614      28.141   
3c. Encoder (MiniLM) + MLP head     88.657         37.941      40.790   
4a. LLM nhỏ zero-shot                0.000          1.064     352.279   
4b. LLM lớn zero-shot                0.000          1.303     270.160   
4c. LLM nhỏ few-shot                 0.000          0.644    2229.488   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.111           0.0  
2. BoW + MLP (deep learning)          1.525           0.0  
3a. Encoder (MiniLM) + kNN           69.451           0.0  
3b. Encoder (MiniLM) + LogReg        39.369           0.0  
3c. Encoder (MiniLM) + MLP head      56.346           0.0  
4a. LLM nhỏ zero-shot              2389.186           0.0  
4b. LLM lớn zero-shot              2304.124           0.0  
4c. LLM nhỏ few-shot               2444.601           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.
## 3. Dự đoán và thực tế

Ba dự đoán lệch đáng chú ý nhất:
1. Dự đoán về Few-shot
- Dự đoán: Khi thêm 4 ví dụ vào prompt, F1 sẽ tăng và chi phí mỗi tin tăng khoảng 0,5 lần.
- Thực tế: F1 của LLM nhỏ tăng từ 0.927 lên 0.930, chỉ tăng 0.003. P95 tăng từ 2389.19 ms lên 2444.60 ms.
- Vì sao lệch: Bốn ví dụ không cung cấp đủ thông tin mới để cải thiện đáng kể khả năng phân loại, trong khi prompt dài hơn làm tăng lượng token đầu vào.
2. Dự đoán về 50 case khó
- Dự đoán: BoW + Logistic Regression đạt F1 khoảng 65–75%, BoW + MLP đạt 70–80%, MiniLM + MLP đạt 85–92%.
- Thực tế: F1 lần lượt là 0.333, 0.421 và 0.706.
- Vì sao lệch: Những tin nhắn này không tuân theo các mẫu từ khóa quen thuộc trong dữ liệu train, khiến các mô hình học từ dữ liệu thông thường giảm hiệu quả mạnh.
3. Dự đoán về LLM lớn và LLM nhỏ
- Dự đoán: LLM lớn có F1 cao hơn LLM nhỏ nhưng chênh lệch không lớn.
- Thực tế: LLM lớn zero-shot đạt F1 1.000, trong khi LLM nhỏ đạt 0.927 trên tập SUB.
- Vì sao lệch: Model lớn xử lý tốt hơn những tin nhắn có cách diễn đạt phức tạp. Tuy nhiên, tập SUB chỉ có 40 tin nên chưa thể kết luận model lớn luôn tốt hơn trong thực tế.
## 4. Phân tích 50 case khó

Nhóm case gây khó khăn:
- Với BoW: nhóm spam giọng tự nhiên, thao túng/lừa đảo và né từ khóa.
- Với Embedding: các tin nhắn có ngữ cảnh phức tạp, đặc biệt là những tin ham chứa nhiều từ khóa spam.
- Với LLM: các tin nhắn ham có chứa từ khóa đáng ngờ và nhóm nói về spam.
## 5. Thử thách cá nhân

đề: Thay đổi decision threshold từ 0.5 sang 0.2 và 0.8 cho mô hình Logistic Regression ở Phần 1. So sánh Precision, Recall và F1.

Đầu tiên, sử dụng predict_proba() để tính xác suất dự đoán spam của từng tin nhắn. Sau đó thay đổi ngưỡng phân loại:
- Nếu xác suất lớn hơn hoặc bằng threshold thì dự đoán spam.
- Nếu nhỏ hơn threshold thì dự đoán ham.
Kết quả thực nghiệm:
Threshold	Precision	Recall	F1
0.2	0.884	0.884	0.884
0.5	1.000	0.826	0.904
0.8	1.000	0.605	0.754

==> Khi tăng threshold từ 0.2 lên 0.8, mô hình trở nên thận trọng hơn khi phân loại một tin nhắn là spam. Vì vậy, Recall giảm từ 0.884 xuống 0.605, trong khi Precision tăng từ 0.884 lên 1.000.

Đối với ngân hàng, chọn threshold 0.8 vì ngân hàng đặc biệt sợ chặn nhầm tin nhắn OTP. Ngưỡng cao giúp hạn chế False Positive, dù phải chấp nhận bỏ sót một số tin nhắn spam.
## 6. Phương án cho sản phẩm thật

Phương án lựa chọn: MiniLM + Logistic Regression (3b).
Lý do:
- F1 test đạt 0.941, cao hơn hai mô hình BoW.
- F1 trên tập SUB đạt 0.974.
- P95 đạt 39.369 ms, đáp ứng yêu cầu dưới 50 ms.
## 7. AI log và điều đã học
Phần 1	Bag-of-Words nhị phân là gì?	Giải thích cách chuyển văn bản thành vector 0 và 1.	Đối chiếu với hàm vectorize() trong notebook.
Phần 1	Cách khôi phục notebook và clone repository?	Hướng dẫn mở notebook từ GitHub và clone repo về máy.	Thực hiện lại trên Colab và máy tính.
Phần 4	Chọn mô hình nào đáp ứng yêu cầu độ trễ và chất lượng?	So sánh các mô hình và đề xuất MiniLM + MLP.	Đối chiếu với số liệu thực nghiệm và nhận thấy MiniLM + LogReg mới là phương án đạt SLA.
Phần 5	Thiết kế hệ thống kết hợp ML và LLM như thế nào?	Đề xuất ML xử lý tin dễ, LLM xử lý tin khó.	Kiểm tra ngưỡng xác suất và chi phí dự kiến.
Phần 5	Điều gì dự đoán sai sau khi thực nghiệm?	Gợi ý nhận xét về hạn chế của ML và ưu điểm của LLM.	Điều chỉnh nội dung theo kết quả thực tế.


Điều đã học: hiểu rõ hơn sự khác nhau giữa BoW, Embedding và LLM trong cách biểu diễn và xử lý ngôn ngữ