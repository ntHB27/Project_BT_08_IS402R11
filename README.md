# Project_BT_08_IS402R11
# Fed-LCCF: Privacy-Preserving Federated Recommendation System with Leiden Community Clustering

> **Đồ án môn học**: Điện toán đám mây
> **Thành viên**: Nguyễn Nhật Minh, Nguyễn Trần Hoài Bảo, Nguyễn Minh Đức, Bùi Đình Trung Kiên
> **Tài liệu tham chiếu**: _Leveraging Leiden communities for enhanced collaborative filtering with matrix factorization techniques_ (Scientific Reports, Nature Portfolio, 2026).

---

## 1. Giới Thiệu Đề Tài

Hệ thống **Fed-LCCF** mô phỏng bài toán gợi ý phim phân tán cho chuỗi rạp chiếu phim (Cinema Chains). Thay vì gom toàn bộ dữ liệu đánh giá thô của khách hàng về máy chủ trung tâm (gây rủi ro lộ lọt quyền riêng tư), hệ thống kết hợp **Federated Learning (FedAvg)** và giải thuật phân cụm đồ thị **Leiden (Leiden Community Clustering)** để giải quyết 3 thách thức:

1. **Bảo vệ quyền riêng tư (Privacy-Preserving)**: Ma trận đặc trưng người dùng ($U_k$) được giữ bí mật 100% tại bộ nhớ máy trạm (Client). Chỉ có ma trận đặc trưng bộ phim ($V_k$) và số lượt tương tác được gửi lên Central Server.
2. **Mất cân bằng dữ liệu (Non-IID Data)**: Phân mảnh dữ liệu không đồng đều giữa các chi nhánh rạp (45%, 30%, 18%, 7%) với cơ chế phạt **L2 Regularization thích ứng** ($\lambda_k$) chống học vẹt trên rạp ít dữ liệu.
3. **Khởi tạo lạnh & Dữ liệu thưa (Cold-Start & Incompleteness)**: Sử dụng tâm cụm cộng đồng Leiden ($\mu_c$) kết hợp cơ chế **Community Shrinkage** ($\gamma=10.0$) để kéo các bộ phim ít người xem về đặc trưng chung của cụm phim cùng gu.

---

## 2. Kiến Trúc Hệ Thống 3 Giai Đoạn

```
                  +-----------------------------------+
                  |   FastAPI Central Server (:8000)  |
                  |   - Async Round Barrier (4/4)     |
                  |   - Leiden k-NN Graph Modularity  |
                  |   - Community Shrinkage Averaging |
                  +-----------------+-----------------+
                                    ^
       +----------------------------+----------------------------+
       | HTTP POST /upload_factors  | HTTP GET /get_global_factors
       v                            v                            v
+--------------+             +--------------+             +--------------+
|   Client 1   |             |   Client 2   |             |   Client 4   |
| (45% Ratings)|             | (30% Ratings)|             | (7% Ratings) |
| client_1.db  |             | client_2.db  |             | client_4.db  |
| Local MF SGD |             | Local MF SGD |             | Cold-Start   |
+--------------+             +--------------+             +--------------+
```

- **Giai đoạn 1 — Warm-up Fed-MF (Vòng 1 – 5)**: Server phát ma trận khởi tạo $V^{(0)}$. 4 Client tự huấn luyện cục bộ bằng giải thuật SGD kèm adaptive L2 regularization. Server tính trung bình có trọng số Vanilla FedAvg để đồng bộ hóa không gian ẩn.
- **Giai đoạn 2 — Phân cụm Leiden (Chạy 1 lần cuối Vòng 5)**: Server tính độ tương đồng Cosine giữa 1,682 phim từ ma trận $V^{(5)}$, dựng đồ thị $k$-NN ($k=20$) và chạy thuật toán Leiden (`leidenalg`) chia các phim thành 5 cộng đồng gắn kết cao, phát bảng `community_map` xuống các Client.
- **Giai đoạn 3 — Community Fed-LCCF (Vòng 6 – 20)**: Client neo giữ các phim chưa có đánh giá (cold-start) về tâm cụm $\mu_c$. Server áp dụng công thức **Community Shrinkage**:
  $$\bar{v}_i = \frac{N_i}{N_i + \gamma} \cdot (\text{Trung bình các rạp}) + \frac{\gamma}{N_i + \gamma} \cdot \mu_c$$

---

## 3. Cấu Trúc Mã Nguồn

```
FedLCCF/
├── 
│   ├── data/                   # [Dev 1] Pipeline tiền xử lý và 4 database SQLite
│   │   ├── preprocess_data.py   # Phân chia 80/20 và phân mảnh ngang Non-IID
│   │   └── client_data_loader.py# Data Access Layer nạp ma trận thưa CSR
│   ├── client_ml/              # [Dev 2] Huấn luyện máy học cục bộ phía Client
│   │   ├── local_mf.py         # Matrix Factorization SGD + Adaptive L2
│   │   └── client_agent.py     # Client Agent giao tiếp HTTP REST
│   ├── cloud_server/           # [Dev 3] Máy chủ trung tâm FastAPI & Thuật toán Leiden
│   │   ├── server.py           # REST API endpoints & Swagger UI
│   │   ├── barrier.py          # Rào cản đồng bộ đa tiến trình
│   │   ├── aggregator.py       # FedAvg & Community Shrinkage
│   │   └── leiden_engine.py    # Dựng đồ thị k-NN và phân cụm Leiden
│   ├── orchestrator/           # [Dev 4] Điều phối mô phỏng & Đánh giá Benchmark
│   │   ├── run_simulation.py   # Script điều phối đa tiến trình tự động
│   │   ├── evaluator.py        # Đo sai số RMSE / MAE trên tập test
│   │   └── benchmark.py        # Chạy đối chuẩn 3 kịch bản qua 20 rounds
│   ├── docs/                   # [Dev 5] Tài liệu kỹ thuật chi tiết
│   │   ├── local_execution.md  # Hướng dẫn chạy Python venv & Docker Desktop
│   │   └── azure_vm_deployment.md # Hướng dẫn triển khai Azure Ubuntu VM
│   ├── docker-compose.yml      # Cấu hình container hóa 1 Server + 4 Clients
│   ├── Dockerfile              # Container base Python 3.12-slim
│   ├── requirements.txt        # Danh mục thư viện phụ thuộc
│   ├── benchmark_results.csv   # Kết quả thực nghiệm đối chuẩn (60 dòng)
│   └── convergence_plot.png    # Biểu đồ đường cong hội tụ chuẩn xuất bản 300 DPI
└── README.md
```

---

## 4. Kết Quả Thực Nghiệm Đối Chuẩn (Round 20)

| Kịch bản thực nghiệm      | Client 1 (45%)<br>RMSE (MAE) | Client 2 (30%)<br>RMSE (MAE) | Client 3 (18%)<br>RMSE (MAE) | Client 4 (7% Cold-start)<br>RMSE (MAE) | Global Test Set<br>RMSE (MAE) |
| :------------------------ | :--------------------------: | :--------------------------: | :--------------------------: | :------------------------------------: | :---------------------------: |
| **1. Local Only**         |       0.9215 (0.7174)        |       0.9589 (0.7511)        |       1.0404 (0.8159)        |            1.0649 (0.8282)             |        0.9655 (0.7530)        |
| **2. Vanilla FedAvg**     |       0.9037 (0.7038)        |   **0.9311** (**0.7343**)    |       0.9845 (0.7751)        |        **0.9508** (**0.7490**)         |    **0.9302** (**0.7289**)    |
| **3. Fed-LCCF (Đề xuất)** |     **0.9035** (0.7091)      |       0.9356 (0.7410)        |     **0.9798** (0.7770)      |            0.9515 (0.7511)             |        0.9306 (0.7338)        |

> **Nhận xét**: Học liên kết giúp giảm mạnh sai số tại rạp ít dữ liệu (Client 4 giảm từ 1.065 xuống ~0.951). Cơ chế Leiden giúp cải thiện độ chính xác trên Client 3 và tăng tính ổn định neo giữ đặc trưng phim quanh các tâm cụm.

---

## 5. Hướng Dẫn Chạy Nhanh

### Cách 1: Chạy mô phỏng tự động bằng Python (Khuyên dùng)

```powershell
# Di chuyển vào thư mục dự án
cd FedLCCF

# Kích hoạt môi trường ảo và chạy thử nghiệm 5 rounds
.\.venv\Scripts\python.exe -m orchestrator.run_simulation --rounds 5 --port 8000 --clean-logs

# Hoặc chạy đầy đủ 20 rounds
.\.venv\Scripts\python.exe -m orchestrator.run_simulation --rounds 20 --port 8000 --clean-logs
```

### Cách 2: Chạy bằng Docker Desktop

```bash
cd fedlccf_prototype
docker compose --profile full up --build
```

### Cách 3: Triển khai Central Server lên Azure VM

Xem hướng dẫn chi tiết từng bước tại: `docs/azure_vm_deployment.md`.
