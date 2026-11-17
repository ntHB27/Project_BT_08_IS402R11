# Hướng Dẫn Vận Hành Fed-LCCF Cục Bộ (Local Execution Guide)

Tài liệu này cung cấp hướng dẫn toàn diện, từng bước để cài đặt, chuẩn bị dữ liệu, chạy mô phỏng học máy liên hiệp (Federated Learning) đa tiến trình và container hóa bằng Docker Desktop trên máy tính cá nhân (hỗ trợ Windows, Linux và macOS).

---

## Mục Lục
1. [Tổng Quan Kiến Trúc Cục Bộ](#1-tổng-quan-kiến-trúc-cục-bộ)
2. [Yêu Cầu Hệ Thống & Phần Mềm](#2-yêu-cầu-hệ-thống--phần-mềm)
3. [Thiết Lập Môi Trường Ảo Python (.venv)](#3-thiết-lập-môi-trường-ảo-python-venv)
4. [Bước 1: Tiền Xử Lý Dữ Liệu & Phân Hoạch SQLite](#4-bước-1-tiền-xử-lý-dữ-liệu--phân-hoạch-sqlite)
5. [Bước 2: Chạy Mô Phỏng Đa Tiến Trình Tự Động (Orchestrator)](#5-bước-2-chạy-mô-phỏng-đa-tiến-trình-tự-động-orchestrator)
6. [Bước 3: Chạy Mô Phỏng Bằng Docker Desktop & Docker Compose](#6-bước-3-chạy-mô-phỏng-bằng-docker-desktop--docker-compose)
7. [Kiểm Tra Logs & Giám Sát Trạng Thái Hệ Thống](#7-kiểm-tra-logs--giám-sát-trạng-thái-hệ-thống)
8. [Khắc Phục Sự Cố Thường Gặp (Troubleshooting)](#8-khắc-phục-sự-cố-thường-gặp-troubleshooting)

---

## 1. Tổng Quan Kiến Trúc Cục Bộ

Hệ thống **Fed-LCCF (Federated Leiden Community Collaborative Filtering)** mô phỏng môi trường học liên hiệp bao gồm 5 nút logic chạy song song trên một máy tính cá nhân:
- **Central Aggregation Server (FastAPI)**: Đóng vai trò máy chủ trung tâm điều phối, quản lý rào cản đồng bộ (Round Barrier), gom cụm cộng đồng vật phẩm qua thuật toán Leiden (`leidenalg`) sau Round 5, và thực hiện co rút cộng đồng (Community Shrinkage) ở Phase 3 (Rounds 6–20).
- **4 Federated Client Workers (Client Agents)**: Đại diện cho 4 client phân tán, huấn luyện mô hình phân tích ma trận Matrix Factorization ($R \approx U \cdot V^T$) cục bộ bằng giải thuật SGD và L2 regularization.
- **Bảo Vệ Quyền Riêng Tư (Privacy-Preserving)**: Ma trận ẩn của người dùng $U_k$ (kích thước $N_k \times d$) được lưu trữ độc quyền trong bộ nhớ của client và **không bao giờ truyền qua mạng**. Client chỉ gửi ma trận đặc trưng vật phẩm $V_k$ ($1682 \times d$) và vector số lượt tương tác $n_k$ lên server.

```
                  +-----------------------------------+
                  |   FastAPI Central Server (:8000)  |
                  |   - Round Barrier Synchronizer    |
                  |   - Leiden Graph Modularity       |
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

---

## 2. Yêu Cầu Hệ Thống & Phần Mềm

### 2.1. Cấu hình phần cứng tối thiểu
- **CPU**: Tối thiểu 2 nhân (Khuyến nghị 4 nhân thực trở lên để chạy đồng thời 5 tiến trình).
- **RAM**: Tối thiểu 4 GB (Khuyến nghị 8 GB trở lên).
- **Ổ cứng**: Còn trống tối thiểu 2 GB (dành cho Python packages, dataset MovieLens 100K và Docker images).

### 2.2. Phần mềm yêu cầu
- **Hệ điều hành**:
  - Windows 10/11 (64-bit) với PowerShell 5.1+ hoặc Windows Terminal.
  - Linux (Ubuntu 20.04/22.04 LTS, Debian 11/12, Fedora, v.v.).
  - macOS 12 Monterey trở lên (hỗ trợ cả Apple Silicon M1/M2/M3 và Intel x86_64).
- **Python**: Phiên bản **Python 3.12** (hoặc 3.10+).
- **Git**: Dùng để quản lý mã nguồn.
- **Docker Desktop** (Tùy chọn): Nếu bạn muốn chạy hệ thống dạng container hóa, hãy cài đặt Docker Desktop và kích hoạt tính năng Docker Compose v2.

---

## 3. Thiết Lập Môi Trường Ảo Python (.venv)

Khuyến nghị khởi tạo một môi trường ảo độc lập nhằm tránh xung đột thư viện với hệ điều hành.

### 3.1. Tạo và kích hoạt môi trường ảo

Mở terminal tại thư mục gốc của dự án (`d:\FedLCCF` hoặc tương đương):

#### Trên Windows (PowerShell):
```powershell
# Di chuyển vào thư mục gốc của repository
cd d:\FedLCCF

# Tạo môi trường ảo .venv sử dụng Python 3.12
python -m venv .venv

# Nếu gặp lỗi ExecutionPolicy trên PowerShell, chạy lệnh sau:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1
```

*(Nếu sử dụng Command Prompt - `cmd.exe`, kích hoạt bằng: `.\.venv\Scripts\activate.bat`)*

#### Trên Linux / macOS (Bash / Zsh):
```bash
# Di chuyển vào thư mục gốc
cd /path/to/FedLCCF

# Tạo môi trường ảo
python3 -m venv .venv

# Kích hoạt môi trường ảo
source .venv/bin/activate
```

Khi kích hoạt thành công, dấu nhắc lệnh terminal sẽ có tiền tố `(.venv)`.

### 3.2. Cài đặt các thư viện phụ thuộc

Cài đặt toàn bộ danh mục thư viện được ghim phiên bản chuẩn xác từ `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r fedlccf_prototype/requirements.txt
```

Các thư viện chính bao gồm:
- `fastapi==0.162.0`, `uvicorn[standard]==0.54.0`: Nền tảng máy chủ ASGI tốc độ cao.
- `numpy==2.5.3`, `scipy==1.18.1`: Tính toán đại số tuyến tính ma trận và tối ưu hóa phân hoạch MILP.
- `igraph==1.0.0`, `leidenalg==0.12.0`: Xây dựng đồ thị k-NN vật phẩm và thuật toán phân cụm Leiden tối ưu hóa modularity.
- `psutil==7.2.2`: Quản lý cây tiến trình con và dọn dẹp tiến trình an toàn khi tắt ứng dụng.
- `scikit-learn==1.9.1`, `matplotlib==3.11.2`: Đánh giá sai số RMSE/MAE và vẽ biểu đồ hội tụ.

---

## 4. Bước 1: Tiền Xử Lý Dữ Liệu & Phân Hoạch SQLite

Hệ thống sử dụng tập dữ liệu chuẩn **MovieLens 100K** gồm 100,000 lượt đánh giá từ 943 người dùng cho 1,682 bộ phim.

### 4.1. Lệnh thực thi tiền xử lý
Chạy module tiền xử lý dữ liệu từ thư mục gốc của repository:

```bash
python fedlccf_prototype/data/preprocess_data.py
```

Nếu tập dữ liệu thô `ml-100k` nằm ở thư mục tùy chỉnh, bạn có thể truyền cờ chỉ định:
```bash
python fedlccf_prototype/data/preprocess_data.py --raw-data-dir ./ml-100k --output-dir ./fedlccf_prototype/data --seed 42
```

### 4.2. Giải thích cơ chế phân hoạch dữ liệu
Quy trình tiền xử lý thực hiện các bước tuần tự và chặt chẽ:
1. **0-Based Indexing**: Ánh xạ lại toàn bộ `user_id` thành chỉ số liên tục từ `0` đến `942` và `item_id` từ `0` đến `1681`.
2. **Stratified Split 80/20**: Phân chia phân tầng theo từng người dùng để đảm bảo mọi người dùng đều có 80% số đánh giá trong tập train (tổng cộng 80,000 ratings) và 20% trong tập test toàn cục (20,000 ratings).
3. **Phân hoạch Non-IID giữa 4 Clients**:
   - Sử dụng giải thuật quy hoạch tuyến tính nguyên hỗn hợp (MILP) với `scipy.optimize.milp` để phân bổ tập người dùng không giao thoa (User-disjoint), đảm bảo phân bổ chính xác tỷ lệ đánh giá:
     * **Client 1 (45%)**: 36,000 ratings — Mô phỏng client quy mô lớn, tương tác nhiều thể loại.
     * **Client 2 (30%)**: 24,000 ratings — Mô phỏng client trung bình khá.
     * **Client 3 (18%)**: 14,400 ratings — Mô phỏng client cỡ nhỏ.
     * **Client 4 (7%)**: 5,600 ratings — Mô phỏng client thưa thớt (Sparse/Cold-Start), thiếu hụt khoảng 50% số lượng vật phẩm trong tập train cục bộ.
4. **Khởi tạo cơ sở dữ liệu SQLite**:
   - Mỗi client sở hữu riêng một cơ sở dữ liệu độc lập: `client_1.db`, `client_2.db`, `client_3.db`, `client_4.db`.
   - Tập kiểm thử toàn cục được ghi ra `test.db` và `test.csv`.
   - Mỗi tệp SQLite được tối ưu hóa chế độ `WAL` (Write-Ahead Logging), `synchronous=NORMAL` và tạo 2 chỉ mục tìm kiếm: `idx_ratings_user` trên cột `user_id` và `idx_ratings_item` trên cột `item_id`.
   - Tệp từ điển metadata phim được xuất ra `item_mapping.json`.

---

## 5. Bước 2: Chạy Mô Phỏng Đa Tiến Trình Tự Động (Orchestrator)

Bộ điều phối `run_simulation.py` tự động hóa toàn bộ vòng đời huấn luyện liên hiệp trên một máy duy nhất:
- Kiểm tra tính khả dụng của cổng mạng (Port availability).
- Khởi chạy tiến trình nền FastAPI Server.
- Thăm dò endpoint `/health` với cơ chế timeout cho đến khi server sẵn sàng.
- Khởi chạy đồng thời 4 tiến trình Client Agents làm việc song song.
- Bắt tín hiệu ngắt (`Ctrl+C`, `SIGINT`, `SIGTERM`) để dọn dẹp sạch sẽ toàn bộ cây tiến trình (process tree) bằng `psutil`.

### 5.1. Chạy thử nghiệm nhanh (Quick Smoke Test - 5 Rounds)
Để kiểm tra tính thông suốt của đường truyền dữ liệu và rào cản đồng bộ trong vòng chưa đầy 1 phút:

```bash
python -m fedlccf_prototype.orchestrator.run_simulation --rounds 5 --clients 4
```

### 5.2. Chạy thử nghiệm đầy đủ (Full 20 Rounds)
Chạy đầy đủ cả 3 giai đoạn của thuật toán Fed-LCCF:

```bash
python -m fedlccf_prototype.orchestrator.run_simulation --rounds 20 --clients 4
```

### 5.3. Bảng giải thích chi tiết các tham số CLI

| Tham số | Giá trị mặc định | Mô tả chi tiết |
| :--- | :---: | :--- |
| `--rounds` | `20` | Tổng số vòng học liên hiệp (Federated Rounds). Vòng 1–5 là Vanilla FedAvg; cuối vòng 5 kích hoạt thuật toán Leiden; vòng 6–20 áp dụng Community Shrinkage. |
| `--clients` | `4` | Số lượng client worker đồng thời tham gia vòng học (từ 1 đến 4). |
| `--host` | `127.0.0.1` | Địa chỉ IP giao diện mạng máy chủ FastAPI lắng nghe. |
| `--port` | `8000` | Cổng TCP máy chủ FastAPI lắng nghe. |
| `--mode` | `full` | Chế độ chạy: <br>• `full`: Khởi chạy cả Server và 4 Clients.<br>• `server-only`: Chỉ khởi chạy Server.<br>• `clients-only`: Chỉ chạy 4 Clients (kết nối vào Server có sẵn).<br>• `dry-run`: Kiểm tra cấu hình và tệp SQLite mà không chạy huấn luyện. |
| `--server-url` | `None` | URL máy chủ tùy chọn cho client (Ví dụ: `http://127.0.0.1:8000`). Nếu không truyền, orchestrator tự động ghép từ `--host` và `--port`. |
| `--gamma` | `10.0` | Tham số co rút trọng số cộng đồng ($\gamma$) ở Phase 3. Giá trị $\gamma$ càng lớn kéo các vật phẩm ít tương tác về tâm cộng đồng mạnh hơn. |
| `--epochs` | `5` | Số epoch huấn luyện giải thuật SGD cục bộ trên mỗi client trong mỗi vòng. |
| `--lr` | `0.01` | Tốc độ học (Learning Rate) của thuật toán SGD trên client. |
| `--latent-dim` | `32` | Số chiều ẩn ($d$) của ma trận đặc trưng người dùng và vật phẩm. |
| `--clean-logs` | `False` | Tự động dọn sạch các file log cũ trong `orchestrator/logs/` trước khi bắt đầu phiên chạy mới. |
| `--verbose`, `-v` | `False` | Bật chế độ hiển thị log chi tiết (DEBUG level) ra màn hình console. |

---

## 6. Bước 3: Chạy Mô Phỏng Bằng Docker Desktop & Docker Compose

Docker Compose cho phép cô lập hoàn toàn môi trường chạy vào các container độc lập, đảm bảo tính tái lập 100% trên mọi hệ điều hành mà không phụ thuộc vào phiên bản Python của máy chủ.

### 6.1. Khởi động Docker Desktop
Đảm bảo biểu tượng cá voi Docker Desktop trên thanh tác vụ hiển thị trạng thái màu xanh lá cây (**Engine running**).

### 6.2. Xây dựng Docker Image
Tại thư mục `fedlccf_prototype/` (nơi chứa file `docker-compose.yml`):

```bash
cd fedlccf_prototype

# Xây dựng image fedlccf:latest
docker compose build
```

### 6.3. Khởi chạy toàn bộ mô phỏng (Server + 4 Clients)
Chạy toàn bộ 5 containers mô phỏng liên hiệp thông qua Docker Compose profile `full`:

```bash
docker compose --profile full up --build
```

Khi chạy lệnh này:
1. Container `fedlccf-server` sẽ được khởi chạy trước.
2. Bộ kiểm tra sức khỏe `healthcheck` trên server sẽ thăm dò định kỳ cổng `8000/health`.
3. Khi server chuyển sang trạng thái `healthy`, 4 container client (`fedlccf-client-1` đến `fedlccf-client-4`) sẽ tự động kích hoạt nhờ thiết lập `depends_on: { condition: service_healthy }`.
4. Toàn bộ logs từ cả 5 containers sẽ được stream trực tiếp lên cửa sổ terminal với các nhãn màu khác nhau.

### 6.4. Chạy riêng lẻ máy chủ Central Server bằng Docker
Nếu bạn muốn máy chủ chạy độc lập trong container và chạy các client từ bên ngoài:

```bash
# Khởi chạy server ở chế độ daemon nền (background)
docker compose up -d server

# Kiểm tra trạng thái container
docker compose ps

# Xem log thời gian thực của server
docker compose logs -f server
```

Sau đó, bạn có thể chạy các client trên máy chủ vật lý bằng lệnh:
```bash
python -m fedlccf_prototype.client_ml.client_agent --client-id 1 --db-path fedlccf_prototype/data/client_1.db --server-url http://127.0.0.1:8000 --rounds 5
```

### 6.5. Dừng và dọn dẹp Containers
Sau khi hoàn thành phiên huấn luyện, giải phóng tài nguyên container và mạng ảo:

```bash
docker compose --profile full down
```

---

## 7. Kiểm Tra Logs & Giám Sát Trạng Thái Hệ Thống

### 7.1. Cấu trúc thư mục Logs
Tất cả các tệp nhật ký thực thi được ghi nhận tự động vào thư mục:
`fedlccf_prototype/orchestrator/logs/`

- `orchestrator.log`: Tiến trình điều phối tổng thể, thời gian chờ khởi động server, tiến độ rào cản barrier từng vòng, thời gian hoàn thành.
- `server.log`: Toàn bộ các yêu cầu HTTP API của FastAPI, chi tiết quá trình tổng hợp trọng số, quá trình gom cụm Leiden sau Round 5 (số lượng cụm phát hiện, độ đo modularity, ma trận khoảng cách cosine).
- `client_1.log` đến `client_4.log`: Chi tiết quá trình huấn luyện SGD cục bộ, hàm mất mát (loss) sau mỗi epoch, thời gian upload và tải xuống ma trận đặc trưng.

Xem log thời gian thực trên Windows PowerShell:
```powershell
Get-Content -Path .\fedlccf_prototype\orchestrator\logs\server.log -Wait -Tail 30
```

Xem log trên Linux / macOS:
```bash
tail -f fedlccf_prototype/orchestrator/logs/server.log
```

### 7.2. Thăm dò trạng thái trực tiếp qua REST API (curl)
Trong khi hệ thống đang vận hành, bạn có thể mở một cửa sổ terminal khác để truy vấn các endpoint giám sát:

#### 1. Kiểm tra trạng thái hoạt động (Health Probe):
```bash
curl -s http://127.0.0.1:8000/health
```
*Dữ liệu phản hồi mẫu:*
```json
{
  "status": "healthy",
  "app": "Fed-LCCF Central Aggregation Server",
  "version": "1.0.0",
  "current_round": 1,
  "phase": 1,
  "leiden_communities_ready": false,
  "uptime_seconds": 12.4
}
```

#### 2. Kiểm tra tiến độ đồng bộ rào cản (Round Barrier Status):
```bash
curl -s http://127.0.0.1:8000/api/v1/fl/round_status
```
*Dữ liệu phản hồi mẫu:*
```json
{
  "target_round": 1,
  "expected_clients": 4,
  "arrived_clients": [1, 2, 3],
  "arrived_count": 3,
  "is_barrier_released": false
}
```

#### 3. Kiểm tra ma trận trọng số toàn cục đã tổng hợp (Global Factors):
```bash
curl -s http://127.0.0.1:8000/api/v1/fl/get_global_factors/1 | head -c 200
```

---

## 8. Khắc Phục Sự Cố Thường Gặp (Troubleshooting)

### 8.1. Lỗi xung đột cổng 8000 (`Address already in use` hoặc `Errno 10048`)
**Hiện tượng**: Khi khởi động server, bạn gặp thông báo lỗi:
`OSError: [WinError 10048] Only one usage of each socket address is normally permitted` hoặc `Address already in use`.

**Nguyên nhân**: Một tiến trình server cũ chưa được giải phóng hoặc có dịch vụ khác đang chiếm giữ cổng 8000.

**Cách khắc phục**:
- **Cách 1: Đổi cổng chạy linh hoạt**:
  Truyền tham số `--port` sang cổng khác (Ví dụ 8080):
  ```bash
  python -m fedlccf_prototype.orchestrator.run_simulation --rounds 5 --port 8080
  ```
- **Cách 2: Tìm và tắt tiến trình đang chiếm cổng trên Windows**:
  ```powershell
  # Tìm PID của tiến trình đang lắng nghe cổng 8000
  netstat -ano | findstr :8000

  # Buộc dừng tiến trình theo PID tìm thấy (ví dụ PID là 14520)
  taskkill /PID 14520 /F
  ```
- **Cách 3: Tìm và tắt tiến trình trên Linux/macOS**:
  ```bash
  # Tắt ngay tiến trình đang lắng nghe cổng 8000
  fuser -k 8000/tcp
  # hoặc:
  lsof -ti :8000 | xargs kill -9
  ```

### 8.2. Lỗi tiến trình mồ côi (Orphan Processes) sau khi dừng đột ngột
**Hiện tượng**: Bạn nhấn `Ctrl+C` nhưng một số tiến trình con của client hoặc uvicorn vẫn chạy ngầm tiêu tốn CPU.

**Khắc phục**:
Bộ điều phối `run_simulation.py` đã tích hợp cơ chế bắt tín hiệu ngắt và dọn dẹp bằng `psutil.Process.children(recursive=True)` cùng lệnh `taskkill /F /T`. Tuy nhiên, nếu bạn ngắt bằng `Kill Terminal`, có thể dọn dẹp thủ công:
- **Trên Windows**:
  ```powershell
  # Tắt toàn bộ tiến trình python đang chạy trong thư mục prototype nếu cần
  Get-Process python | Where-Object { $_.CommandLine -like "*fedlccf*" } | Stop-Process -Force
  ```
- **Trên Linux/macOS**:
  ```bash
  pkill -f "fedlccf_prototype"
  ```

### 8.3. Lỗi quyền thực thi Script trên Windows PowerShell (`ExecutionPolicy`)
**Hiện tượng**: Khi kích hoạt `.venv\Scripts\Activate.ps1`, PowerShell báo lỗi:
`File ... cannot be loaded because running scripts is disabled on this system.`

**Khắc phục**:
Chạy lệnh phân quyền tạm thời cho phiên làm việc hiện tại:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 8.4. Lỗi kết nối Docker Daemon (`Cannot connect to the Docker daemon`)
**Hiện tượng**: Chạy `docker compose up` báo lỗi:
`error during connect: This error may indicate that the docker daemon is not running.`

**Khắc phục**:
1. Mở phần mềm **Docker Desktop** trên Windows/macOS và đợi ứng dụng chuyển sang trạng thái `Running`.
2. Trên Linux, khởi động dịch vụ docker:
   ```bash
   sudo systemctl start docker
   sudo systemctl status docker
   ```
3. Đảm bảo tài khoản của bạn đã được thêm vào nhóm docker:
   ```bash
   sudo usermod -aG docker $USER
   ```

### 8.5. Lỗi thiếu tệp SQLite (`FileNotFoundError: client_1.db`)
**Hiện tượng**: Client báo lỗi không tìm thấy tệp cơ sở dữ liệu `client_1.db`.

**Khắc phục**:
Đảm bảo bạn đã thực hiện **Bước 1** tiền xử lý dữ liệu trước khi chạy mô phỏng:
```bash
python fedlccf_prototype/data/preprocess_data.py
```
Kiểm tra thư mục `fedlccf_prototype/data/` đảm bảo tồn tại đủ 4 tệp `client_1.db` đến `client_4.db` và `test.db`.
