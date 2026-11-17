# Hướng Dẫn Triển Khai Fed-LCCF Trên Azure Virtual Machine (Ubuntu 22.04 LTS)

Tài liệu này cung cấp hướng dẫn chi tiết, từng bước từ thiết lập hạ tầng đám mây Microsoft Azure, cấu hình máy chủ ảo Ubuntu Server 22.04 LTS, triển khai Central Aggregation Server chạy vĩnh viễn dưới dạng `systemd` service (hoặc Docker), cho đến việc kết nối các Client học liên hiệp từ xa qua mạng Internet công cộng với các chuẩn mực bảo mật cao cấp.

---

## Mục Lục
1. [Tổng Quan Kiến Trúc Triển Khai Phân Tán](#1-tổng-quan-kiến-trúc-triển-khai-phân-tán)
2. [Bước 1: Khởi Tạo Azure Virtual Machine](#2-bước-1-khởi-tạo-azure-virtual-machine)
3. [Bước 2: Cấu Hình Network Security Group (NSG)](#3-bước-2-cấu-hình-network-security-group-nsg)
4. [Bước 3: Cài Đặt Môi Trường Máy Chủ (VM Provisioning)](#4-bước-3-cài-đặt-môi-trường-máy-chủ-vm-provisioning)
5. [Bước 4: Triển Khai Central Server Dưới Dạng Systemd Service](#5-bước-4-triển-khai-central-server-dưới-dạng-systemd-service)
6. [Bước 5: Cấu Hình và Kết Nối Client Từ Xa (Remote Client Agents)](#6-bước-5-cấu-hình-và-kết-nối-client-từ-xa-remote-client-agents)
7. [Bước 6: Tùy Chọn Triển Khai Bằng Docker Compose Trên Azure VM](#7-bước-6-tùy-chọn-triển-khai-bằng-docker-compose-trên-azure-vm)
8. [Bước 7: Khuyến Nghị Bảo Mật Nâng Cao (UFW & Nginx SSL/TLS)](#8-bước-7-khuyến-nghị-bảo-mật-nâng-cao-ufw--nginx-ssltls)

---

## 1. Tổng Quan Kiến Trúc Triển Khai Phân Tán

Trong kịch bản triển khai phân tán thực tế của hệ thống **Fed-LCCF**:
- **Central Aggregation Server** được đặt trên máy chủ ảo Azure (Azure Virtual Machine) với địa chỉ IP tĩnh công khai (Public IP). Máy chủ này giữ vai trò:
  - Tiếp nhận các ma trận đặc trưng vật phẩm $V_k$ ($1682 \times 32$) và vector tần suất $n_k$ từ các client qua giao thức HTTP/HTTPS.
  - Quản lý cơ chế đồng bộ rào cản vòng học (Round Barrier Synchronization) đảm bảo đủ 4 client trước khi tổng hợp.
  - Thực thi gom cụm đồ thị Leiden phân tích cộng đồng vật phẩm sau Round 5 và tính toán tâm cụm $\mu_c$.
  - Phân phối trọng số toàn cục đã co rút (Community Shrinkage) về cho các client.
- **Client Agents** chạy phân tán trên máy trạm của các thành viên trong nhóm nghiên cứu hoặc các máy ảo biên (Edge Devices).
- **Nguyên Tắc Bảo Mật Quyền Riêng Tư (Strict Privacy Guarantee)**:
  - Ma trận đặc trưng người dùng $U_k$ (User Latent Matrix) cùng toàn bộ lịch sử đánh giá cá nhân lưu trong tệp SQLite cục bộ (`client_k.db`) được giữ an toàn tuyệt đối trên máy trạm của client.
  - Không có bất kỳ thông tin nhận dạng người dùng nào bị gửi lên Azure VM.

```
+--------------------------------------------------------------------------+
|                     MICROSOFT AZURE CLOUD (Ubuntu 22.04)                 |
|                                                                          |
|   +------------------------------------------------------------------+   |
|   | Azure Network Security Group (NSG)                               |   |
|   |  - Port 22: SSH (Chỉ cho phép IP Admin)                          |   |
|   |  - Port 8000: FastAPI Server (Hoặc Port 443 Nginx HTTPS Reverse) |   |
|   +---------------------------------+--------------------------------+   |
|                                     |                                    |
|   +---------------------------------v--------------------------------+   |
|   | Fed-LCCF Aggregator Service (systemd: fedlccf-server.service)    |   |
|   |  - Host: 0.0.0.0 | Port: 8000 | Expected Clients: 4 | Gamma: 10.0|   |
|   |  - REST API: /health, /upload_factors, /get_global_factors       |   |
|   +------------------------------------------------------------------+   |
+-------------------------------------^------------------------------------+
                                      |
                 INTERNET (HTTP / HTTPS / REST API)
                                      |
        +-----------------------------+-----------------------------+
        |                             |                             |
+-------v-------+             +-------v-------+             +-------v-------+
| Local Client 1|             | Local Client 2|             | Local Client 4|
| Máy trạm 1    |             | Máy trạm 2    |             | Máy trạm 4    |
| client_1.db   |             | client_2.db   |             | client_4.db   |
| Local MF SGD  |             | Local MF SGD  |             | Cold-Start MF |
+---------------+             +---------------+             +---------------+
```

---

## 2. Bước 1: Khởi Tạo Azure Virtual Machine

Bạn có thể tạo Azure VM thông qua giao diện đồ họa **Azure Portal** hoặc bằng dòng lệnh **Azure CLI**.

### 2.1. Cấu hình khuyến nghị
- **Subscription**: Gói thuê bao Azure của bạn (Ví dụ: Azure for Students, Pay-As-You-Go).
- **Resource Group**: `rg-fedlccf`
- **Region**: `Southeast Asia` (Singapore) hoặc `East Asia` (Hong Kong) nhằm đạt độ trễ mạng thấp nhất từ Việt Nam.
- **Operating System Image**: `Ubuntu Server 22.04 LTS - x64 Gen2`.
- **VM Sizing (Kích thước phần cứng)**:
  - **Khuyến nghị chuẩn**: `Standard_B2s` (2 vCPUs, 4 GiB RAM, ~0.04 USD/giờ) — Tối ưu chi phí cho các đợt huấn luyện đồ thị và phục vụ API liên tục.
  - **Hiệu năng cao**: `Standard_D2s_v5` (2 vCPUs, 8 GiB RAM) — Đáp ứng nhu cầu tính toán ma trận tương đồng cosine lớn và nhiều vòng học.
- **Authentication type**: `SSH public key` (Khuyến nghị sử dụng cặp khóa SSH để đảm bảo an ninh thay vì mật khẩu).
- **Username**: `azureuser`

### 2.2. Tạo cặp khóa SSH trên máy cá nhân
Nếu chưa có khóa SSH, mở terminal trên máy tính của bạn và khởi tạo:

```bash
# Tạo cặp khóa SSH Ed25519 hoặc RSA 4096-bit
ssh-keygen -t rsa -b 4096 -f ~/.ssh/azure_fedlccf -C "fedlccf-admin"
```
*(Lưu ý: Giữ bí mật tệp khóa riêng tư `~/.ssh/azure_fedlccf`, chỉ tải nội dung tệp công khai `~/.ssh/azure_fedlccf.pub` lên Azure).*

### 2.3. Khởi tạo VM nhanh bằng Azure CLI (Tùy chọn)
Nếu bạn đã cài đặt công cụ `az cli`:

```bash
# 1. Đăng nhập Azure
az login

# 2. Tạo Resource Group
az group create --name rg-fedlccf --location southeastasia

# 3. Tạo Virtual Machine
az vm create \
  --resource-group rg-fedlccf \
  --name vm-fedlccf-server \
  --image Ubuntu2204 \
  --size Standard_B2s \
  --admin-username azureuser \
  --ssh-key-values ~/.ssh/azure_fedlccf.pub \
  --public-ip-sku Standard
```

Lưu lại địa chỉ **Public IP** xuất hiện trong kết quả đầu ra (Ví dụ: `20.198.110.45`).

---

## 3. Bước 2: Cấu Hình Network Security Group (NSG)

Mặc định, Azure đóng toàn bộ các cổng mạng ngoại trừ cổng 22. Cần cấu hình tường lửa NSG để mở cổng dịch vụ cho hệ thống liên hiệp.

### 3.1. Thiết lập Inbound Security Rules trong Azure Portal
1. Truy cập vào **Azure Portal** -> Chọn máy ảo `vm-fedlccf-server`.
2. Trong menu bên trái, mục **Settings**, chọn **Networking** (hoặc **Network settings**).
3. Nhấp nút **Add inbound security rule** để thêm 2 quy tắc sau:

#### Quy tắc 1: Cổng SSH (Quản trị từ xa)
- **Source**: `My IP` (Khuyến nghị chỉ cho phép địa chỉ IP hiện tại của bạn) hoặc `Any`.
- **Source port ranges**: `*`
- **Destination port ranges**: `22`
- **Protocol**: `TCP`
- **Action**: `Allow`
- **Priority**: `300`
- **Name**: `Allow-SSH-Port-22`

#### Quy tắc 2: Cổng FastAPI Server (Giao tiếp học liên hiệp)
- **Source**: `Any` (Hoặc nhập dải địa chỉ IP công cộng của 4 máy trạm client để thắt chặt bảo mật)
- **Source port ranges**: `*`
- **Destination port ranges**: `8000`
- **Protocol**: `TCP`
- **Action**: `Allow`
- **Priority**: `310`
- **Name**: `Allow-FedLCCF-FastAPI-8000`

### 3.2. Mở cổng bằng Azure CLI
```bash
az network nsg rule create \
  --resource-group rg-fedlccf \
  --nsg-name vm-fedlccf-serverNSG \
  --name Allow-FedLCCF-FastAPI-8000 \
  --priority 310 \
  --destination-port-ranges 8000 \
  --protocol Tcp \
  --access Allow
```

---

## 4. Bước 3: Cài Đặt Môi Trường Máy Chủ (VM Provisioning)

### 4.1. Kết nối vào máy ảo qua SSH
Từ terminal máy tính của bạn:

```bash
ssh -i ~/.ssh/azure_fedlccf azureuser@<AZURE_VM_PUBLIC_IP>
```

### 4.2. Cập nhật hệ điều hành và gói hệ thống
Sau khi đăng nhập thành công vào Ubuntu VM:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git build-essential curl htop
```

### 4.3. Tải mã nguồn dự án về máy ảo
Clone mã nguồn kho lưu trữ của bạn về thư mục người dùng:

```bash
cd /home/azureuser
# Thay thế đường dẫn kho lưu trữ thực tế của bạn:
git clone https://github.com/your-org/FedLCCF.git FedLCCF
cd FedLCCF
```

*(Nếu sao chép trực tiếp từ máy cá nhân qua SCP)*:
```bash
# Chạy lệnh này từ máy cá nhân của bạn:
scp -i ~/.ssh/azure_fedlccf -r d:/FedLCCF/fedlccf_prototype azureuser@<AZURE_VM_PUBLIC_IP>:/home/azureuser/FedLCCF/
```

### 4.4. Tạo môi trường ảo và cài đặt thư viện
```bash
cd /home/azureuser/FedLCCF

# Tạo môi trường ảo Python 3
python3 -m venv .venv

# Kích hoạt môi trường ảo
source .venv/bin/activate

# Nâng cấp pip và cài đặt phụ thuộc
pip install --upgrade pip
pip install -r fedlccf_prototype/requirements.txt
```

---

## 5. Bước 4: Triển Khai Central Server Dưới Dạng Systemd Service

Để máy chủ FastAPI hoạt động bền bỉ, chạy ngầm liên tục 24/7 và **tự động khởi động lại nếu gặp sự cố (Auto-restart on failure)** hoặc khi VM bị reboot, chúng ta thiết lập một `systemd` service chuẩn sản xuất.

### 5.1. Khởi tạo tệp cấu hình Service
Mở trình soạn thảo `nano` để tạo tệp service:

```bash
sudo nano /etc/systemd/system/fedlccf-server.service
```

Dán toàn bộ nội dung cấu hình sau vào tệp:

```ini
[Unit]
Description=Fed-LCCF FastAPI Central Aggregation Server
After=network.target

[Service]
Type=simple
User=azureuser
Group=azureuser
WorkingDirectory=/home/azureuser/FedLCCF
Environment="PYTHONPATH=/home/azureuser/FedLCCF"
Environment="PYTHONUNBUFFERED=1"
Environment="HOST=0.0.0.0"
Environment="PORT=8000"
Environment="EXPECTED_CLIENTS=4"
Environment="GAMMA=10.0"
ExecStart=/home/azureuser/FedLCCF/.venv/bin/python -m fedlccf_prototype.cloud_server.server --host 0.0.0.0 --port 8000 --clients 4 --gamma 10.0
Restart=on-failure
RestartSec=5s
LimitNOFILE=65536
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Nhấn `Ctrl+O` -> `Enter` để lưu, sau đó nhấn `Ctrl+X` để thoát `nano`.

### 5.2. Kích hoạt và khởi động dịch vụ
Chạy chuỗi lệnh sau để nạp cấu hình mới và kích hoạt dịch vụ:

```bash
# 1. Nạp lại cấu hình systemd daemon
sudo systemctl daemon-reload

# 2. Thiết lập dịch vụ tự động khởi động cùng hệ điều hành
sudo systemctl enable fedlccf-server

# 3. Khởi động dịch vụ
sudo systemctl start fedlccf-server

# 4. Kiểm tra trạng thái dịch vụ (đảm bảo hiển thị active (running) màu xanh)
sudo systemctl status fedlccf-server
```

### 5.3. Các lệnh quản trị dịch vụ thường dùng
- **Xem nhật ký log thời gian thực**:
  ```bash
  journalctl -u fedlccf-server -f
  ```
- **Khởi động lại dịch vụ** (Ví dụ sau khi cập nhật mã nguồn):
  ```bash
  sudo systemctl restart fedlccf-server
  ```
- **Tạm dừng dịch vụ**:
  ```bash
  sudo systemctl stop fedlccf-server
  ```

### 5.4. Kiểm tra sức khỏe máy chủ từ xa
Từ máy tính cá nhân của bạn, mở trình duyệt hoặc chạy `curl`:

```bash
curl http://<AZURE_VM_PUBLIC_IP>:8000/health
```

Nếu trả về kết quả JSON với `"status": "healthy"`, máy chủ trung tâm trên Azure đã sẵn sàng tiếp nhận các client!

---

## 6. Bước 5: Cấu Hình và Kết Nối Client Từ Xa (Remote Client Agents)

Khi máy chủ trung tâm đã chạy trên Azure, 4 client trên các máy tính cục bộ khác nhau sẽ đóng vai trò các thành viên tham gia học liên hiệp.

### 6.1. Chuẩn bị dữ liệu trên máy trạm Client
Mỗi máy trạm client cần có mã nguồn `fedlccf_prototype` và tập dữ liệu SQLite tương ứng.
Nếu chưa có, chạy lệnh tiền xử lý trên máy trạm:
```bash
python fedlccf_prototype/data/preprocess_data.py
```
Lệnh này sẽ tạo ra các tệp `client_1.db` đến `client_4.db` trong thư mục `fedlccf_prototype/data/`.

### 6.2. Lệnh kết nối từng Client Agent
Mỗi thành viên trong nhóm mở terminal trên máy tính của mình và kích hoạt môi trường ảo `.venv`, sau đó chạy lệnh tương ứng với Client ID được phân công:

#### Client 1 (Máy trạm 1 - Nắm giữ 45% dữ liệu):
```bash
python -m fedlccf_prototype.client_ml.client_agent \
  --client-id 1 \
  --db-path fedlccf_prototype/data/client_1.db \
  --server-url http://<AZURE_VM_PUBLIC_IP>:8000 \
  --rounds 20 \
  --epochs-per-round 5 \
  --lr 0.01 \
  --latent-dim 32
```

#### Client 2 (Máy trạm 2 - Nắm giữ 30% dữ liệu):
```bash
python -m fedlccf_prototype.client_ml.client_agent \
  --client-id 2 \
  --db-path fedlccf_prototype/data/client_2.db \
  --server-url http://<AZURE_VM_PUBLIC_IP>:8000 \
  --rounds 20 \
  --epochs-per-round 5 \
  --lr 0.01 \
  --latent-dim 32
```

#### Client 3 (Máy trạm 3 - Nắm giữ 18% dữ liệu):
```bash
python -m fedlccf_prototype.client_ml.client_agent \
  --client-id 3 \
  --db-path fedlccf_prototype/data/client_3.db \
  --server-url http://<AZURE_VM_PUBLIC_IP>:8000 \
  --rounds 20 \
  --epochs-per-round 5 \
  --lr 0.01 \
  --latent-dim 32
```

#### Client 4 (Máy trạm 4 - Nắm giữ 7% dữ liệu, Cold-Start thưa thớt):
```bash
python -m fedlccf_prototype.client_ml.client_agent \
  --client-id 4 \
  --db-path fedlccf_prototype/data/client_4.db \
  --server-url http://<AZURE_VM_PUBLIC_IP>:8000 \
  --rounds 20 \
  --epochs-per-round 5 \
  --lr 0.01 \
  --latent-dim 32
```

### 6.3. Chạy toàn bộ 4 Clients từ xa bằng Orchestrator (Tùy chọn)
Nếu một thành viên muốn kiểm thử toàn bộ 4 clients kết nối đồng thời lên máy chủ Azure từ máy cá nhân của mình, sử dụng chế độ `clients-only`:

```bash
python -m fedlccf_prototype.orchestrator.run_simulation \
  --mode clients-only \
  --server-url http://<AZURE_VM_PUBLIC_IP>:8000 \
  --rounds 20 \
  --clients 4
```

Orchestrator sẽ khởi chạy 4 tiến trình nền cục bộ, liên tục gửi ma trận đặc trưng $V$ lên máy chủ Azure, đồng bộ rào cản và tải ma trận toàn cục về một cách tự động.

---

## 7. Bước 6: Tùy Chọn Triển Khai Bằng Docker Compose Trên Azure VM

Nếu bạn không muốn quản lý môi trường Python trực tiếp qua `systemd`, bạn có thể chạy máy chủ thông qua Docker Compose trên Azure VM.

### 7.1. Cài đặt Docker trên Ubuntu 22.04 LTS
```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER
```
*(Đăng xuất SSH và đăng nhập lại để quyền nhóm docker có hiệu lực).*

### 7.2. Khởi chạy Central Server Container
Di chuyển vào thư mục dự án và bật container:

```bash
cd /home/azureuser/FedLCCF/fedlccf_prototype

# Khởi chạy server dạng background daemon
docker compose up -d server

# Kiểm tra trạng thái hoạt động
docker compose ps
```

### 7.3. Xem logs container server
```bash
docker compose logs -f server
```

Container đã được cấu hình cờ `restart: unless-stopped` trong `docker-compose.yml`, do đó nếu Docker daemon hoặc VM khởi động lại, container máy chủ cũng sẽ tự động phục hồi.

---

## 8. Bước 7: Khuyến Nghị Bảo Mật Nâng Cao (UFW & Nginx SSL/TLS)

Để chuyển giao hệ thống sang môi trường sản xuất hoặc nghiệm thu đồ án an toàn, bạn nên kích hoạt các lớp bảo vệ mạng sau:

### 8.1. Cấu hình tường lửa cục bộ UFW (Uncomplicated Firewall)
Bên cạnh Azure NSG, cấu hình UFW trên chính Ubuntu VM giúp bảo vệ nhiều lớp:

```bash
# Đặt chính sách mặc định: chặn toàn bộ kết nối đến, cho phép kết nối đi
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Mở cổng SSH 22
sudo ufw allow 22/tcp

# Mở cổng FastAPI 8000 (hoặc cổng HTTP 80 / HTTPS 443 nếu dùng Nginx)
sudo ufw allow 8000/tcp

# Kích hoạt UFW
sudo ufw enable

# Kiểm tra trạng thái
sudo ufw status verbose
```

*(Mẹo bảo mật nâng cao: Chỉ cho phép các IP máy trạm xác định truy cập cổng 8000)*:
```bash
sudo ufw delete allow 8000/tcp
sudo ufw allow from <IP_MAY_TRAM_1> to any port 8000 proto tcp
sudo ufw allow from <IP_MAY_TRAM_2> to any port 8000 proto tcp
```

### 8.2. Cấu hình Nginx Reverse Proxy & Chứng chỉ HTTPS (Let's Encrypt)
Thay vì để FastAPI lắng nghe trực tiếp trên Internet, khuyến nghị đưa Nginx lên làm Reverse Proxy để mã hóa đường truyền SSL/TLS và xử lý tải tĩnh:

1. **Cài đặt Nginx và Certbot**:
   ```bash
   sudo apt install -y nginx certbot python3-certbot-nginx
   ```

2. **Tạo cấu hình ảo cho Nginx**:
   ```bash
   sudo nano /etc/nginx/sites-available/fedlccf
   ```
   Dán cấu hình sau:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com; # Hoặc sử dụng Public IP nếu không có tên miền

       # Tăng kích thước payload tối đa cho các ma trận đặc trưng lớn
       client_max_body_size 50M;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
           proxy_read_timeout 300s;
           proxy_connect_timeout 300s;
       }
   }
   ```

3. **Kích hoạt Virtual Host**:
   ```bash
   sudo ln -s /etc/nginx/sites-available/fedlccf /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

4. **Cấp phát chứng chỉ SSL/TLS miễn phí qua Certbot**:
   *(Yêu cầu đã trỏ tên miền DNS về Azure VM Public IP)*:
   ```bash
   sudo certbot --nginx -d your-domain.com
   ```
   Sau khi hoàn tất, mọi giao tiếp của Client sẽ sử dụng URL an toàn: `https://your-domain.com`.

---

## 9. Tổng Kết & Checklist Bàn Giao

Trước khi tiến hành buổi đánh giá hoặc thử nghiệm thực tế, hãy rà soát danh sách kiểm tra sau:
- [x] Azure VM đã mở cổng 22 (SSH) và 8000 (FastAPI) trên Azure NSG.
- [x] Dịch vụ `fedlccf-server.service` hiển thị trạng thái `active (running)`.
- [x] Lệnh `curl http://<AZURE_VM_PUBLIC_IP>:8000/health` trả về mã HTTP `200 OK`.
- [x] Dữ liệu SQLite `client_1.db` đến `client_4.db` đã được chuẩn bị sẵn trên các máy trạm.
- [x] Các máy trạm Client kết nối thành công và vượt qua rào cản đồng bộ qua 20 rounds liên hiệp.
