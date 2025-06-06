# 人脸识别考勤系统 (Face Recognition Attendance System)

一个基于树莓派的人脸识别考勤系统，支持实时人脸检测、识别和考勤记录，具有网页管理界面。

## 系统架构

该系统由三个主要组件组成：

### 1. 人脸识别系统 (FRAS.py)

核心人脸识别引擎，负责：
- 实时摄像头捕获
- 人脸检测与追踪
- 人脸特征比对与识别
- 考勤记录
- GPIO 控制（LED 指示灯）
- 系统资源监控

### 2. 人脸特征提取与编码 (FEFE.py)

负责处理人脸图像并提取特征向量：
- 从图像数据集中提取人脸特征
- 生成 128 维特征向量
- 保存编码数据用于识别

### 3. 网页管理界面 (HTML 目录)

提供用户友好的管理界面：
- 用户管理（添加、编辑、删除）
- 人脸图像采集
- 考勤记录查询
- 系统日志查看
- 系统监控
- 多语言支持（中文、英文）

## 主要功能和特性

- **实时人脸识别**：使用高效的人脸检测和识别算法
- **多人同时识别**：支持同时识别多个人脸
- **高精度识别**：使用 128 维特征向量确保高准确率
- **自动考勤记录**：识别成功后自动记录考勤信息
- **网页管理界面**：易于使用的用户管理和数据查询界面
- **移动设备兼容**：响应式设计，支持在移动设备上使用
- **多语言支持**：支持中文和英文界面
- **系统监控**：实时监控 CPU、内存、温度等系统资源
- **日志记录**：详细记录系统操作和错误信息

## 使用说明

### 系统要求

- 树莓派 5（推荐）或兼容设备
- Python 3.7+
- 网络摄像头或树莓派摄像头模块
- 至少 2GB RAM
- 至少 16GB 存储空间
- MySQL/MariaDB 数据库

### 初始设置

1. 克隆或下载项目到树莓派

2. 创建并激活虚拟环境（推荐）：

```bash
# 创建虚拟环境
python -m venv FRAS_env

# 激活虚拟环境
source FRAS_env/bin/activate
```

3. 安装依赖：

```bash
# 安装所需的 Python 包
pip install face_recognition opencv-python numpy scipy psutil picamera2 mysql-connector-python
```

4. 配置数据库：

```bash
# 登录 MySQL
mysql -u root -p

# 创建数据库
CREATE DATABASE Facial_Recognition_Attendance_System;

# 创建用户并授权
CREATE USER 'fras_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON Facial_Recognition_Attendance_System.* TO 'fras_user'@'localhost';
FLUSH PRIVILEGES;
```

5. 修改数据库配置：
   - 编辑 `HTML/config.php` 文件，更新数据库连接信息
   - 编辑 `db_connector.py` 文件，更新数据库连接信息

### 启动系统

#### 方法 1：直接启动各组件

1. 启动人脸识别系统：

```bash
# 确保已激活虚拟环境
source FRAS_env/bin/activate

# 启动人脸识别系统
python FRAS.py
```

2. 启动网页服务器（如果未在 FRAS.py 中自动启动）：

```bash
# 确保已激活虚拟环境
source FRAS_env/bin/activate

# 启动网页服务器
python web_server.py
```

3. 运行人脸特征提取（按需）：

```bash
# 确保已激活虚拟环境
source FRAS_env/bin/activate

# 运行人脸特征提取
python FEFE.py
```

### 访问网页界面

系统启动后，可以通过浏览器访问网页界面：

```
http://[树莓派IP地址]:80
```

## 人脸数据采集

### 方法 1：通过网页界面采集

1. 登录网页界面
2. 导航至"图像采集"页面
3. 选择用户或创建新用户
4. 使用网页摄像头捕获人脸图像
5. 系统会自动处理并保存图像

### 方法 2：上传图像文件

1. 登录网页界面
2. 导航至"图像采集"页面
3. 选择用户
4. 点击"上传图像"按钮
5. 选择包含清晰人脸的图像文件（支持 JPG、JPEG、PNG 格式）
6. 系统会自动处理并保存图像

## 系统配置选项

### FRAS.py 配置选项

- `CAMERA_ROTATION`：摄像头旋转角度（默认：0）
- `CAMERA_RESOLUTION`：摄像头分辨率（默认：(1280, 720)）
- `RECOGNITION_TOLERANCE`：人脸识别容差值（默认：0.06，越小越严格）

### 数据收集器配置选项

在 FEFE.py 中可配置：
- `IMAGE_DIR`：存储人脸图像的目录（默认：'Image_DataSet'）
- `ENCODING_DIR`：存储特征编码的目录（默认：'Encoding_DataSet'）
- `SCAN_INTERVAL`：自动扫描新图像的时间间隔，单位秒（默认：60）
- `FORCE_RECOMPUTE`：是否强制重新生成所有特征编码（默认：False）

## 目录结构

```
├── FRAS.py                 # 主人脸识别系统
├── FEFE.py                 # 人脸特征提取与编码
├── web_server.py           # Web 服务器
├── db_connector.py         # 数据库连接器
├── gpio_control.py         # GPIO 控制模块
├── HTML/                   # 网页界面文件
│   ├── config.php          # 数据库配置
│   ├── dashboard.php       # 仪表盘页面
│   ├── image_acquisition.php # 图像采集页面
│   ├── login.php           # 登录页面
│   ├── log.php             # 日志查看页面
│   └── ...                 # 其他网页文件
├── Image_DataSet/          # 人脸图像数据集
└── Encoding_DataSet/       # 特征编码数据集
```

## 常见问题和解决方案

### 1. 系统无法识别人脸

- **注意：系统使用的人脸识别模型相对过时**，准确率有限，且识别代码实现不算优质，可能导致识别问题
- 系统在理想条件下可以正常工作，但在复杂环境中可能表现不佳
- 确保光线充足，避免背光
- 调整摄像头位置，确保人脸清晰可见
- **注意距离因素**：距离过远会导致无法识别，建议自己把控合适的距离（通常 0.5-1.5 米为宜）
- 增加每个用户的人脸图像数量（建议至少 10 张）
- 尝试降低 `RECOGNITION_TOLERANCE` 值（在 FRAS.py 中修改）
- 检查 `Encoding_DataSet` 目录中是否存在用户的特征编码文件
- 运行 `FEFE.py` 重新生成特征编码
- 考虑使用更现代的人脸识别库或模型来提高准确率（如需用于实际应用）

### 2. 网页界面无法访问

- 确认 Web 服务器已启动
- 检查防火墙设置，确保端口 80 已开放
- 验证树莓派 IP 地址是否正确
- 检查网络连接
- 查看 Web 服务器日志：`tail -f logs/web_server.log`
- 尝试重启 Web 服务器：`python web_server.py`

### 3. 图像采集失败

- 确保用户有足够的存储空间：`df -h`
- 检查摄像头连接是否正常：`vcgencmd get_camera`
- 验证用户对 Image_DataSet 目录有写入权限：`ls -la Image_DataSet/`
- 检查日志文件中的错误信息：`tail -f logs/system.log`
- 确保目录存在：`mkdir -p Image_DataSet`
- 尝试重启系统

### 4. 系统性能问题

- 降低摄像头分辨率（当前默认为 1280x720，可在 FRAS.py 中修改 CAMERA_WIDTH 和 CAMERA_HEIGHT）
- 减少同时识别的人脸数量
- 关闭不必要的系统服务：`sudo systemctl disable <service_name>`
- 监控系统资源使用情况：`top` 或 `htop`
- 检查系统温度：`vcgencmd measure_temp`
- 考虑升级硬件（更多 RAM 或更快的 SD 卡）

### 5. 数据库连接问题

- 检查数据库服务是否运行：`sudo systemctl status mysql`
- 验证数据库凭据是否正确（检查 config.php 和 db_connector.py）
- 尝试手动连接数据库：`mysql -u fras_user -p Facial_Recognition_Attendance_System`
- 检查数据库日志：`sudo tail -f /var/log/mysql/error.log`
- 重启数据库服务：`sudo systemctl restart mysql`

### 6. 摄像头问题

- 检查摄像头是否被其他进程占用：`sudo lsof /dev/video0`
- 验证摄像头权限：`ls -la /dev/video*`
- 尝试重新加载摄像头模块：
  ```bash
  sudo rmmod bcm2835-v4l2
  sudo modprobe bcm2835-v4l2
  ```
- 对于 Pi 摄像头，确保在 `raspi-config` 中启用了摄像头接口

## 注意事项

- **本系统主要用于学校毕业设计项目**，适合作为学习和研究用途的毕设项目
- 系统存在一定局限性和优化空间，不建议直接用于实际生产环境
- 作为教学演示和学术研究项目，本系统实现了基本功能，但在稳定性、安全性和性能方面还有提升空间
- 若要基于本项目进行实际应用开发，建议进行全面的代码审查和性能优化
- 本系统仅用于考勤管理，不应用于安全关键型应用
- 请妥善保管用户数据，遵守相关隐私法规
- 定期备份数据库和人脸图像数据集
- 系统需要稳定的电源供应，建议使用官方电源适配器

