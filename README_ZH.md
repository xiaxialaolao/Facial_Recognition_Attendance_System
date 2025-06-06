# 人脸识别考勤系统 (Face Recognition Attendance System)

一个基于树莓派的人脸识别考勤系统，支持实时人脸检测、识别和考勤记录，具有网页管理界面。

## 示意图

<img src="https://github.com/xiaxialaolao/Facial_Recognition_Attendance_System/blob/main/Flow%20Diagram/Schematic%20Diagram.png" width="75%">

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

<img src="https://github.com/xiaxialaolao/Facial_Recognition_Attendance_System/blob/main/Flow%20Diagram/Directory%20Structure.png" width="40%">

## 注意事项

- **本系统主要用于学校毕业设计项目**，适合作为学习和研究用途的毕设项目
- 系统存在一定局限性和优化空间，不建议直接用于实际生产环境
- 作为教学演示和学术研究项目，本系统实现了基本功能，但在稳定性、安全性和性能方面还有提升空间
- 若要基于本项目进行实际应用开发，建议进行全面的代码审查和性能优化
- 本系统仅用于考勤管理，不应用于安全关键型应用
- 请妥善保管用户数据，遵守相关隐私法规
- 定期备份数据库和人脸图像数据集
- 系统需要稳定的电源供应，建议使用官方电源适配器
