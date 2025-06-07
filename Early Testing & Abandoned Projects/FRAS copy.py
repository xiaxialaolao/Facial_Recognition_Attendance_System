from concurrent.futures import ThreadPoolExecutor
from picamera2 import Picamera2
import cv2
import face_recognition
import numpy as np
import os
import time
import logging
import queue
import json
from threading import Event, Thread, Lock, Timer
import multiprocessing
from scipy import ndimage
from scipy import signal
from scipy.spatial import distance
import psutil
import datetime
from pathlib import Path

# 导入 GPIO 控制模块
from gpio_control import (
    initialize_gpio, set_gpio_state, cleanup_gpio,
    set_recognized_face, set_unknown_face, set_standby_state
)

# 导入 Web 服务器模块
try:
    import web_server
    WEB_ENABLED = True
except ImportError:
    logging.warning("Web server module import failed, web streaming will be disabled")
    WEB_ENABLED = False

# 导入数据库连接器
try:
    from db_connector import db_connector
    DB_ENABLED = True
except ImportError:
    logging.warning("Database connector import failed, attendance recording will be disabled")
    DB_ENABLED = False


# 配置日志 - 发布版本调回INFO级别以减少写日志的开销
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FaceRecognition")

# GPIO将在需要时自动初始化
# initialize_gpio()

# 检查系统CPU核心数
CPU_COUNT = multiprocessing.cpu_count()
logger.info(f"System CPU cores: {CPU_COUNT}")

class SystemMonitor:
    """系统性能监控类，用于记录CPU、内存、温度等系统资源使用情况"""

    def __init__(self, interval=60):
        """初始化系统监控器

        Args:
            interval: 监控间隔，单位为秒，默认60秒
        """
        self.interval = interval
        self.running = Event()
        self.running.set()
        self.monitor_thread = None

        # 创建logs目录（如果不存在）
        Path("logs").mkdir(exist_ok=True)

        # 创建日志文件名
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        self.log_file = f"logs/system_monitor_{today}.log"

        # 创建专用的日志记录器
        self.monitor_logger = logging.getLogger("SystemMonitor")
        self.monitor_logger.setLevel(logging.INFO)

        # 创建文件处理器
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)

        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 添加处理器到日志记录器
        self.monitor_logger.addHandler(file_handler)

        # 记录初始信息
        logger.info(f"System monitor initialized. Log file: {self.log_file}")
        self.monitor_logger.info("System monitoring started")
        self.monitor_logger.info(f"System information: CPU cores: {CPU_COUNT}, Platform: {os.sys.platform}")

    def start(self):
        """启动监控线程"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.running.set()
            self.monitor_thread = Thread(target=self._monitor_loop, daemon=True, name="SystemMonitorThread")
            self.monitor_thread.start()
            logger.info("System monitor thread started")

    def stop(self):
        """停止监控线程"""
        self.running.clear()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
            logger.info("System monitor thread stopped")
            self.monitor_logger.info("System monitoring stopped")

    def _get_cpu_temperature(self):
        """获取CPU温度（树莓派特定）"""
        try:
            # 尝试从树莓派的温度文件读取温度
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = float(f.read()) / 1000.0
            return temp
        except Exception as e:
            self.monitor_logger.warning(f"Failed to read CPU temperature: {str(e)}")
            return None

    def _estimate_power_consumption(self, cpu_percent, cpu_temp):
        """估算功耗（基于CPU使用率和温度的简单估算）"""
        if cpu_temp is None:
            cpu_temp = 50.0  # 假设默认温度

        # 简单估算：基础功耗 + CPU负载相关功耗 + 温度相关功耗
        # 树莓派5基础功耗约5W，满载约10-15W
        base_power = 5.0  # 基础功耗（W）
        cpu_power = (cpu_percent / 100.0) * 8.0  # CPU负载相关功耗（最高8W）
        temp_factor = max(0, (cpu_temp - 40) / 40) * 2.0  # 温度相关额外功耗（最高2W）

        total_power = base_power + cpu_power + temp_factor
        return total_power

    def _monitor_loop(self):
        """监控循环，定期记录系统资源使用情况"""
        last_log_time = 0

        while self.running.is_set():
            current_time = time.time()

            # 每隔指定时间记录一次
            if current_time - last_log_time >= self.interval:
                try:
                    # 获取CPU使用率（所有核心）
                    cpu_percent_all = psutil.cpu_percent(interval=1)
                    cpu_percent_per_core = psutil.cpu_percent(interval=1, percpu=True)

                    # 获取内存使用情况
                    memory = psutil.virtual_memory()
                    memory_total_mb = memory.total / (1024 * 1024)
                    memory_used_mb = memory.used / (1024 * 1024)
                    memory_available_mb = memory.available / (1024 * 1024)
                    memory_percent = memory.percent

                    # 获取CPU温度
                    cpu_temp = self._get_cpu_temperature()

                    # 估算功耗
                    power_consumption = self._estimate_power_consumption(cpu_percent_all, cpu_temp)

                    # 创建监控数据字典
                    monitor_data = {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "cpu": {
                            "total_percent": cpu_percent_all,
                            "per_core_percent": cpu_percent_per_core
                        },
                        "memory": {
                            "total_mb": round(memory_total_mb, 2),
                            "used_mb": round(memory_used_mb, 2),
                            "available_mb": round(memory_available_mb, 2),
                            "percent": memory_percent
                        },
                        "temperature": {
                            "cpu": cpu_temp
                        },
                        "power": {
                            "estimated_watts": round(power_consumption, 2)
                        }
                    }

                    # 记录监控数据
                    self.monitor_logger.info(json.dumps(monitor_data))

                    # 更新上次记录时间
                    last_log_time = current_time

                except Exception as e:
                    self.monitor_logger.error(f"Error in monitoring loop: {str(e)}")

            # 短暂休眠，避免CPU占用过高
            time.sleep(1)

# 预加载关键模块
logger.info("Preloading OpenCV and face_recognition modules...")
try:
    # 预热OpenCV
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.resize(dummy_img, (50, 50))
    cv2.cvtColor(dummy_img, cv2.COLOR_BGR2GRAY)

    # 预热face_recognition
    _ = face_recognition.face_locations(dummy_img)
    _ = face_recognition.face_encodings(dummy_img)

    logger.info("Modules preloaded successfully")
except Exception as e:
    logger.warning(f"Module preloading failed: {str(e)}")

class FrameProcessor:
    """帧处理器基类"""
    def __init__(self, name):
        self.name = name
        self.input_queue = queue.Queue(maxsize=3)
        self.output_queue = queue.Queue(maxsize=3)
        self.running = Event()
        self.running.set()

    def put_frame(self, frame):
        """将帧放入输入队列"""
        try:
            if self.input_queue.full():
                try:
                    self.input_queue.get_nowait()  # 移除旧帧
                except queue.Empty:
                    pass
            self.input_queue.put(frame, block=False)
        except Exception:
            pass  # 忽略队列错误

    def get_result(self):
        """从输出队列获取结果"""
        try:
            return self.output_queue.get(block=False)
        except queue.Empty:
            return None

    def start(self):
        """启动处理线程"""
        self.thread = Thread(target=self._process_loop, name=self.name)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """停止处理线程"""
        self.running.clear()

    def _process_loop(self):
        """处理循环，子类需要重写"""
        raise NotImplementedError("Subclasses must implement _process_loop method")

class FaceTracker:
    """增强版人脸追踪器 - 使用多特征融合追踪"""
    def __init__(self):
        self.tracked_faces = {}  # 记录追踪的人脸
        self.next_id = 0
        self.max_lost_frames = 20  # 进一步减少最大丢失帧数，使系统更快地重置人脸状态
        self.match_threshold = 0.35  # 进一步降低IOU匹配阈值，提高匹配灵敏度
        self.velocity_weight = 0.3  # 速度预测权重
        self.size_change_threshold = 0.5  # 尺寸变化阈值
        self.kalman_filters = {}  # 卡尔曼滤波器字典
        self.recognition_results_history = {}  # 添加对识别结果历史的引用

    def _init_kalman(self, rect):
        """初始化卡尔曼滤波器"""
        x, y, w, h = rect
        # 状态向量 [x, y, w, h, dx, dy, dw, dh]
        kalman = cv2.KalmanFilter(8, 4)
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)

        kalman.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + dx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + dy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + dw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + dh
            [0, 0, 0, 0, 1, 0, 0, 0],  # dx = dx
            [0, 0, 0, 0, 0, 1, 0, 0],  # dy = dy
            [0, 0, 0, 0, 0, 0, 1, 0],  # dw = dw
            [0, 0, 0, 0, 0, 0, 0, 1]   # dh = dh
        ], np.float32)

        # 设置过程噪声协方差矩阵
        kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03

        # 设置测量噪声协方差矩阵
        kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1

        # 初始化状态
        kalman.statePre = np.array([x, y, w, h, 0, 0, 0, 0], np.float32)
        kalman.statePost = np.array([x, y, w, h, 0, 0, 0, 0], np.float32)

        return kalman

    def update(self, faces):
        """更新追踪状态"""
        if not self.tracked_faces and len(faces) == 0:
            # 如果没有人脸
            return {}

        # 初始化新的追踪字典
        new_tracked = {}

        # 预测所有现有追踪的下一个位置
        predicted_positions = {}
        for face_id, face_info in self.tracked_faces.items():
            if face_id in self.kalman_filters:
                kalman = self.kalman_filters[face_id]
                prediction = kalman.predict()
                # 提取预测的矩形
                x, y, w, h = prediction[0:4]
                # 确保尺寸为正
                w = max(10, w)
                h = max(10, h)
                predicted_positions[face_id] = (float(x), float(y), float(w), float(h))
            else:
                # 如果没有卡尔曼滤波器，使用当前位置
                predicted_positions[face_id] = face_info["rect"]

        # 如果有检测到的人脸
        if len(faces) > 0:
            # 计算所有可能的匹配组合的分数矩阵
            match_scores = np.zeros((len(faces), len(self.tracked_faces)))

            for i, (x, y, w, h) in enumerate(faces):
                face_rect = (x, y, w, h)
                face_center = (x + w/2, y + h/2)

                for j, face_id in enumerate(self.tracked_faces.keys()):
                    # 获取预测位置
                    pred_rect = predicted_positions[face_id]
                    pred_center = (pred_rect[0] + pred_rect[2]/2, pred_rect[1] + pred_rect[3]/2)

                    # 计算IOU分数
                    iou_score = self._calculate_iou(face_rect, pred_rect)

                    # 计算中心点距离分数
                    center_dist = np.sqrt((face_center[0] - pred_center[0])**2 +
                                         (face_center[1] - pred_center[1])**2)
                    max_dim = max(w, h, pred_rect[2], pred_rect[3])
                    center_score = max(0, 1 - center_dist / max_dim)

                    # 计算尺寸相似度分数
                    size_ratio = min(w * h, pred_rect[2] * pred_rect[3]) / max(w * h, pred_rect[2] * pred_rect[3])

                    # 融合多个特征的分数
                    combined_score = iou_score * 0.5 + center_score * 0.3 + size_ratio * 0.2
                    match_scores[i, j] = combined_score

            # 贪婪匹配算法
            matched_faces = set()
            matched_tracks = set()

            # 按分数从高到低排序所有可能的匹配
            match_indices = np.dstack(np.unravel_index(np.argsort(match_scores.ravel())[::-1],
                                                     match_scores.shape))[0]

            for i, j in match_indices:
                if i not in matched_faces and j not in matched_tracks and match_scores[i, j] > self.match_threshold:
                    face_id = list(self.tracked_faces.keys())[j]
                    face_rect = faces[i]

                    # 更新卡尔曼滤波器
                    if face_id in self.kalman_filters:
                        kalman = self.kalman_filters[face_id]
                        measurement = np.array([face_rect[0], face_rect[1],
                                              face_rect[2], face_rect[3]], np.float32)
                        kalman.correct(measurement)
                    else:
                        # 创建新的卡尔曼滤波器
                        self.kalman_filters[face_id] = self._init_kalman(face_rect)

                    # 更新已匹配的人脸
                    face_info = self.tracked_faces[face_id]
                    face_info["rect"] = face_rect
                    face_info["lost_frames"] = 0
                    face_info["total_frames"] += 1

                    # 实时刷新识别状态 - 每帧都允许重新识别
                    # 这样可以立即识别新出现的人脸，不需要等待回到未识别状态
                    face_info["need_recognition"] = True

                    # 如果已经有稳定的识别结果，可以降低识别频率以节省资源
                    if face_id in self.recognition_results_history:
                        history = self.recognition_results_history.get(face_id, {})
                        # 如果已经有稳定的已知人脸识别结果，降低识别频率
                        if history.get("name", "Unknown") != "Unknown" and history.get("stable_count", 0) > 3:
                            # 每2帧识别一次已知的稳定人脸，平衡实时性和性能
                            face_info["need_recognition"] = (face_info["total_frames"] % 2 == 0)

                    new_tracked[face_id] = face_info
                    matched_faces.add(i)
                    matched_tracks.add(j)

            # 处理未匹配的检测结果 - 创建新的追踪
            for i, face_rect in enumerate(faces):
                if i not in matched_faces:
                    # 创建新的卡尔曼滤波器
                    new_id = self.next_id
                    self.kalman_filters[new_id] = self._init_kalman(face_rect)

                    # 创建新的追踪记录
                    new_tracked[new_id] = {
                        "rect": face_rect,
                        "lost_frames": 0,
                        "total_frames": 1,
                        "need_recognition": True,  # 新人脸需要立即识别
                        "name": None,
                        "confidence": 0,
                        "velocity": (0, 0)  # 初始速度为0
                    }
                    self.next_id += 1

        # 更新未匹配的追踪记录（丢失帧数增加）
        for face_id, face_info in self.tracked_faces.items():
            if face_id not in new_tracked:
                face_info["lost_frames"] += 1

                # 使用卡尔曼滤波器预测位置
                if face_id in self.kalman_filters and face_info["lost_frames"] <= self.max_lost_frames:
                    # 获取预测位置
                    pred_rect = predicted_positions[face_id]
                    # 确保矩形坐标是整数
                    pred_rect = (int(pred_rect[0]), int(pred_rect[1]),
                                int(pred_rect[2]), int(pred_rect[3]))
                    # 更新矩形位置
                    face_info["rect"] = pred_rect
                    new_tracked[face_id] = face_info
                elif face_info["lost_frames"] <= self.max_lost_frames:
                    # 如果没有卡尔曼滤波器但仍在最大丢失帧数内
                    # 确保矩形坐标是整数
                    rect = face_info["rect"]
                    face_info["rect"] = (int(rect[0]), int(rect[1]),
                                        int(rect[2]), int(rect[3]))
                    new_tracked[face_id] = face_info

        # 清理长时间未匹配的卡尔曼滤波器
        for face_id in list(self.kalman_filters.keys()):
            if face_id not in new_tracked:
                del self.kalman_filters[face_id]

        self.tracked_faces = new_tracked

        # GPIO control code has been removed

        return self.tracked_faces

    def _calculate_iou(self, rect1, rect2):
        """计算两个矩形的IOU"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        # 计算交集区域
        x_inter = max(x1, x2)
        y_inter = max(y1, y2)
        w_inter = min(x1 + w1, x2 + w2) - x_inter
        h_inter = min(y1 + h1, y2 + h2) - y_inter

        if w_inter <= 0 or h_inter <= 0:
            return 0

        # 计算交集面积
        inter_area = w_inter * h_inter

        # 计算两个矩形的面积
        area1 = w1 * h1
        area2 = w2 * h2

        # 计算并集面积
        union_area = area1 + area2 - inter_area

        # 返回IOU
        return inter_area / union_area

class FaceRecognitionWorker:
    """增强版异步人脸识别工作线程"""
    def __init__(self, face_encodings, face_names, face_files=None, tolerance=0.06):  # 设置极低的容忍度，要求置信度>0.94
        self.known_face_encodings = face_encodings
        self.known_face_names = face_names
        self.known_face_files = face_files if face_files is not None else ["unknown.json"] * len(face_names)
        self.tolerance = tolerance
        self.tasks = queue.Queue(maxsize=5)  # 增加任务队列大小
        self.results = queue.Queue()
        self.running = Event()
        self.running.set()

        # 增加批量处理
        self.process_batch_size = 1  # 默认每次处理1个任务
        if CPU_COUNT > 2:
            self.process_batch_size = 2  # 多核CPU时可以增加批处理大小

        # 优化缓存机制，平衡实时性和性能
        self.recognition_cache = {}  # 缓存识别结果
        self.cache_timeout = 2.0  # 降低缓存超时时间，提高实时性
        self.known_face_cache_timeout = 5.0  # 已知人脸可以使用更长的缓存时间

        # 启动工作线程
        self.thread = Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

        # 预热模型
        self._warmup_model()

    def _warmup_model(self):
        """预热人脸识别模型，减少首次识别延迟"""
        try:
            # 创建一个简单的测试图像
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            # 预热face_recognition
            _ = face_recognition.face_locations(test_img, model="hog")
            _ = face_recognition.face_encodings(test_img)
            logger.info("Face recognition model warmed up")
        except Exception as e:
            logger.warning(f"Model warmup failed: {str(e)}")

    def submit_task(self, face_id, face_img):
        """提交人脸识别任务"""
        try:
            # 检查缓存中是否有最近的结果
            if face_id in self.recognition_cache:
                cache_entry = self.recognition_cache[face_id]
                current_time = time.time()
                cache_age = current_time - cache_entry["timestamp"]

                # 根据人脸类型使用不同的缓存超时时间
                # 已知人脸使用更长的缓存时间，未知人脸使用更短的缓存时间
                timeout = self.known_face_cache_timeout if cache_entry["name"] != "Unknown" else self.cache_timeout

                # 如果缓存未过期，直接返回缓存结果
                if cache_age < timeout:
                    # 将缓存结果放入结果队列
                    self.results.put(cache_entry)
                    return True

            # 如果队列满，不阻塞，直接返回失败
            if self.tasks.full():
                return False

            # 计算图像质量分数
            quality_score = self._calculate_image_quality(face_img)

            # 将任务放入队列
            self.tasks.put((face_id, face_img, quality_score), block=False)
            return True
        except Exception as e:
            logger.error(f"Error submitting task: {str(e)}")
            return False

    def _calculate_image_quality(self, image):
        """计算图像质量分数"""
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # 计算拉普拉斯方差（清晰度指标）
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # 计算亮度
            brightness = np.mean(gray)

            # 计算对比度
            contrast = np.std(gray)

            # 综合评分 (0-100)
            quality_score = min(100, max(0,
                               (laplacian_var / 500) * 50 +  # 清晰度权重
                               (brightness / 255) * 25 +     # 亮度权重
                               (contrast / 128) * 25))       # 对比度权重

            return quality_score
        except Exception:
            return 50  # 默认中等质量

    def get_result(self):
        """获取识别结果"""
        try:
            return self.results.get(block=False)
        except queue.Empty:
            return None

    def stop(self):
        """停止工作线程"""
        self.running.clear()

    def _enhance_face_image(self, face_img):
        """增强人脸图像质量"""
        try:
            # 确保图像大小合适
            if face_img.shape[0] < 80 or face_img.shape[1] < 80:
                # 放大小图像
                face_img = cv2.resize(face_img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            # 转换为YUV颜色空间进行处理
            yuv = cv2.cvtColor(face_img, cv2.COLOR_BGR2YUV)

            # 自适应直方图均衡化 - 仅应用于Y通道（亮度）
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            yuv[:,:,0] = clahe.apply(yuv[:,:,0])

            # 转回BGR
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

            # 应用轻微锐化
            kernel = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)

            # 应用轻微的高斯模糊去除噪点
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)

            return enhanced
        except Exception as e:
            logger.warning(f"Image enhancement failed: {str(e)}")
            return face_img

    def _worker_loop(self):
        """工作线程循环"""
        while self.running.is_set():
            try:
                # 获取任务
                face_id, face_img, quality_score = self.tasks.get(block=True, timeout=0.5)

                # 确保图像有效
                if face_img is None or face_img.size == 0 or face_img.shape[0] < 20 or face_img.shape[1] < 20:
                    # 图像无效，返回未知结果
                    result = {
                        "face_id": face_id,
                        "name": "Unknown",
                        "confidence": 0,
                        "timestamp": time.time()
                    }
                    self.results.put(result)
                    continue  # 跳过处理

                # 增强图像质量
                enhanced_img = self._enhance_face_image(face_img)

                # 确保图像是RGB格式
                rgb_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)

                # 尝试多种尺寸进行处理，提高检测率
                face_locations = []
                face_encoding = None

                # 根据图像质量选择不同的处理策略
                if quality_score > 70:  # 高质量图像
                    # 直接使用原始大小和高精度模型
                    face_locations = face_recognition.face_locations(rgb_img, model="hog")
                    if face_locations:
                        face_encoding = face_recognition.face_encodings(
                            rgb_img,
                            [face_locations[0]],
                            num_jitters=2,  # 增加抖动采样次数提高精度
                            model="large"  # 使用大模型提高精度
                        )[0]
                else:
                    # 先尝试原始大小
                    face_locations = face_recognition.face_locations(rgb_img, model="hog")

                    # 如果检测失败，尝试缩小图像
                    if not face_locations:
                        # 缩小图像以加速处理
                        small_img = cv2.resize(rgb_img, (0, 0), fx=0.5, fy=0.5)
                        face_locations = face_recognition.face_locations(small_img, model="hog")

                        # 如果仍然失败，尝试不同参数
                        if not face_locations:
                            # 尝试调整参数
                            face_locations = face_recognition.face_locations(
                                small_img,
                                model="hog",
                                number_of_times_to_upsample=2  # 增加上采样次数
                            )

                            if face_locations:
                                # 使用小图像进行编码
                                face_encoding = face_recognition.face_encodings(
                                    small_img,
                                    [face_locations[0]],
                                    num_jitters=2,  # 增加抖动采样次数提高精度
                                    model="small"
                                )[0]
                        else:
                            # 使用小图像进行编码
                            face_encoding = face_recognition.face_encodings(
                                small_img,
                                [face_locations[0]],
                                num_jitters=1,
                                model="small"
                            )[0]
                    else:
                        # 使用原始图像进行编码
                        face_encoding = face_recognition.face_encodings(
                            rgb_img,
                            [face_locations[0]],
                            num_jitters=1,
                            model="small"
                        )[0]

                name = "Unknown"
                confidence = 0

                # 如果检测到人脸并成功编码
                if face_encoding is not None:
                    # 与已知人脸比较
                    if len(self.known_face_encodings) > 0:
                        # 使用余弦相似度计算距离
                        face_distances = []
                        for known_encoding in self.known_face_encodings:
                            # 计算余弦相似度
                            similarity = np.dot(face_encoding, known_encoding) / (
                                np.linalg.norm(face_encoding) * np.linalg.norm(known_encoding))
                            # 转换为距离 (0-1范围，0表示完全相似)
                            distance = 1.0 - similarity
                            face_distances.append(distance)

                        face_distances = np.array(face_distances)
                        best_match_index = np.argmin(face_distances)

                        # 使用更严格的阈值
                        if face_distances[best_match_index] < self.tolerance:
                            name = self.known_face_names[best_match_index]
                            file_name = self.known_face_files[best_match_index]
                            confidence = 1 - face_distances[best_match_index]

                            # 增加高置信度检测，防止误识别
                            if confidence < 0.94:  # 提高置信度要求
                                # 再次尝试更严格的比对
                                second_encoding = face_recognition.face_encodings(
                                    rgb_img,
                                    face_locations,
                                    num_jitters=3,  # 增加抖动采样
                                    model="large"   # 使用大模型提高精度
                                )[0]

                                # 再次计算余弦相似度
                                second_distances = []
                                for known_encoding in self.known_face_encodings:
                                    similarity = np.dot(second_encoding, known_encoding) / (
                                        np.linalg.norm(second_encoding) * np.linalg.norm(known_encoding))
                                    distance = 1.0 - similarity
                                    second_distances.append(distance)

                                second_distances = np.array(second_distances)

                                if second_distances[best_match_index] < face_distances[best_match_index]:
                                    # 第二次识别更好，使用它
                                    confidence = 1 - second_distances[best_match_index]

                                # 置信度太低，可能是误识别，标记为未知
                                if confidence < 0.94:  # 严格要求置信度>0.94
                                    name = "Unknown"
                                    file_name = "unknown.json"
                                    confidence = 0

                # 创建结果
                result = {
                    "face_id": face_id,
                    "name": name,
                    "file_name": file_name if 'file_name' in locals() else "unknown.json",
                    "confidence": confidence,
                    "timestamp": time.time()
                }

                # 更新缓存
                self.recognition_cache[face_id] = result

                # 将结果放入结果队列
                self.results.put(result)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Face recognition worker thread error: {str(e)}")
                # 返回一个错误结果防止阻塞
                try:
                    self.results.put({
                        "face_id": -1,
                        "name": "Error",
                        "confidence": 0,
                        "timestamp": time.time()
                    })
                except:
                    pass

class FaceDetector(FrameProcessor):
    """人脸检测处理器"""
    def __init__(self, cascade_path=None):
        super().__init__("FaceDetector")

        # 使用OpenCV的级联分类器
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"  # 改用alt2版本
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # 使用更高效的检测参数
        self.scale_factor = 1.1  # 更低的缩放因子，提高检测率
        self.min_neighbors = 4   # 更多的最小邻居，减少误检
        self.min_size = (45, 45) # 更大的最小尺寸，过滤小误检

        logger.info("Using OpenCV cascade classifier (alt2) for face detection")

        # 增加自适应处理控制
        self.last_process_time = time.time()
        self.process_interval = 0.05  # 初始处理间隔 - 降低间隔提高响应速度
        self.load_factor = 0.0  # 系统负载因子

        # 跟踪上一帧检测到的人脸，用于稳定检测
        self.prev_faces = []
        self.face_stabilization_counter = 0

        # 检查是否可以使用GPU加速
        self.use_gpu = False
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.use_gpu = True
                self.gpu_cascade = cv2.cuda.CascadeClassifier_create(cascade_path)
                logger.info("Using CUDA acceleration for face detection")
            elif cv2.ocl.useOpenCL():
                self.use_gpu = True
                logger.info("Using OpenCL acceleration for face detection")
        except Exception as e:
            logger.warning(f"GPU acceleration not available for face detection: {str(e)}")

    def _process_loop(self):
        while self.running.is_set():
            try:
                # 获取帧
                frame = self.input_queue.get(block=True, timeout=0.1)

                # 自适应处理控制 - 根据处理时间动态调整间隔
                current_time = time.time()
                if current_time - self.last_process_time < self.process_interval:
                    # 跳过此帧处理，但传递前一帧的结果保持流畅度
                    if len(self.prev_faces) > 0:
                        self.output_queue.put({"frame": frame, "faces": self.prev_faces, "timestamp": current_time})
                    else:
                        self.output_queue.put({"frame": frame, "faces": [], "timestamp": current_time})
                    continue

                process_start = time.time()

                # 灰度转换以加速处理
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 对于高分辨率输入，使用更小的缩放因子
                scale_factor = 0.25  # 1920x1080 -> 480x270
                small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
                small_gray = cv2.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor)

                # 使用GPU加速进行人脸检测
                if self.use_gpu:
                    try:
                        if hasattr(self, 'gpu_cascade'):  # CUDA加速
                            # 转换为CUDA格式
                            gpu_small_gray = cv2.cuda_GpuMat()
                            gpu_small_gray.upload(small_gray)

                            # 使用CUDA进行检测
                            faces = self.gpu_cascade.detectMultiScale(gpu_small_gray)
                            faces = faces.download()
                        else:  # OpenCL加速
                            # 使用UMat进行OpenCL加速
                            small_gray_umat = cv2.UMat(small_gray)
                            faces = self.face_cascade.detectMultiScale(
                                small_gray_umat,
                                scaleFactor=self.scale_factor,
                                minNeighbors=self.min_neighbors,
                                minSize=self.min_size,
                                flags=cv2.CASCADE_SCALE_IMAGE
                            )
                    except Exception as e:
                        logger.warning(f"GPU detection failed, falling back to CPU: {str(e)}")
                        self.use_gpu = False
                        faces = self.face_cascade.detectMultiScale(
                            small_gray,
                            scaleFactor=self.scale_factor,
                            minNeighbors=self.min_neighbors,
                            minSize=self.min_size,
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )
                else:
                    # CPU检测
                    faces = self.face_cascade.detectMultiScale(
                        small_gray,
                        scaleFactor=self.scale_factor,
                        minNeighbors=self.min_neighbors,
                        minSize=self.min_size,
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

                # 将人脸坐标放大回原始尺寸
                if len(faces) > 0:
                    # 动态调整放大比例
                    scaled_faces = faces * (1.0 / scale_factor)
                    # 确保所有坐标都为整数
                    faces = scaled_faces.astype(int)

                # 稳定检测 - 在连续几帧都没有检测到人脸的情况下保持前一帧的结果
                if len(faces) > 0:
                    self.prev_faces = faces
                    self.face_stabilization_counter = 0
                else:
                    # 增加稳定计数
                    self.face_stabilization_counter += 1
                    # 如果不超过5帧没有检测到人脸，继续使用前一帧结果
                    if self.face_stabilization_counter < 5 and len(self.prev_faces) > 0:
                        faces = self.prev_faces
                    else:
                        # 超过阈值，清空前一帧结果
                        self.prev_faces = []

                # 测量处理时间并调整下次处理间隔
                process_time = time.time() - process_start
                self.load_factor = 0.8 * self.load_factor + 0.2 * process_time
                # 降低最长间隔时间以提高响应性
                self.process_interval = max(0.03, min(0.2, self.load_factor * 2))
                self.last_process_time = current_time

                # 输出检测结果和原始帧
                result = {"frame": frame, "faces": faces, "timestamp": time.time()}

                # 放入输出队列
                try:
                    if self.output_queue.full():
                        try:
                            self.output_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.output_queue.put(result, block=False)
                except Exception:
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Face detection error: {str(e)}")

class FaceRecognizer(FrameProcessor):
    """人脸识别处理器"""
    def _compare_faces(self, face_encoding):
        if len(self.known_face_encodings) == 0:
            return "Unknown", 0

        # 使用 SciPy 的空间距离计算
        face_distances = distance.cdist([face_encoding],
                                     self.known_face_encodings,
                                     metric='euclidean')[0]

        best_match_index = np.argmin(face_distances)
        min_distance = face_distances[best_match_index]

        if min_distance < self.tolerance:
            name = self.known_face_names[best_match_index]
            confidence = 1 - min_distance
        else:
            name = "Unknown"
            confidence = 0

        return name, confidence

    def _extract_face_features(self, face_img):
        """使用 SciPy 优化特征提取"""
        # 图像预处理
        enhanced = self._enhance_image(face_img)

        # 使用 SciPy 的形态学操作
        # 去除小的噪声区域
        mask = ndimage.binary_opening(enhanced > enhanced.mean(),
                                    structure=np.ones((3,3)))

        # 填充空洞
        mask = ndimage.binary_fill_holes(mask)

        # 应用掩码
        cleaned = enhanced * mask[..., np.newaxis]

        return cleaned

    def _adaptive_frame_rate(self):
        """使用 SciPy 的信号处理优化帧率"""
        # 使用移动平均平滑帧率
        window = signal.windows.hann(10)
        smoothed_fps = signal.convolve(self.fps_history, window, mode='same')

        # 自适应调整处理间隔
        target_fps = 30
        current_fps = np.mean(smoothed_fps[-5:])

        if current_fps < target_fps * 0.8:
            self.process_interval = max(0.01, self.process_interval * 0.9)
        elif current_fps > target_fps * 1.2:
            self.process_interval = min(0.1, self.process_interval * 1.1)

    def _enhance_image(self, frame):
        """使用 SciPy 增强图像质量"""
        # 中值滤波去噪
        denoised = ndimage.median_filter(frame, size=3)

        # 直方图均衡化
        yuv = cv2.cvtColor(denoised, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        # 锐化
        kernel = np.array([[-1,-1,-1],
                        [-1, 9,-1],
                        [-1,-1,-1]])
        sharpened = signal.convolve2d(enhanced, kernel, mode='same')

        return sharpened.astype(np.uint8)

    def __init__(self, face_encodings, face_names, tolerance=0.06):  # 设置极低的容忍度，要求置信度>0.94
        super().__init__("FaceRecognizer")
        self.known_face_encodings = face_encodings
        self.known_face_names = face_names
        self.tolerance = tolerance
        #添加缓存机制
        self.recognition_cache = {}  # 缓存识别结果
        self.cache_timeout = 5.0  # 缓存超时时间，单位秒

    def _process_frame(self, face_img):
        # 缩小图像加快处理
        small_face = cv2.resize(face_img, (0, 0), fx=0.5, fy=0.5)
        # 使用HOG模型加快检测
        face_locations = face_recognition.face_locations(small_face, model="hog")
        if not face_locations:
            return "Unknown", 0

        face_encoding = face_recognition.face_encodings(
            small_face,
            face_locations,
            num_jitters=1  # 减少抖动次数
        )[0]

        return self._compare_faces(face_encoding)

    def _process_loop(self):
        #预分配内存
        combined_frame = np.zeros((self.height, self.width*2, 3), dtype=np.uint8)
        resized_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        while self.running.is_set():
            try:
                frame = self.input_queue.get(block=True, timeout=0.1)

                # 应用图像增强
                enhanced_frame = self._enhance_image(frame)

                # 特征提取
                face_features = self._extract_face_features(enhanced_frame)

                # 自适应帧率控制
                self._adaptive_frame_rate()

                # 转换为RGB格式，用于face_recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 缩小图像以加快处理速度
                small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)

                # 准备face_recognition需要的位置格式
                face_locations = []
                for (x, y, w, h) in face_features:
                    top = int(y * 0.25)
                    right = int((x + w) * 0.25)
                    bottom = int((y + h) * 0.25)
                    left = int(x * 0.25)
                    face_locations.append((top, right, bottom, left))

                # 批量进行人脸编码
                face_encodings = face_recognition.face_encodings(small_frame, face_locations, num_jitters=0)

                # 对每个人脸进行识别
                for (x, y, w, h), face_encoding in zip(face_features, face_encodings):
                    name = "Unknown"
                    confidence = 0

                    if len(self.known_face_encodings) > 0:
                        # 计算与已知人脸的距离
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)

                        if face_distances[best_match_index] < self.tolerance:
                            name = self.known_face_names[best_match_index]
                            confidence = 1 - face_distances[best_match_index]

                    # 保存结果
                    face_locations.append({
                        "bbox": (x, y, w, h),
                        "name": name,
                        "confidence": confidence
                    })

                # 输出结果
                result = {
                    "frame": frame,
                    "faces": face_features,
                    "recognition_results": face_locations,
                    "timestamp": time.time()
                }

                # 放入输出队列
                try:
                    if self.output_queue.full():
                        try:
                            self.output_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.output_queue.put(result, block=False)
                except Exception:
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Face recognition error: {str(e)}")

class OptimizedFaceRecognizer(FrameProcessor):
    """优化后的人脸识别处理器，结合人脸追踪和异步识别，减少冗余识别处理"""
    def __init__(self, face_encodings, face_names, face_files=None, tolerance=0.06):  # 设置极低的容忍度，要求置信度>0.94
        super().__init__("OptimizedFaceRecognizer")
        self.known_face_encodings = face_encodings
        self.known_face_names = face_names
        self.known_face_files = face_files if face_files is not None else ["unknown.json"] * len(face_names)
        self.tolerance = tolerance
        self.face_tracker = FaceTracker()
        self.worker = FaceRecognitionWorker(face_encodings, face_names, face_files=self.known_face_files, tolerance=tolerance)
        # 添加识别稳定性控制
        self.recognition_results_history = {}  # 存储历史识别结果
        self.stability_threshold = 2  # 减少需要连续几帧保持相同结果才更新显示的阈值

    def _process_loop(self):
        while self.running.is_set():
            try:
                # 获取输入数据：帧、检测到的人脸及时间戳
                data = self.input_queue.get(block=True, timeout=0.1)
                frame = data["frame"]
                faces = data["faces"]

                # 更新人脸追踪信息
                tracked_faces = self.face_tracker.update([ (x, y, w, h) for (x, y, w, h) in faces ])

                # 对每个追踪到的人脸提交异步识别任务（防止频繁重复识别）
                for face_id, info in tracked_faces.items():
                    if info.get("need_recognition", True):
                        x, y, w, h = info["rect"]
                        # 确保所有坐标都是整数
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        # 确保坐标在图像范围内
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, frame.shape[1] - x)
                        h = min(h, frame.shape[0] - y)
                        # 确保宽高大于0
                        if w > 0 and h > 0:
                            face_img = frame[y:y+h, x:x+w]
                            submitted = self.worker.submit_task(face_id, face_img)
                            if submitted:
                                info["need_recognition"] = False  # 提交后临时关闭识别

                # 尝试获取异步识别结果
                result = self.worker.get_result()
                recognition_results = []

                # 处理新的识别结果
                if result:
                    face_id = result["face_id"]
                    face_name = result["name"]
                    confidence = result["confidence"]

                    # 如果该人脸在追踪中，更新识别历史
                    if face_id in tracked_faces:
                        # 更新识别历史
                        if face_id not in self.recognition_results_history:
                            self.recognition_results_history[face_id] = {
                                "name": face_name,
                                "file_name": result.get("file_name", "unknown.json"),
                                "confidence": confidence,
                                "stable_count": 1 if face_name != "Unknown" else 0,
                                "unstable_count": 1 if face_name == "Unknown" else 0
                            }
                        else:
                            history = self.recognition_results_history[face_id]

                            # 如果新识别的名称与历史相同，增加稳定计数
                            if face_name == history["name"]:
                                history["stable_count"] += 1
                                history["unstable_count"] = 0
                            else:
                                # 如果是未知切换到已知，立即接受高置信度的变化
                                if history["name"] == "Unknown" and face_name != "Unknown":
                                    # 降低置信度阈值，更快地接受变化
                                    if confidence > 0.85:  # 降低阈值从0.90到0.85
                                        history["name"] = face_name
                                        history["file_name"] = result.get("file_name", "unknown.json")
                                        history["confidence"] = confidence
                                        history["stable_count"] = 2
                                        history["unstable_count"] = 0
                                        # 立即重置识别状态，允许下一次识别
                                        tracked_faces[face_id]["need_recognition"] = True
                                        logger.debug(f"人脸ID {face_id} 从未知变为已知: {face_name}, 置信度: {confidence:.2f}")
                                # 如果是已知变成未知，更谨慎地接受变化
                                elif history["name"] != "Unknown" and face_name == "Unknown":
                                    history["unstable_count"] += 1
                                    # 需要更多的未知帧才会切换，避免误识别
                                    if history["unstable_count"] > 2:  # 从3减少到2，但仍保持一定稳定性
                                        history["name"] = face_name
                                        history["file_name"] = "unknown.json"
                                        history["confidence"] = confidence
                                        history["stable_count"] = 0
                                        # 立即重置识别状态，允许下一次识别
                                        tracked_faces[face_id]["need_recognition"] = True
                                        logger.debug(f"人脸ID {face_id} 从已知变为未知, 不稳定计数: {history['unstable_count']}")
                                # 已知人脸之间的切换，更快地接受高置信度的变化
                                else:
                                    history["unstable_count"] += 1
                                    # 降低需要的帧数和置信度阈值，更快地接受变化
                                    if ((history["unstable_count"] > 1 and confidence > 0.90) or  # 高置信度情况下只需2帧
                                       (history["unstable_count"] > 2 and confidence > 0.85)):    # 中等置信度需要3帧
                                        history["name"] = face_name
                                        history["file_name"] = result.get("file_name", "unknown.json")
                                        history["confidence"] = confidence
                                        history["stable_count"] = 1
                                        history["unstable_count"] = 0
                                        # 立即重置识别状态，允许下一次识别
                                        tracked_faces[face_id]["need_recognition"] = True
                                        logger.debug(f"人脸ID {face_id} 从 {history['name']} 变为 {face_name}, 置信度: {confidence:.2f}")

                        # 允许下一次识别
                        tracked_faces[face_id]["need_recognition"] = True
                        bbox = tracked_faces[face_id]["rect"]

                        # 使用稳定的结果
                        stable_result = self.recognition_results_history[face_id]
                        recognition_results.append({
                            "bbox": bbox,
                            "name": stable_result["name"],
                            "file_name": stable_result.get("file_name", "unknown.json"),
                            "confidence": stable_result["confidence"]
                        })
                    else:
                        # 如果人脸已经不在追踪中，创建临时结果
                        bbox = (0, 0, 0, 0)
                        recognition_results.append({
                            "bbox": bbox,
                            "name": face_name,
                            "file_name": result.get("file_name", "unknown.json"),
                            "confidence": confidence
                        })

                # 添加所有正在追踪但没有新识别结果的人脸
                for face_id, info in tracked_faces.items():
                    # 检查这个face_id是否已经在recognition_results中
                    if not any(face_id == result.get("face_id", -1) for result in recognition_results):
                        # 如果这个人脸有历史识别结果，使用它
                        if face_id in self.recognition_results_history:
                            stable_result = self.recognition_results_history[face_id]
                            recognition_results.append({
                                "bbox": info["rect"],
                                "name": stable_result["name"],
                                "file_name": stable_result.get("file_name", "unknown.json"),
                                "confidence": stable_result["confidence"]
                            })

                # 清理已不再追踪的人脸历史
                face_ids_to_remove = []
                for face_id in self.recognition_results_history:
                    if face_id not in tracked_faces:
                        face_ids_to_remove.append(face_id)

                for face_id in face_ids_to_remove:
                    del self.recognition_results_history[face_id]

                # 控制 GPIO 状态
                known_face_detected = False
                unknown_face_detected = False

                # 检查是否有已知人脸或未知人脸
                for result in recognition_results:
                    if result["name"] != "Unknown":
                        known_face_detected = True

                        # 如果启用了数据库功能，记录打卡信息
                        if DB_ENABLED:
                            try:
                                # 确定摄像头来源
                                # 通过分析帧的位置判断是哪个摄像头
                                # 假设帧是水平拼接的，左侧是入口摄像头(cam0)，右侧是出口摄像头(cam1)
                                bbox = result["bbox"]
                                frame_width = frame.shape[1]
                                camera_source = "cam0"  # 默认为入口摄像头

                                # 如果人脸框的中心点在右半部分，则认为是出口摄像头
                                face_center_x = bbox[0] + bbox[2] // 2
                                if face_center_x > frame_width // 2:
                                    camera_source = "cam1"  # 出口摄像头

                                # 获取员工ID
                                employee_id = db_connector.get_employee_id_by_name(result["name"])

                                if employee_id:
                                    # 根据摄像头确定打卡类型
                                    # cam0（入口摄像头）对应"in"（上班打卡）
                                    # cam1（出口摄像头）对应"out"（下班打卡）
                                    session_type = "in" if camera_source == "cam0" else "out"

                                    # 记录打卡信息
                                    success, message = db_connector.record_attendance(
                                        employee_id, camera_source, session_type)

                                    if success:
                                        logger.info(f"Successfully recorded attendance for {result['name']} (ID: {employee_id}) at {camera_source}")
                                    else:
                                        # 如果是重复打卡，输出更详细的日志以便调试
                                        if message == "repeat_attendance":
                                            # 每10秒只输出一次重复打卡的日志
                                            current_time = time.time()
                                            if not hasattr(self, '_last_repeat_log_time') or current_time - self._last_repeat_log_time > 10:
                                                logger.info(f"Repeat attendance ignored: {result['name']} (ID: {employee_id}) at {camera_source}")
                                                self._last_repeat_log_time = current_time
                                        else:
                                            logger.info(f"Failed to record attendance: {message}")
                                else:
                                    logger.warning(f"Could not find employee ID for {result['name']}")
                            except Exception as e:
                                logger.error(f"Error recording attendance: {str(e)}")
                        break
                    else:
                        unknown_face_detected = True

                # 根据识别结果控制 GPIO 状态
                if known_face_detected:
                    # 状态C：识别到已知人脸，GPIO 14持续亮起直到人脸消失
                    # 减少日志输出，只在状态变化时记录
                    set_gpio_state(recognized=True, unknown=False)
                elif unknown_face_detected:
                    # 状态B：检测到未知人脸，GPIO 15持续亮起直到人脸消失
                    # 减少日志输出，只在状态变化时记录
                    set_gpio_state(recognized=False, unknown=True)
                elif len(faces) == 0:
                    # 无人脸状态，3秒后返回待机状态（GPIO 18常亮）
                    # 减少日志输出，只在状态变化时记录
                    set_gpio_state(recognized=False, unknown=False)

                # 构造输出结果
                output = {
                    "frame": frame,
                    "faces": faces,
                    "recognition_results": recognition_results,
                    "timestamp": time.time()
                }
                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.output_queue.put(output, block=False)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Optimized face recognition processing error: {str(e)}")

class ResultRenderer(FrameProcessor):
    """结果渲染器"""
    def __init__(self, camera_width, camera_height, display_fps=True):
        super().__init__("ResultRenderer")
        self.width = camera_width
        self.height = camera_height
        self.display_fps = display_fps
        self.last_fps_time = time.time()
        self.fps = 0
        self.frame_count = 0
        # 添加渲染稳定性控制
        self.prev_results = {}  # 存储之前的渲染结果，用于平滑过渡

    def _get_font_scale_from_pixels(self, font, text, desired_height_px, thickness=1):
        """将像素高度转换为OpenCV的字体缩放因子"""
        # 先以缩放因子1.0计算文本大小
        base_scale = 1.0
        (_, base_height), _ = cv2.getTextSize(text, font, base_scale, thickness)

        if base_height == 0:  # 防止除以零
            return base_scale

        # 计算所需的缩放因子
        return desired_height_px / base_height

    def _process_loop(self):
        while self.running.is_set():
            try:
                # 获取识别结果
                data = self.input_queue.get(block=True, timeout=0.1)

                frame = data["frame"]
                recognition_results = data.get("recognition_results", [])

                # 更新FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time > 1.0:  # 每秒更新一次
                    self.fps = self.frame_count / (current_time - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = current_time

                # 字体设置
                font = cv2.FONT_HERSHEY_SIMPLEX

                # 参考test2.py的渲染方式，更直接清晰地显示人脸框和标签
                for result in recognition_results:
                    x, y, w, h = result["bbox"]
                    # 确保所有坐标都是整数
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    name = result["name"]
                    confidence = result["confidence"]

                    # 使用更明显的颜色和线宽
                    if name != "Unknown":
                        # 已知人脸用绿色
                        color = (0, 255, 0)
                        thickness = 3  # 增加线宽
                    else:
                        # 未知人脸用红色
                        color = (0, 0, 255)
                        thickness = 2  # 增加线宽

                    # 确保坐标在图像范围内
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, frame.shape[1] - x)
                    h = min(h, frame.shape[0] - y)

                    # 确保宽高大于0
                    if w > 0 and h > 0:
                        # 画框 - 使用更清晰的矩形
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

                    # 只显示名称和置信度 - 使用带黑边的文字
                    label = f"{name} ({confidence:.2f})"

                    # 以像素单位指定字体大小 - 姓名标签使用30像素高
                    font_thickness = 2
                    desired_height_px = 30
                    font_size = self._get_font_scale_from_pixels(font, label, desired_height_px, font_thickness)

                    # 计算文本大小以便定位
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, font, font_size, font_thickness)

                    # 文字位置
                    text_x = x
                    text_y = max(y - 10, 10)  # 确保文字不会超出图像上边界

                    # 先绘制黑色描边
                    for offset_x in [-2, 0, 2]:
                        for offset_y in [-2, 0, 2]:
                            if offset_x == 0 and offset_y == 0:
                                continue  # 跳过中心点
                            cv2.putText(
                                frame,
                                label,
                                (text_x + offset_x, text_y + offset_y),
                                font,
                                font_size,
                                (0, 0, 0),  # 黑色边缘
                                font_thickness + 1
                            )

                    # 再绘制原色文字
                    cv2.putText(
                        frame,
                        label,
                        (text_x, text_y),
                        font,
                        font_size,
                        (255, 255, 255),  # 白色文字
                        font_thickness
                    )

                # 显示提示和FPS - 添加黑色边缘
                instruction_text = "Press 'q' to exit"
                text_x, text_y = 10, 40

                # 以像素单位指定字体大小 - 指令文本使用24像素高
                font_thickness = 2
                desired_height_px = 24
                font_size = self._get_font_scale_from_pixels(font, instruction_text, desired_height_px, font_thickness)

                # 先绘制黑色描边
                for offset_x in [-2, 0, 2]:
                    for offset_y in [-2, 0, 2]:
                        if offset_x == 0 and offset_y == 0:
                            continue
                        cv2.putText(
                            frame,
                            instruction_text,
                            (text_x + offset_x, text_y + offset_y),
                            font,
                            font_size,
                            (0, 0, 0),  # 黑色边缘
                            font_thickness + 1
                        )

                # 再绘制紫色文字
                cv2.putText(
                    frame,
                    instruction_text,
                    (text_x, text_y),
                    font,
                    font_size,
                    (255, 0, 255),  # 紫色文字
                    font_thickness
                )

                if self.display_fps:
                    fps_text = f"FPS: {self.fps:.1f}"
                    text_x, text_y = 10, 80

                    # 以像素单位指定字体大小 - FPS文本使用24像素高
                    desired_height_px = 24
                    font_size = self._get_font_scale_from_pixels(font, fps_text, desired_height_px, font_thickness)

                    # 先绘制黑色描边
                    for offset_x in [-2, 0, 2]:
                        for offset_y in [-2, 0, 2]:
                            if offset_x == 0 and offset_y == 0:
                                continue
                            cv2.putText(
                                frame,
                                fps_text,
                                (text_x + offset_x, text_y + offset_y),
                                font,
                                font_size,
                                (0, 0, 0),  # 黑色边缘
                                font_thickness + 1
                            )

                    # 再绘制绿色文字
                    cv2.putText(
                        frame,
                        fps_text,
                        (text_x, text_y),
                        font,
                        font_size,
                        (0, 255, 0),  # 绿色文字
                        font_thickness
                    )

                # 放入输出队列
                try:
                    if self.output_queue.full():
                        try:
                            self.output_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.output_queue.put(frame, block=False)
                except Exception:
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Result rendering error: {str(e)}")

class FrameSourceManager:
    """优化版帧源管理器"""
    def __init__(self, width, height, frame_rate=30):  # 进一步提高帧率
        # 降低分辨率以提升性能
        self.width = 1280  # 从1920降至1280
        self.height = 720  # 从1080降至720
        self.frame_rate = frame_rate
        self.display_width = width
        self.display_height = height
        self.frame = None
        self.frame_lock = Lock()
        self.running = Event()
        self.running.set()
        self.subscribers = []
        self.picam2 = None
        self.picam2_1 = None  # 入口摄像头
        self.picam2_2 = None  # 出口摄像头

        # 添加帧缓冲区，减少丢帧
        self.frame_buffer_size = 3
        self.frame_buffer = queue.Queue(maxsize=self.frame_buffer_size)

        # 添加性能监控
        self.fps_history = []
        self.last_fps_time = time.time()
        self.frame_count = 0

    def initialize(self):
        """初始化双摄像头"""
        try:
            logger.info("Initializing entrance and exit cameras...")

            # 初始化入口摄像头
            self.picam2_1 = Picamera2(0)  # 入口摄像头
            config1 = self.picam2_1.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls={"FrameRate": self.frame_rate,
                         "AwbEnable": 1,
                         "NoiseReductionMode": 1}
            )
            self.picam2_1.configure(config1)
            self.picam2_1.start()

            # 初始化出口摄像头
            self.picam2_2 = Picamera2(1)  # 出口摄像头
            config2 = self.picam2_2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls={"FrameRate": self.frame_rate,
                         "AwbEnable": 1,
                         "NoiseReductionMode": 1}
            )
            self.picam2_2.configure(config2)
            self.picam2_2.start()

            # 测试两个摄像头
            test_frame1 = self.picam2_1.capture_array()
            test_frame2 = self.picam2_2.capture_array()

            if test_frame1 is None or test_frame2 is None:
                raise ValueError("Unable to get test frames")

            logger.info("Entrance and exit cameras initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Camera initialization failed: {str(e)}")
            return False

    def start_capture(self):
        """启动帧捕获线程"""
        self.capture_thread = Thread(target=self._capture_loop, name="FrameCapture")
        self.capture_thread.daemon = True
        self.capture_thread.start()

        self.distribution_thread = Thread(target=self._distribution_loop, name="FrameDistribution")
        self.distribution_thread.daemon = True
        self.distribution_thread.start()

    def subscribe(self, processor):
        """添加帧订阅者"""
        self.subscribers.append(processor)

    def _capture_loop(self):
        """捕获帧循环"""
        logger.info("Starting frame capture thread")
        frame_count = 0
        last_fps_update = time.time()

        # 定义标记文字的样式
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0  # 增大字体
        thickness = 5     # 增加文字粗细
        text_color = (255, 255, 255)  # 白色文字
        border_color = (0, 0, 0)      # 黑色边框

        while self.running.is_set():
            try:
                # 同时从入口和出口摄像头捕获帧
                frame1 = self.picam2_1.capture_array()  # 入口画面
                frame2 = self.picam2_2.capture_array()  # 出口画面

                if frame1 is None or frame2 is None:
                    time.sleep(0.01)
                    continue

                # 处理格式
                if len(frame1.shape) == 3 and frame1.shape[2] == 4:
                    frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGBA2BGR)
                elif len(frame1.shape) == 3 and frame1.shape[2] == 3:
                    frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)

                if len(frame2.shape) == 3 and frame2.shape[2] == 4:
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGBA2BGR)
                elif len(frame2.shape) == 3 and frame2.shape[2] == 3:
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)

                # 将出口摄像头(cam1)的画面旋转180度
                # 由于亚克力设计限制，需要翻转cam1的输出以正确显示人脸
                frame2 = cv2.rotate(frame2, cv2.ROTATE_180)

                # 调整两个画面大小，保持原始宽高比，但更大一些
                # 计算缩放因子，保持宽高比，目标宽度为960像素
                scale_factor = min(960 / frame1.shape[1], 720 / frame1.shape[0])
                new_width = int(frame1.shape[1] * scale_factor)
                new_height = int(frame1.shape[0] * scale_factor)

                # 调整大小，保持宽高比
                frame1 = cv2.resize(frame1, (new_width, new_height))
                frame2 = cv2.resize(frame2, (new_width, new_height))

                # 创建960x720的画布，居中放置调整后的图像
                canvas1 = np.zeros((720, 960, 3), dtype=np.uint8)
                canvas2 = np.zeros((720, 960, 3), dtype=np.uint8)

                # 计算偏移量以居中放置（确保是整数）
                x_offset = int((960 - new_width) // 2)
                y_offset = int((720 - new_height) // 2)

                # 将调整后的图像放置在画布中央
                # 确保所有索引都是整数
                canvas1[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame1
                canvas2[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame2

                # 使用画布作为新的帧
                frame1 = canvas1
                frame2 = canvas2

                # 在画面上添加醒目的标识文字，使用更美观的样式
                # 入口标记 - 使用绿色渐变
                text1 = "IN"
                (text_width1, text_height1), _ = cv2.getTextSize(text1, font, font_scale, thickness)

                # 计算文字位置（中间下方）
                x1 = int((960 - text_width1) // 2)
                y1 = int(720 - 50)  # 距离底部50像素

                # 创建半透明背景 - 入口标记
                bg_rect_x1 = x1 - 20
                bg_rect_y1 = y1 - text_height1 - 10
                bg_rect_w1 = text_width1 + 40
                bg_rect_h1 = text_height1 + 20

                # 创建一个与原图相同大小的透明图层
                overlay1 = frame1.copy()

                # 绘制半透明背景 - 入口标记（使用深绿色背景）
                cv2.rectangle(overlay1,
                             (bg_rect_x1, bg_rect_y1),
                             (bg_rect_x1 + bg_rect_w1, bg_rect_y1 + bg_rect_h1),
                             (0, 80, 0), -1)  # 深绿色背景

                # 添加半透明效果
                alpha = 0.7  # 透明度
                cv2.addWeighted(overlay1, alpha, frame1, 1 - alpha, 0, frame1)

                # 绘制文本边框 - 入口标记
                cv2.rectangle(frame1,
                             (bg_rect_x1, bg_rect_y1),
                             (bg_rect_x1 + bg_rect_w1, bg_rect_y1 + bg_rect_h1),
                             (0, 255, 0), 2)  # 绿色边框

                # 绘制入口标记文字
                cv2.putText(frame1, text1, (x1, y1), font, font_scale, (255, 255, 255), thickness)

                # 出口标记 - 使用红色渐变
                text2 = "OUT"
                (text_width2, text_height2), _ = cv2.getTextSize(text2, font, font_scale, thickness)

                # 计算文字位置（中间下方）
                x2 = int((960 - text_width2) // 2)
                y2 = int(720 - 50)  # 距离底部50像素

                # 创建半透明背景 - 出口标记
                bg_rect_x2 = x2 - 20
                bg_rect_y2 = y2 - text_height2 - 10
                bg_rect_w2 = text_width2 + 40
                bg_rect_h2 = text_height2 + 20

                # 创建一个与原图相同大小的透明图层
                overlay2 = frame2.copy()

                # 绘制半透明背景 - 出口标记（使用深红色背景）
                cv2.rectangle(overlay2,
                             (bg_rect_x2, bg_rect_y2),
                             (bg_rect_x2 + bg_rect_w2, bg_rect_y2 + bg_rect_h2),
                             (0, 0, 80), -1)  # 深红色背景

                # 添加半透明效果
                cv2.addWeighted(overlay2, alpha, frame2, 1 - alpha, 0, frame2)

                # 绘制文本边框 - 出口标记
                cv2.rectangle(frame2,
                             (bg_rect_x2, bg_rect_y2),
                             (bg_rect_x2 + bg_rect_w2, bg_rect_y2 + bg_rect_h2),
                             (0, 0, 255), 2)  # 红色边框

                # 绘制出口标记文字
                cv2.putText(frame2, text2, (x2, y2), font, font_scale, (255, 255, 255), thickness)

                # 水平拼接两个画面
                combined_frame = np.hstack((frame1, frame2))

                # 更新帧
                with self.frame_lock:
                    self.frame = combined_frame.copy()

                # 计算帧率
                frame_count += 1
                current_time = time.time()
                if current_time - last_fps_update >= 5.0:  # 每5秒报告一次
                    fps = frame_count / (current_time - last_fps_update)
                    logger.info(f"Frame capture rate: {fps:.1f} FPS")
                    frame_count = 0
                    last_fps_update = current_time

                # 控制帧率
                time.sleep(1.0 / self.frame_rate)

            except Exception as e:
                logger.error(f"Frame capture error: {str(e)}")
                time.sleep(0.01)

    def _distribution_loop(self):
        """帧分发循环"""
        logger.info("Starting frame distribution thread")

        while self.running.is_set():
            try:
                # 获取当前帧
                with self.frame_lock:
                    if self.frame is None:
                        time.sleep(0.01)
                        continue
                    current_frame = self.frame.copy()

                # 分发帧给所有订阅者
                for processor in self.subscribers:
                    processor.put_frame(current_frame)

                # 控制帧分发速率
                time.sleep(0.01)  # 约100FPS分发速率

            except Exception as e:
                logger.error(f"Frame distribution error: {str(e)}")
                time.sleep(0.01)

    def stop(self):
        """停止所有线程并释放资源"""
        self.running.clear()
        if hasattr(self, 'picam2_1'):
            self.picam2_1.close()
        if hasattr(self, 'picam2_2'):
            self.picam2_2.close()

class FaceRecognitionSystem:
    def __init__(self):
        # 系统参数
        self.LOWER_RESOLUTION = False  # 改为显示高分辨率
        self.DISPLAY_FPS = True
        self.CAMERA_WIDTH = 1920 if not self.LOWER_RESOLUTION else 960  # 增加宽度
        self.CAMERA_HEIGHT = 1080 if not self.LOWER_RESOLUTION else 540  # 调整高度保持16:9比例
        self.RECOGNITION_INTERVAL = 3  # 增加识别间隔以减少处理负担
        self.TOLERANCE = 0.06  # 设置极低的容忍度，要求置信度>0.94
        # 添加线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=350)

        # 调整人脸检测缩放比例，适应高分辨率
        self.FACE_DETECTION_SCALE = 0.25  # 以1920x1080输入，缩放到480x270进行检测

        # 系统状态
        self.running_event = Event()
        self.running_event.set()

        # 人脸数据
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_files = []  # 存储对应的文件名

        # 处理管道
        self.frame_count = 0
        self.last_recognition_frame = 0

        # 处理器组件
        self.frame_source = None
        self.face_detector = None
        self.face_recognizer = None
        self.result_renderer = None

        # GPU加速配置
        self.use_gpu = False
        self.gpu_backend = None
        self.gpu_target = None

        # Web 服务器配置
        self.web_server_thread = None
        self.web_enabled = WEB_ENABLED
        self.web_host = '0.0.0.0'  # 允许从任何 IP 访问
        self.web_port = 5000       # 默认端口

        # 初始化数据库连接
        if DB_ENABLED:
            try:
                if db_connector.connect():
                    logger.info("Successfully connected to database")
                else:
                    logger.warning("Unable to connect to database, attendance recording will be disabled")
            except Exception as e:
                logger.error(f"Database connection initialization error: {str(e)}")

        # 初始化系统监控器
        self.system_monitor = SystemMonitor(interval=60)  # 每60秒记录一次系统性能数据

    def process_faces(self, frame, faces):
        """并行处理多个人脸"""
        futures = []
        for face in faces:
            future = self.thread_pool.submit(self._process_single_face, frame, face)
            futures.append(future)
        return [f.result() for f in futures]

    def load_known_faces(self, encoding_dir: str = "Encoding_DataSet"):
        """加载已知人脸 - 只从JSON文件加载"""
        try:
            # 从Encoding_DataSet加载JSON文件
            json_count = self._load_faces_from_json(encoding_dir)

            logger.info(f"Total loaded faces from JSON: {json_count}")

            # 如果没有找到人脸，添加测试数据
            if json_count == 0:
                test_encoding = np.random.rand(128)
                self.known_face_encodings.append(test_encoding)
                self.known_face_names.append("Test User")
                self.known_face_files.append("test_user.json")
                logger.info("Added test user data")

            return True
        except Exception as e:
            logger.error(f"Failed to load face data: {str(e)}")
            return False

    def _load_faces_from_json(self, encoding_dir: str):
        """从JSON文件加载人脸特征"""
        count = 0
        try:
            logger.info(f"Loading face encodings from JSON: {encoding_dir}")
            if not os.path.exists(encoding_dir):
                logger.warning(f"Encoding directory does not exist: {encoding_dir}")
                return count

            # 获取所有JSON文件
            json_files = [f for f in os.listdir(encoding_dir) if f.lower().endswith('.json')]
            logger.info(f"Found {len(json_files)} JSON files in {encoding_dir}")

            # 处理每个JSON文件
            for json_file in json_files:
                json_path = os.path.join(encoding_dir, json_file)
                logger.info(f"Loading JSON file: {json_file}")

                try:
                    # 加载JSON文件
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    # 检查JSON结构
                    if isinstance(data, list):
                        # 列表格式：包含多个人
                        for person in data:
                            name = person.get('name', os.path.splitext(json_file)[0])
                            features = person.get('features', [])

                            if features:
                                # 添加每个特征向量
                                for feature in features:
                                    if isinstance(feature, list) and len(feature) > 0:
                                        self.known_face_encodings.append(np.array(feature, dtype=np.float32))
                                        self.known_face_names.append(name)
                                        self.known_face_files.append(json_file)
                                        count += 1
                    elif isinstance(data, dict):
                        # 字典格式：单个人
                        name = data.get('name', os.path.splitext(json_file)[0])

                        # 尝试不同的键名
                        features = []
                        if 'encodings' in data and data['encodings']:
                            features = data['encodings']
                        elif 'features' in data and data['features']:
                            features = data['features']

                        for feature in features:
                            if isinstance(feature, list) and len(feature) > 0:
                                self.known_face_encodings.append(np.array(feature, dtype=np.float32))
                                self.known_face_names.append(name)
                                self.known_face_files.append(json_file)
                                count += 1

                    logger.info(f"Loaded features from {json_file}, current total: {count}")

                except Exception as e:
                    logger.error(f"Error loading JSON file {json_file}: {str(e)}")

            return count
        except Exception as e:
            logger.error(f"Error loading JSON encodings: {str(e)}")
            return count

    def _load_faces_from_images(self, faces_dir: str):
        """从图像文件加载人脸特征"""
        count = 0
        try:
            logger.info(f"Loading face data from images: {faces_dir}")
            if not os.path.exists(faces_dir):
                logger.warning(f"Face directory does not exist, creating: {faces_dir}")
                os.makedirs(faces_dir, exist_ok=True)
                return count

            # 递归加载所有子文件夹中的图片
            def load_images_from_dir(directory, parent_name=None):
                nonlocal count
                for item in os.listdir(directory):
                    item_path = os.path.join(directory, item)

                    # 如果是子文件夹，递归处理
                    if os.path.isdir(item_path):
                        folder_name = os.path.basename(item_path)
                        logger.info(f"Processing subfolder: {folder_name}")
                        load_images_from_dir(item_path, folder_name)

                    # 如果是图片文件，处理图片
                    elif item.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # 使用文件名作为人名，如果在子文件夹中，则使用文件夹名
                        if parent_name:
                            name = parent_name
                        else:
                            name = os.path.splitext(item)[0]

                        try:
                            image = face_recognition.load_image_file(item_path)
                            # 缩小图像加快编码
                            small_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                            encodings = face_recognition.face_encodings(small_image)

                            if encodings:
                                self.known_face_encodings.append(encodings[0])
                                self.known_face_names.append(name)
                                self.known_face_files.append(item)  # 存储图像文件名
                                logger.info(f"Loaded face: {name} from {item_path}")
                                count += 1
                            else:
                                logger.warning(f"No face detected: {item_path}")
                        except Exception as e:
                            logger.error(f"Error loading face {item_path}: {str(e)}")

            # 开始递归加载
            load_images_from_dir(faces_dir)
            return count

        except Exception as e:
            logger.error(f"Failed to load face data from images: {str(e)}")
            return count

    def initialize_system(self):
        """初始化整个系统"""
        try:
            # 使用预加载的picamera配置
            time_start = time.time()
            logger.info("Starting system initialization...")

            # 创建帧源管理器 - 使用预优化配置
            self.frame_source = FrameSourceManager(
                self.CAMERA_WIDTH,
                self.CAMERA_HEIGHT,
                frame_rate=20
            )

            # 预初始化摄像头
            if not self.frame_source.initialize():
                return False

            # 创建人脸检测器 - 使用OpenCV级联分类器
            self.face_detector = FaceDetector()

            # 使用优化的人脸识别器
            self.face_recognizer = OptimizedFaceRecognizer(
                self.known_face_encodings,
                self.known_face_names,
                self.known_face_files,
                tolerance=self.TOLERANCE
            )

            # 创建结果渲染器
            self.result_renderer = ResultRenderer(
                self.CAMERA_WIDTH,
                self.CAMERA_HEIGHT,
                display_fps=self.DISPLAY_FPS
            )

            # 初始化 Web 服务器
            if self.web_enabled:
                self._start_web_server()

            # 启动系统监控器
            self.system_monitor.start()
            logger.info(f"System monitor started. Log file: {self.system_monitor.log_file}")

            logger.info(f"System initialization complete, time taken: {time.time() - time_start:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            return False

    def _start_web_server(self):
        """启动 Web 服务器线程"""
        if not self.web_enabled:
            logger.warning("Web server is disabled")
            return

        try:
            logger.info("Starting web server...")
            # 创建并启动 Web 服务器线程
            self.web_server_thread = Thread(
                target=web_server.start_server,
                args=(self.web_host, self.web_port),
                daemon=True,
                name="WebServerThread"
            )
            self.web_server_thread.start()
            logger.info(f"Web server started on http://{self.web_host}:{self.web_port}/")

            # 显示访问信息
            if self.web_host == '0.0.0.0':
                import socket
                try:
                    # 获取本机 IP 地址
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(("8.8.8.8", 80))
                    local_ip = s.getsockname()[0]
                    s.close()
                    logger.info(f"Web interface available at: http://{local_ip}:{self.web_port}/")
                except:
                    pass
        except Exception as e:
            logger.error(f"Failed to start web server: {str(e)}")
            self.web_enabled = False

    def start_processing(self):
        """启动所有处理线程"""
        # 启动各处理器
        self.face_detector.start()
        self.face_recognizer.start()
        self.result_renderer.start()

        # 设置处理管道连接
        # 帧源 -> 人脸检测器
        self.frame_source.subscribe(self.face_detector)

        # 启动帧源管理器
        self.frame_source.start_capture()

        # 启动管道连接线程
        self.pipeline_thread = Thread(target=self._pipeline_connector, name="PipelineConnector")
        self.pipeline_thread.daemon = True
        self.pipeline_thread.start()

    def _pipeline_connector(self):
        """管道连接器：在各处理器之间传递数据"""
        logger.info("Starting pipeline connector")

        while self.running_event.is_set():
            try:
                # 从人脸检测器获取结果，传给人脸识别器
                detection_result = self.face_detector.get_result()
                if detection_result:
                    self.face_recognizer.put_frame(detection_result)

                # 从人脸识别器获取结果，传给结果渲染器
                recognition_result = self.face_recognizer.get_result()
                if recognition_result:
                    self.result_renderer.put_frame(recognition_result)

                # 短暂休眠以避免CPU占用过高
                time.sleep(0.001)

            except Exception as e:
                logger.error(f"Pipeline connection error: {str(e)}")

    def display_loop(self):
        """显示循环"""
        logger.info("Starting display loop")

        # 创建窗口
        cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Recognition", 1920, 720)  # 调整显示尺寸为1920x720（左右布局）

        while self.running_event.is_set():
            try:
                # 获取渲染后的帧
                frame = self.result_renderer.get_result()

                if frame is not None:
                    # 显示帧
                    cv2.imshow("Face Recognition", frame)

                    # 如果 Web 服务器已启用，将帧发送到 Web 服务器
                    if self.web_enabled:
                        try:
                            web_server.set_frame(frame)
                        except Exception as e:
                            logger.error(f"Error sending frame to web server: {str(e)}")

                # 检查键盘输入
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    logger.info("User pressed 'q', exiting")
                    break

            except Exception as e:
                logger.error(f"Display loop error: {str(e)}")

        # 关闭窗口
        cv2.destroyAllWindows()
        # 确保窗口关闭
        for i in range(3):
            cv2.waitKey(1)

    def stop_system(self):
        """停止并清理系统"""
        logger.info("Stopping system...")

        # 停止所有组件
        self.running_event.clear()

        if self.face_detector:
            self.face_detector.stop()

        if self.face_recognizer:
            self.face_recognizer.stop()

        if self.result_renderer:
            self.result_renderer.stop()

        if self.frame_source:
            self.frame_source.stop()

        # Web 服务器会自动关闭（因为是守护线程）
        if self.web_enabled and self.web_server_thread and self.web_server_thread.is_alive():
            logger.info("Web server will shut down automatically")

        # 关闭数据库连接
        if DB_ENABLED:
            try:
                db_connector.disconnect()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error disconnecting from database: {str(e)}")

        # 停止系统监控器
        try:
            if hasattr(self, 'system_monitor'):
                self.system_monitor.stop()
                logger.info("System monitor stopped")
        except Exception as e:
            logger.error(f"Error stopping system monitor: {str(e)}")

        logger.info("System stopped")

    def run(self):
        """运行系统"""
        try:
            # 只从Encoding_DataSet加载人脸数据
            if not self.load_known_faces(encoding_dir="Encoding_DataSet"):
                return False

            # 初始化系统
            if not self.initialize_system():
                return False

            # 启动处理
            self.start_processing()

            # 开始显示循环
            self.display_loop()

            return True
        except Exception as e:
            logger.error(f"System runtime error: {str(e)}")
            return False
        finally:
            # 停止系统
            self.stop_system()

def main():
    """主函数"""
    try:
        # 输出系统信息
        logger.info(f"OpenCV version: {cv2.__version__}")
        logger.info(f"Python version: {os.sys.version}")
        logger.info(f"System monitor logs will be saved to: {os.path.abspath('logs/system_monitor_' + datetime.datetime.now().strftime('%Y-%m-%d') + '.log')}")

        # 确保Encoding_DataSet目录存在
        if not os.path.exists("Encoding_DataSet"):
            logger.info("Creating Encoding_DataSet directory")
            os.makedirs("Encoding_DataSet")

        # 创建并运行系统
        system = FaceRecognitionSystem()
        if not system.run():
            logger.error("System run failed")

    except Exception as e:
        logger.critical(f"Program encountered a critical error: {str(e)}")
        # 显示错误窗口
        error_img = np.zeros((240, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, f"Program error: {str(e)}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Error", error_img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        print(f"Program encountered a critical error: {str(e)}")
    finally:
        # 清理 GPIO 资源
        try:
            logger.info("Cleaning up GPIO resources...")
            if cleanup_gpio():
                logger.info("GPIO resources successfully cleaned up")
            else:
                logger.info("GPIO cleanup completed (no resources needed cleaning)")
        except Exception as e:
            logger.error(f"Error cleaning up GPIO resources: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # 处理Ctrl+C中断
        logger.info("Program terminated by user")
        # 注意：GPIO资源已经在main()函数的finally块中清理