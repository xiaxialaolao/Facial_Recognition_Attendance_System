#!/usr/bin/env python3
"""
5FRAS.py - 第五代人脸识别考勤系统
使用ONNX版本的MobileFaceNet模型进行人脸识别
"""

from concurrent.futures import ThreadPoolExecutor
from picamera2 import Picamera2
import cv2
import numpy as np
import os
import time
import logging
import queue
from threading import Event, Thread, Lock, Timer
import multiprocessing
from scipy import ndimage
from scipy import signal
from scipy.spatial import distance
import json

# 导入ONNX版本的MobileFaceNet
from mobilefacenet_utils_onnx import MobileFaceNetONNX

# 导入 GPIO 控制模块
from gpio_control import (
    initialize_gpio, set_gpio_state, cleanup_gpio,
    set_recognized_face, set_unknown_face, set_standby_state
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fras.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("5FRAS")

# 全局配置
CPU_COUNT = multiprocessing.cpu_count()
logger.info(f"系统CPU核心数: {CPU_COUNT}")

# 数据库配置
DB_ENABLED = False
try:
    from db_connector import DBConnector
    db_connector = DBConnector()
    if db_connector.test_connection():
        DB_ENABLED = True
        logger.info("数据库连接成功，启用数据库功能")
    else:
        logger.warning("数据库连接失败，禁用数据库功能")
except ImportError:
    logger.warning("未找到db_connector模块，禁用数据库功能")
except Exception as e:
    logger.warning(f"数据库初始化错误: {str(e)}，禁用数据库功能")

# Web服务器配置
WEB_ENABLED = False
try:
    from web_server import WebServer
    WEB_ENABLED = True
    logger.info("找到web_server模块，启用Web服务器功能")
except ImportError:
    logger.warning("未找到web_server模块，禁用Web服务器功能")

# 预加载关键模块
logger.info("预加载OpenCV和ONNX Runtime模块...")
try:
    # 预热OpenCV
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.resize(dummy_img, (50, 50))
    cv2.cvtColor(dummy_img, cv2.COLOR_BGR2GRAY)

    # 预热MobileFaceNet ONNX模型
    try:
        # 初始化MobileFaceNet ONNX模型
        # 如果模型文件不存在，这里会抛出异常，但不会中断程序
        mobilefacenet = MobileFaceNetONNX()
        # 使用一个空白图像进行预热
        _ = mobilefacenet.get_face_encoding(dummy_img)
        logger.info("MobileFaceNet ONNX模型预热成功")
    except FileNotFoundError:
        logger.warning("MobileFaceNet ONNX模型文件不存在，请先转换模型")
    except Exception as e:
        logger.warning(f"MobileFaceNet ONNX模型预热失败: {str(e)}")

    logger.info("模块预加载成功")
except Exception as e:
    logger.warning(f"模块预加载失败: {str(e)}")

# 初始化GPIO
try:
    initialize_gpio()
    logger.info("GPIO初始化成功")
except Exception as e:
    logger.warning(f"GPIO初始化失败: {str(e)}")

class FrameProcessor:
    """帧处理器基类"""
    def __init__(self, name):
        self.name = name
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        self.running = Event()
        self.running.set()
        self.thread = None
        self.width = 640
        self.height = 480
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.processing_times = []  # 存储处理时间
        self.max_processing_times = 100  # 最多存储100个处理时间

    def set_resolution(self, width, height):
        """设置处理分辨率"""
        self.width = width
        self.height = height

    def start(self):
        """启动处理线程"""
        self.thread = Thread(target=self._process_loop, name=f"{self.name}Thread")
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """停止处理线程"""
        self.running.clear()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def put_frame(self, frame):
        """将帧放入输入队列"""
        try:
            self.input_queue.put(frame, block=False)
            return True
        except queue.Full:
            return False

    def get_result(self, block=True, timeout=None):
        """从输出队列获取结果"""
        try:
            return self.output_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def _update_fps(self):
        """更新FPS计数"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = current_time

    def _record_processing_time(self, start_time):
        """记录处理时间"""
        elapsed = time.time() - start_time
        self.processing_times.append(elapsed)
        if len(self.processing_times) > self.max_processing_times:
            self.processing_times.pop(0)

    def get_avg_processing_time(self):
        """获取平均处理时间"""
        if not self.processing_times:
            return 0
        return sum(self.processing_times) / len(self.processing_times)

    def _process_loop(self):
        """处理循环，子类需要重写此方法"""
        pass

class FaceDetector(FrameProcessor):
    """人脸检测处理器"""
    def __init__(self):
        super().__init__("FaceDetector")
        # 加载OpenCV的人脸检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        # 添加眼睛检测器，减少误检
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        # 添加缓存机制
        self.detection_cache = {}  # 缓存检测结果
        self.cache_timeout = 0.5  # 缓存超时时间，单位秒

    def _extract_face_features(self, frame):
        """从帧中提取人脸特征"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用直方图均衡化增强对比度
        gray = cv2.equalizeHist(gray)

        # 检测人脸
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # 过滤人脸，确保检测到眼睛，减少误检
        valid_faces = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 1:  # 至少检测到一只眼睛
                valid_faces.append((x, y, w, h))

        return valid_faces

    def _process_loop(self):
        """人脸检测处理循环"""
        while self.running.is_set():
            try:
                # 从输入队列获取帧
                frame = self.input_queue.get(block=True, timeout=0.1)
                start_time = time.time()

                # 提取人脸特征
                face_features = self._extract_face_features(frame)

                # 创建结果
                result = {
                    "frame": frame,
                    "faces": face_features,
                    "timestamp": time.time()
                }

                # 将结果放入输出队列
                self.output_queue.put(result)

                # 更新FPS和处理时间
                self._update_fps()
                self._record_processing_time(start_time)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"人脸检测错误: {str(e)}")

class FaceRecognitionWorker:
    """增强版异步人脸识别工作线程 - 使用预编码的JSON数据和MobileFaceNet ONNX"""
    def __init__(self, face_encodings, face_names, tolerance=0.42, face_files=None):  # 降低容差阈值，减少误识别
        self.known_face_encodings = face_encodings
        self.known_face_names = face_names
        self.known_face_files = face_files if face_files is not None else [f"{name}.json" for name in face_names]
        self.tolerance = tolerance
        self.tasks = queue.Queue(maxsize=5)  # 增加任务队列大小
        self.results = queue.Queue()
        self.running = Event()
        self.running.set()

        # 增加批量处理
        self.process_batch_size = 1  # 默认每次处理1个任务
        if CPU_COUNT > 2:
            self.process_batch_size = 2  # 多核CPU时可以增加批处理大小

        # 添加缓存机制，减少重复识别
        self.recognition_cache = {}  # 缓存识别结果
        self.cache_timeout = 10.0  # 缓存超时时间，单位秒

        # 初始化MobileFaceNet ONNX模型
        self.mobilefacenet = MobileFaceNetONNX()

        # 启动工作线程
        self.thread = Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

        # 预热模型
        self._warmup_model()

    def _warmup_model(self):
        """预热人脸识别模型，减少首次识别延迟"""
        try:
            # 创建一个简单的测试图像 - 使用float32类型
            test_img = np.zeros((112, 112, 3), dtype=np.float32)
            # 预热MobileFaceNet ONNX
            _ = self.mobilefacenet.get_face_encoding(test_img)
            logger.info("FaceRecognitionWorker: MobileFaceNet ONNX模型预热成功")
        except Exception as e:
            logger.warning(f"FaceRecognitionWorker: 模型预热失败: {str(e)}")

    def submit_task(self, face_img, face_id=None):
        """提交人脸识别任务"""
        try:
            # 检查缓存
            img_hash = hash(face_img.tobytes())
            current_time = time.time()

            if img_hash in self.recognition_cache:
                cache_entry = self.recognition_cache[img_hash]
                # 检查缓存是否过期
                if current_time - cache_entry["timestamp"] < self.cache_timeout:
                    # 返回缓存的结果
                    return cache_entry["result"]

            # 将任务放入队列
            task = {"face_img": face_img, "face_id": face_id, "timestamp": current_time}
            self.tasks.put(task, block=False)
            return None
        except queue.Full:
            return None
        except Exception as e:
            logger.error(f"提交任务错误: {str(e)}")
            return None

    def get_result(self, block=True, timeout=None):
        """获取识别结果"""
        try:
            return self.results.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """停止工作线程"""
        self.running.clear()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _worker_loop(self):
        """工作线程循环"""
        while self.running.is_set():
            try:
                # 批量处理任务
                tasks_to_process = []
                for _ in range(self.process_batch_size):
                    try:
                        task = self.tasks.get(block=True, timeout=0.1)
                        tasks_to_process.append(task)
                    except queue.Empty:
                        break

                # 如果没有任务，继续下一轮循环
                if not tasks_to_process:
                    continue

                # 处理每个任务
                for task in tasks_to_process:
                    face_img = task["face_img"]
                    face_id = task["face_id"]

                    # 从图像提取人脸编码 - 使用MobileFaceNet ONNX
                    name = "Unknown"
                    confidence = 0

                    try:
                        # 使用MobileFaceNet ONNX提取人脸编码
                        face_encoding = self.mobilefacenet.get_face_encoding(face_img)

                        # 与预编码的人脸数据进行比较
                        if len(self.known_face_encodings) > 0:
                            # 使用MobileFaceNet的距离计算方法
                            face_distances = self.mobilefacenet.face_distance(self.known_face_encodings, face_encoding)

                            if len(face_distances) > 0:
                                best_match_index = np.argmin(face_distances)

                                # 使用更严格的阈值
                                if face_distances[best_match_index] < self.tolerance:
                                    name = self.known_face_names[best_match_index]
                                    file_name = self.known_face_files[best_match_index]
                                    confidence = 1 - face_distances[best_match_index]
                    except Exception as e:
                        logger.error(f"提取人脸编码错误: {str(e)}")

                    # 创建结果
                    result = {
                        "face_id": face_id,
                        "name": name,
                        "confidence": confidence,
                        "timestamp": time.time(),
                        "file_name": file_name if name != "Unknown" else "Unknown"
                    }

                    # 更新缓存
                    img_hash = hash(face_img.tobytes())
                    self.recognition_cache[img_hash] = {
                        "result": result,
                        "timestamp": time.time()
                    }

                    # 清理过期缓存
                    current_time = time.time()
                    expired_keys = [k for k, v in self.recognition_cache.items()
                                   if current_time - v["timestamp"] > self.cache_timeout]
                    for k in expired_keys:
                        del self.recognition_cache[k]

                    # 将结果放入结果队列
                    self.results.put(result)

            except Exception as e:
                logger.error(f"工作线程错误: {str(e)}")

class FaceTracker:
    """人脸追踪器，使用简单的IOU追踪算法"""
    def __init__(self, iou_threshold=0.3):
        self.tracked_faces = {}  # 字典: face_id -> face_info
        self.next_face_id = 0
        self.iou_threshold = iou_threshold
        self.max_disappeared = 30  # 最大消失帧数
        self.max_distance = 100  # 最大距离阈值

    def _calculate_iou(self, boxA, boxB):
        """计算两个边界框的IOU"""
        # 确保输入格式正确
        if len(boxA) == 4:
            xA, yA, wA, hA = boxA
            boxA = (xA, yA, xA + wA, yA + hA)
        if len(boxB) == 4:
            xB, yB, wB, hB = boxB
            boxB = (xB, yB, xB + wB, yB + hB)

        # 计算交集区域
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # 计算交集面积
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # 计算两个边界框的面积
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # 计算IOU
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def _calculate_distance(self, centroidA, centroidB):
        """计算两个中心点之间的欧氏距离"""
        return np.sqrt((centroidA[0] - centroidB[0])**2 + (centroidA[1] - centroidB[1])**2)

    def update(self, face_boxes):
        """更新追踪状态"""
        # 如果没有检测到人脸
        if len(face_boxes) == 0:
            # 增加所有已追踪人脸的消失计数
            for face_id in list(self.tracked_faces.keys()):
                self.tracked_faces[face_id]["disappeared"] += 1

                # 如果人脸消失太久，删除它
                if self.tracked_faces[face_id]["disappeared"] > self.max_disappeared:
                    del self.tracked_faces[face_id]

            return self.tracked_faces

        # 如果当前没有追踪任何人脸，为每个检测到的人脸分配新ID
        if len(self.tracked_faces) == 0:
            for i, bbox in enumerate(face_boxes):
                x, y, w, h = bbox
                centroid = (x + w//2, y + h//2)
                self.tracked_faces[self.next_face_id] = {
                    "bbox": bbox,
                    "centroid": centroid,
                    "disappeared": 0
                }
                self.next_face_id += 1
        else:
            # 获取当前追踪的人脸ID和中心点
            tracked_ids = list(self.tracked_faces.keys())
            tracked_centroids = [self.tracked_faces[face_id]["centroid"] for face_id in tracked_ids]

            # 计算新检测到的人脸的中心点
            new_centroids = []
            for bbox in face_boxes:
                x, y, w, h = bbox
                centroid = (x + w//2, y + h//2)
                new_centroids.append((centroid, bbox))

            # 计算距离矩阵
            distance_matrix = np.zeros((len(tracked_centroids), len(new_centroids)))
            for i, tracked_centroid in enumerate(tracked_centroids):
                for j, (new_centroid, _) in enumerate(new_centroids):
                    distance = self._calculate_distance(tracked_centroid, new_centroid)
                    distance_matrix[i, j] = distance

            # 使用匈牙利算法进行分配
            try:
                from scipy.optimize import linear_sum_assignment
                row_indices, col_indices = linear_sum_assignment(distance_matrix)
            except ImportError:
                # 如果没有scipy，使用贪婪算法
                row_indices = list(range(min(len(tracked_centroids), len(new_centroids))))
                col_indices = list(range(min(len(tracked_centroids), len(new_centroids))))

            # 创建已使用的行和列集合
            used_rows = set()
            used_cols = set()

            # 遍历分配结果
            for (row_idx, col_idx) in zip(row_indices, col_indices):
                # 如果距离太大，不进行分配
                if distance_matrix[row_idx, col_idx] > self.max_distance:
                    continue

                # 获取人脸ID和新的边界框
                face_id = tracked_ids[row_idx]
                new_centroid, new_bbox = new_centroids[col_idx]

                # 更新追踪信息
                self.tracked_faces[face_id] = {
                    "bbox": new_bbox,
                    "centroid": new_centroid,
                    "disappeared": 0
                }

                # 标记为已使用
                used_rows.add(row_idx)
                used_cols.add(col_idx)

            # 处理未分配的行（消失的人脸）
            for row_idx in range(len(tracked_centroids)):
                if row_idx not in used_rows:
                    face_id = tracked_ids[row_idx]
                    self.tracked_faces[face_id]["disappeared"] += 1

                    # 如果人脸消失太久，删除它
                    if self.tracked_faces[face_id]["disappeared"] > self.max_disappeared:
                        del self.tracked_faces[face_id]

            # 处理未分配的列（新出现的人脸）
            for col_idx in range(len(new_centroids)):
                if col_idx not in used_cols:
                    new_centroid, new_bbox = new_centroids[col_idx]
                    self.tracked_faces[self.next_face_id] = {
                        "bbox": new_bbox,
                        "centroid": new_centroid,
                        "disappeared": 0
                    }
                    self.next_face_id += 1

        return self.tracked_faces

class FaceRecognizer(FrameProcessor):
    """人脸识别处理器"""
    def __init__(self, face_encodings, face_names, tolerance=0.45, face_files=None):
        super().__init__("FaceRecognizer")
        self.known_face_encodings = face_encodings
        self.known_face_names = face_names
        self.known_face_files = face_files if face_files is not None else [f"{name}.json" for name in face_names]
        self.tolerance = tolerance
        #添加缓存机制
        self.recognition_cache = {}  # 缓存识别结果
        self.cache_timeout = 5.0  # 缓存超时时间，单位秒

        # 初始化MobileFaceNet ONNX模型
        self.mobilefacenet = MobileFaceNetONNX()

        # 预热模型
        self._warmup_model()

    def _warmup_model(self):
        """预热人脸识别模型，减少首次识别延迟"""
        try:
            # 创建一个简单的测试图像 - 使用float32类型
            test_img = np.zeros((112, 112, 3), dtype=np.float32)
            # 预热MobileFaceNet ONNX
            _ = self.mobilefacenet.get_face_encoding(test_img)
            logger.info("FaceRecognizer: MobileFaceNet ONNX模型预热成功")
        except Exception as e:
            logger.warning(f"FaceRecognizer: 模型预热失败: {str(e)}")

    def _process_frame(self, face_img):
        try:
            # 使用MobileFaceNet ONNX提取人脸编码
            face_encoding = self.mobilefacenet.get_face_encoding(face_img)
            return self._compare_faces(face_encoding)
        except Exception as e:
            logger.error(f"处理人脸错误: {str(e)}")
            return "Unknown", 0, "Unknown"

    def _compare_faces(self, face_encoding):
        """比较人脸编码与已知人脸"""
        name = "Unknown"
        confidence = 0
        file_name = "Unknown"

        if len(self.known_face_encodings) > 0:
            # 计算与已知人脸的距离
            face_distances = self.mobilefacenet.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < self.tolerance:
                name = self.known_face_names[best_match_index]
                file_name = self.known_face_files[best_match_index]
                confidence = 1 - face_distances[best_match_index]

        return name, confidence, file_name

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

    def _process_loop(self):
        """人脸识别处理循环"""
        #预分配内存
        combined_frame = np.zeros((self.height, self.width*2, 3), dtype=np.uint8)
        resized_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        while self.running.is_set():
            try:
                # 从输入队列获取帧
                frame = self.input_queue.get(block=True, timeout=0.1)
                start_time = time.time()

                # 应用图像增强
                enhanced_frame = self._enhance_image(frame)

                # 特征提取
                face_features = self._extract_face_features(enhanced_frame)

                # 自适应帧率控制
                self._adaptive_frame_rate()

                # 转换为RGB格式
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 缩小图像以加快处理速度
                small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)

                # 准备位置格式
                face_locations = []
                for (x, y, w, h) in face_features:
                    top = int(y * 0.25)
                    right = int((x + w) * 0.25)
                    bottom = int((y + h) * 0.25)
                    left = int(x * 0.25)
                    face_locations.append((top, right, bottom, left))

                # 对每个人脸进行识别
                face_encodings = []
                for (x, y, w, h) in face_features:
                    # 确保坐标在图像范围内
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, frame.shape[1] - x)
                    h = min(h, frame.shape[0] - y)

                    # 确保宽高大于0
                    if w > 0 and h > 0:
                        face_img = frame[y:y+h, x:x+w]
                        try:
                            # 使用MobileFaceNet ONNX提取人脸编码
                            face_encoding = self.mobilefacenet.get_face_encoding(face_img)
                            face_encodings.append(face_encoding)
                        except Exception as e:
                            logger.error(f"提取人脸编码错误: {str(e)}")
                            face_encodings.append(None)

                # 对每个人脸进行识别
                face_recognition_results = []
                for i, ((x, y, w, h), face_encoding) in enumerate(zip(face_features, face_encodings)):
                    name = "Unknown"
                    confidence = 0

                    if face_encoding is not None and len(self.known_face_encodings) > 0:
                        # 使用MobileFaceNet ONNX的距离计算方法
                        face_distances = self.mobilefacenet.face_distance(self.known_face_encodings, face_encoding)

                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)

                            if face_distances[best_match_index] < self.tolerance:
                                name = self.known_face_names[best_match_index]
                                file_name = self.known_face_files[best_match_index]
                                confidence = 1 - face_distances[best_match_index]

                    # 保存结果
                    face_recognition_results.append({
                        "bbox": (x, y, w, h),
                        "name": name,
                        "confidence": confidence,
                        "file_name": file_name if name != "Unknown" else "Unknown"
                    })

                # 输出结果
                result = {
                    "frame": frame,
                    "faces": face_features,
                    "recognition_results": face_recognition_results,
                    "timestamp": time.time()
                }

                # 将结果放入输出队列
                self.output_queue.put(result)

                # 更新FPS和处理时间
                self._update_fps()
                self._record_processing_time(start_time)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"人脸识别错误: {str(e)}")

class OptimizedFaceRecognizer(FrameProcessor):
    """优化后的人脸识别处理器，结合人脸追踪和异步识别，减少冗余识别处理"""
    def __init__(self, face_encodings, face_names, tolerance=0.48, face_files=None):
        super().__init__("OptimizedFaceRecognizer")
        self.known_face_encodings = face_encodings
        self.known_face_names = face_names
        self.known_face_files = face_files if face_files is not None else [f"{name}.json" for name in face_names]
        self.tolerance = tolerance
        self.face_tracker = FaceTracker()
        self.worker = FaceRecognitionWorker(face_encodings, face_names, tolerance=tolerance, face_files=self.known_face_files)
        # 添加识别稳定性控制
        self.recognition_results_history = {}  # 存储历史识别结果
        self.stability_threshold = 2  # 从3降低到2，需要连续几帧保持相同结果才更新显示
        # 初始化MobileFaceNet ONNX模型
        self.mobilefacenet = MobileFaceNetONNX()

        # 预热模型
        self._warmup_model()

    def _warmup_model(self):
        """预热人脸识别模型，减少首次识别延迟"""
        try:
            # 创建一个简单的测试图像 - 使用float32类型
            test_img = np.zeros((112, 112, 3), dtype=np.float32)
            # 预热MobileFaceNet ONNX
            _ = self.mobilefacenet.get_face_encoding(test_img)
            logger.info("OptimizedFaceRecognizer: MobileFaceNet ONNX模型预热成功")
        except Exception as e:
            logger.warning(f"OptimizedFaceRecognizer: 模型预热失败: {str(e)}")

    def _extract_face_features(self, frame):
        """从帧中提取人脸特征"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用直方图均衡化增强对比度
        gray = cv2.equalizeHist(gray)

        # 检测人脸
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return [(x, y, w, h) for (x, y, w, h) in faces]

    def _process_loop(self):
        """优化的人脸识别处理循环"""
        while self.running.is_set():
            try:
                # 从输入队列获取帧
                detection_result = self.input_queue.get(block=True, timeout=0.1)
                start_time = time.time()

                frame = detection_result["frame"]
                face_boxes = detection_result["faces"]

                # 更新人脸追踪器
                tracked_faces = self.face_tracker.update(face_boxes)

                # 处理每个追踪到的人脸
                recognition_results = []
                for face_id, face_info in tracked_faces.items():
                    bbox = face_info["bbox"]
                    x, y, w, h = bbox

                    # 确保坐标在图像范围内
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, frame.shape[1] - x)
                    h = min(h, frame.shape[0] - y)

                    # 确保宽高大于0
                    if w <= 0 or h <= 0:
                        continue

                    # 提取人脸图像
                    face_img = frame[y:y+h, x:x+w]

                    # 检查是否已经有识别结果
                    cached_result = self.worker.submit_task(face_img, face_id)

                    if cached_result:
                        # 使用缓存的结果
                        name = cached_result["name"]
                        confidence = cached_result["confidence"]
                        file_name = cached_result.get("file_name", "Unknown")
                    else:
                        # 默认为未知，等待异步识别结果
                        name = "Unknown"
                        confidence = 0
                        file_name = "Unknown"

                    # 添加到结果列表
                    recognition_results.append({
                        "face_id": face_id,
                        "bbox": bbox,
                        "name": name,
                        "confidence": confidence,
                        "file_name": file_name
                    })

                # 检查是否有新的识别结果
                while True:
                    try:
                        result = self.worker.get_result(block=False)
                        if result:
                            face_id = result["face_id"]
                            name = result["name"]
                            confidence = result["confidence"]
                            file_name = result.get("file_name", "Unknown")

                            # 更新历史记录
                            if face_id not in self.recognition_results_history:
                                self.recognition_results_history[face_id] = []

                            self.recognition_results_history[face_id].append(name)

                            # 只保留最近的几个结果
                            if len(self.recognition_results_history[face_id]) > self.stability_threshold:
                                self.recognition_results_history[face_id].pop(0)

                            # 更新识别结果
                            for i, res in enumerate(recognition_results):
                                if res["face_id"] == face_id:
                                    # 检查稳定性
                                    history = self.recognition_results_history[face_id]
                                    if len(history) >= self.stability_threshold and all(x == history[0] for x in history):
                                        recognition_results[i]["name"] = name
                                        recognition_results[i]["confidence"] = confidence
                                        recognition_results[i]["file_name"] = file_name
                    except queue.Empty:
                        break

                # 创建结果
                result = {
                    "frame": frame,
                    "faces": face_boxes,
                    "tracked_faces": tracked_faces,
                    "recognition_results": recognition_results,
                    "timestamp": time.time()
                }

                # 将结果放入输出队列
                self.output_queue.put(result)

                # 更新FPS和处理时间
                self._update_fps()
                self._record_processing_time(start_time)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"优化人脸识别错误: {str(e)}")

class ResultRenderer(FrameProcessor):
    """结果渲染器"""
    def __init__(self, width, height, display_fps=True):
        super().__init__("ResultRenderer")
        self.width = width
        self.height = height
        self.display_fps = display_fps
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.line_thickness = 2
        self.text_color = (255, 255, 255)  # 白色
        self.box_colors = [
            (0, 255, 0),    # 绿色 - 已知人脸
            (0, 0, 255),    # 红色 - 未知人脸
            (255, 0, 0)     # 蓝色 - 待识别
        ]

        # 添加帧率显示
        self.fps_history = []
        self.max_fps_history = 10

        # 添加状态显示
        self.status_text = ""
        self.status_color = (0, 255, 0)  # 绿色
        self.status_timeout = 0

        # 添加通知队列
        self.notifications = []
        self.max_notifications = 5
        self.notification_timeout = 3.0  # 3秒

        # 添加图像增强
        self.apply_enhancement = False

    def set_status(self, text, color=(0, 255, 0), timeout=3.0):
        """设置状态文本"""
        self.status_text = text
        self.status_color = color
        self.status_timeout = time.time() + timeout

    def add_notification(self, text, color=(255, 255, 255), timeout=3.0):
        """添加通知"""
        self.notifications.append({
            "text": text,
            "color": color,
            "timeout": time.time() + timeout
        })

        # 限制通知数量
        if len(self.notifications) > self.max_notifications:
            self.notifications.pop(0)

    def _draw_text_with_background(self, img, text, pos, font, font_scale, text_color, bg_color, thickness=1):
        """绘制带背景的文本"""
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size

        # 绘制背景矩形
        cv2.rectangle(img, pos, (pos[0] + text_w, pos[1] + text_h), bg_color, -1)

        # 绘制文本
        cv2.putText(img, text, (pos[0], pos[1] + text_h), font, font_scale, text_color, thickness)

    def _process_loop(self):
        """结果渲染循环"""
        while self.running.is_set():
            try:
                # 从输入队列获取结果
                result = self.input_queue.get(block=True, timeout=0.1)
                start_time = time.time()

                frame = result["frame"]
                recognition_results = result.get("recognition_results", [])

                # 创建结果帧的副本
                result_frame = frame.copy()

                # 绘制每个人脸的结果
                for face_result in recognition_results:
                    # 获取人脸信息
                    bbox = face_result["bbox"]
                    name = face_result["name"]
                    confidence = face_result.get("confidence", 0)

                    # 确定边框颜色
                    if name == "Unknown":
                        box_color = self.box_colors[1]  # 红色 - 未知人脸
                    else:
                        box_color = self.box_colors[0]  # 绿色 - 已知人脸

                    # 绘制人脸边框
                    x, y, w, h = bbox
                    cv2.rectangle(result_frame, (x, y), (x+w, y+h), box_color, 2)

                    # 准备显示文本
                    if name != "Unknown":
                        file_name = face_result.get("file_name", "Unknown")
                        display_text = f"{name} ({confidence:.2f}) - {file_name}"
                    else:
                        display_text = "Unknown"

                    # 绘制带背景的文本
                    text_bg_color = (0, 0, 0)  # 黑色背景
                    self._draw_text_with_background(
                        result_frame, display_text, (x, y-10),
                        self.font, self.font_scale, self.text_color, text_bg_color, self.line_thickness
                    )

                # 显示FPS
                if self.display_fps:
                    # 更新FPS历史
                    current_fps = self.fps
                    self.fps_history.append(current_fps)
                    if len(self.fps_history) > self.max_fps_history:
                        self.fps_history.pop(0)

                    # 计算平均FPS
                    avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

                    # 显示FPS
                    fps_text = f"FPS: {avg_fps:.1f}"
                    cv2.putText(result_frame, fps_text, (10, 30), self.font, self.font_scale, self.text_color, self.line_thickness)

                # 显示状态文本
                if self.status_text and time.time() < self.status_timeout:
                    status_pos = (10, result_frame.shape[0] - 10)
                    self._draw_text_with_background(
                        result_frame, self.status_text, status_pos,
                        self.font, self.font_scale, self.text_color, self.status_color, self.line_thickness
                    )

                # 显示通知
                notification_y = 70
                current_time = time.time()
                active_notifications = []

                for notification in self.notifications:
                    if current_time < notification["timeout"]:
                        # 显示通知
                        notification_pos = (10, notification_y)
                        self._draw_text_with_background(
                            result_frame, notification["text"], notification_pos,
                            self.font, self.font_scale, self.text_color, notification["color"], self.line_thickness
                        )
                        notification_y += 40
                        active_notifications.append(notification)

                # 更新活跃通知列表
                self.notifications = active_notifications

                # 将结果放入输出队列
                self.output_queue.put(result_frame)

                # 更新FPS和处理时间
                self._update_fps()
                self._record_processing_time(start_time)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"结果渲染错误: {str(e)}")

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

        # 摄像头模式
        self.single_camera_mode = False  # 默认为双摄像头模式
        self.use_mock_camera = False     # 默认不使用模拟摄像头

        # 添加帧缓冲区，减少丢帧
        self.frame_buffer_size = 3
        self.frame_buffer = queue.Queue(maxsize=self.frame_buffer_size)

        # 添加性能监控
        self.fps_history = []
        self.last_fps_time = time.time()
        self.frame_count = 0

    def initialize(self):
        """初始化摄像头"""
        try:
            logger.info("初始化摄像头...")

            # 尝试初始化单个摄像头
            try:
                logger.info("尝试初始化单个摄像头...")
                self.picam2 = Picamera2(0)  # 使用默认摄像头
                config = self.picam2.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"},
                    controls={"FrameRate": self.frame_rate,
                             "AwbEnable": 1,
                             "NoiseReductionMode": 1}
                )
                self.picam2.configure(config)
                self.picam2.start()

                # 测试摄像头
                test_frame = self.picam2.capture_array()

                if test_frame is None:
                    raise ValueError("无法获取测试帧")

                # 如果成功，设置为单摄像头模式
                self.single_camera_mode = True
                logger.info("单摄像头初始化成功")
                return True

            except Exception as e:
                logger.warning(f"单摄像头初始化失败: {str(e)}，尝试双摄像头模式")

                # 如果单摄像头失败，尝试双摄像头模式
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
                    raise ValueError("无法获取测试帧")

                # 设置为双摄像头模式
                self.single_camera_mode = False
                logger.info("双摄像头初始化成功")
                return True

        except Exception as e:
            logger.error(f"摄像头初始化失败: {str(e)}")
            return False

    def start_capture(self):
        """启动帧捕获线程"""
        self.capture_thread = Thread(target=self._capture_loop, name="FrameCapture")
        self.capture_thread.daemon = True
        self.capture_thread.start()

        self.distribution_thread = Thread(target=self._distribution_loop, name="FrameDistribution")
        self.distribution_thread.daemon = True
        self.distribution_thread.start()

    def stop_capture(self):
        """停止帧捕获"""
        self.running.clear()

        # 停止摄像头
        if hasattr(self, 'single_camera_mode') and self.single_camera_mode:
            if hasattr(self, 'picam2') and self.picam2 is not None:
                try:
                    self.picam2.stop()
                    logger.info("单摄像头已停止")
                except Exception as e:
                    logger.error(f"停止单摄像头错误: {str(e)}")
        else:
            # 停止双摄像头
            if hasattr(self, 'picam2_1') and self.picam2_1 is not None:
                try:
                    self.picam2_1.stop()
                    logger.info("入口摄像头已停止")
                except Exception as e:
                    logger.error(f"停止入口摄像头错误: {str(e)}")

            if hasattr(self, 'picam2_2') and self.picam2_2 is not None:
                try:
                    self.picam2_2.stop()
                    logger.info("出口摄像头已停止")
                except Exception as e:
                    logger.error(f"停止出口摄像头错误: {str(e)}")

        # 停止线程
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if hasattr(self, 'distribution_thread') and self.distribution_thread.is_alive():
            self.distribution_thread.join(timeout=1.0)

    def subscribe(self, processor):
        """添加订阅者"""
        if processor not in self.subscribers:
            self.subscribers.append(processor)

    def unsubscribe(self, processor):
        """移除订阅者"""
        if processor in self.subscribers:
            self.subscribers.remove(processor)

    def _capture_loop(self):
        """帧捕获循环"""
        while self.running.is_set():
            try:
                # 根据摄像头模式捕获帧
                if hasattr(self, 'use_mock_camera') and self.use_mock_camera:
                    # 模拟摄像头模式 - 创建一个黑色画面
                    mock_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    # 添加一些文本，表明这是模拟模式
                    cv2.putText(mock_frame, "Mock Camera Mode", (50, self.height//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # 创建两个相同的黑色帧
                    combined_frame = np.hstack((mock_frame, mock_frame.copy()))
                elif hasattr(self, 'single_camera_mode') and self.single_camera_mode:
                    # 单摄像头模式
                    try:
                        frame = self.picam2.capture_array()

                        # 创建一个相同大小的黑色帧作为第二个摄像头的替代
                        black_frame = np.zeros_like(frame)

                        # 水平拼接两个帧
                        combined_frame = np.hstack((frame, black_frame))
                    except Exception as e:
                        logger.error(f"单摄像头捕获错误: {str(e)}")
                        # 创建一个空白帧
                        mock_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                        cv2.putText(mock_frame, "Camera Error", (50, self.height//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        combined_frame = np.hstack((mock_frame, mock_frame.copy()))
                else:
                    # 双摄像头模式
                    try:
                        frame1 = self.picam2_1.capture_array()
                        frame2 = self.picam2_2.capture_array()

                        # 水平拼接两个帧
                        combined_frame = np.hstack((frame1, frame2))
                    except Exception as e:
                        logger.error(f"双摄像头捕获错误: {str(e)}，尝试使用单摄像头")

                        # 如果双摄像头失败，尝试使用单摄像头
                        if hasattr(self, 'picam2_1'):
                            try:
                                frame = self.picam2_1.capture_array()
                                black_frame = np.zeros_like(frame)
                                combined_frame = np.hstack((frame, black_frame))
                            except:
                                # 创建一个空白帧
                                combined_frame = np.zeros((self.height, self.width*2, 3), dtype=np.uint8)
                        else:
                            # 创建一个空白帧
                            combined_frame = np.zeros((self.height, self.width*2, 3), dtype=np.uint8)

                # 更新帧
                with self.frame_lock:
                    self.frame = combined_frame

                # 放入缓冲区
                try:
                    self.frame_buffer.put(combined_frame, block=False)
                except queue.Full:
                    # 如果缓冲区已满，移除最旧的帧
                    try:
                        _ = self.frame_buffer.get(block=False)
                        self.frame_buffer.put(combined_frame, block=False)
                    except queue.Empty:
                        pass

                # 更新FPS
                self.frame_count += 1
                current_time = time.time()
                elapsed = current_time - self.last_fps_time
                if elapsed >= 1.0:
                    fps = self.frame_count / elapsed
                    self.fps_history.append(fps)
                    if len(self.fps_history) > 10:
                        self.fps_history.pop(0)
                    self.frame_count = 0
                    self.last_fps_time = current_time

                # 控制帧率
                time.sleep(1.0 / self.frame_rate)

            except Exception as e:
                logger.error(f"帧捕获错误: {str(e)}")
                time.sleep(0.1)  # 出错时短暂暂停

    def _distribution_loop(self):
        """帧分发循环"""
        while self.running.is_set():
            try:
                # 从缓冲区获取帧
                frame = self.frame_buffer.get(block=True, timeout=0.1)

                # 分发给所有订阅者
                for processor in self.subscribers:
                    processor.put_frame(frame)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"帧分发错误: {str(e)}")

    def get_frame(self):
        """获取当前帧"""
        with self.frame_lock:
            if self.frame is None:
                return np.zeros((self.height, self.width*2, 3), dtype=np.uint8)
            return self.frame.copy()

    def get_fps(self):
        """获取平均FPS"""
        if not self.fps_history:
            return 0
        return sum(self.fps_history) / len(self.fps_history)

class FaceRecognitionSystem:
    def __init__(self):
        # 系统参数
        self.LOWER_RESOLUTION = False  # 改为显示高分辨率
        self.DISPLAY_FPS = True
        self.CAMERA_WIDTH = 1920 if not self.LOWER_RESOLUTION else 960  # 增加宽度
        self.CAMERA_HEIGHT = 1080 if not self.LOWER_RESOLUTION else 540  # 调整高度保持16:9比例
        self.RECOGNITION_INTERVAL = 3  # 增加识别间隔以减少处理负担
        self.TOLERANCE = 0.40  # 降低容差阈值，减少误识别
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

        # 处理管道
        self.frame_count = 0
        self.last_recognition_frame = 0

        # Web服务器
        self.web_enabled = WEB_ENABLED
        self.web_server = None

    def load_known_faces(self):
        """加载已知人脸数据"""
        try:
            # 首先尝试加载NPZ格式的特征向量
            npz_path = "face_features.npz"
            if os.path.exists(npz_path):
                logger.info(f"从NPZ文件加载人脸特征: {npz_path}")
                try:
                    data = np.load(npz_path)
                    self.known_face_encodings = data["face_encodings"]
                    self.known_face_names = data["face_names"]

                    # 加载文件名（如果存在）
                    if "face_files" in data:
                        self.known_face_files = data["face_files"]
                    else:
                        # 如果没有文件名信息，使用名称作为文件名
                        self.known_face_files = np.array([f"{name}.json" for name in self.known_face_names])

                    logger.info(f"成功加载 {len(self.known_face_names)} 个人脸特征")
                    return True
                except Exception as e:
                    logger.error(f"加载NPZ文件失败: {str(e)}")

            # 检查encoding_dataset目录是否存在
            encoding_dir = "encoding_dataset"
            if not os.path.exists(encoding_dir):
                logger.warning(f"{encoding_dir}目录不存在，创建空目录")
                os.makedirs(encoding_dir)
                return self._load_known_faces_from_images()

            # 加载encoding_dataset目录中的所有JSON文件
            logger.info(f"从{encoding_dir}目录加载人脸特征...")
            self.known_face_encodings = []
            self.known_face_names = []
            self.known_face_files = []  # 存储文件名，用于显示

            json_files = [f for f in os.listdir(encoding_dir) if f.endswith('.json')]
            if not json_files:
                logger.warning(f"{encoding_dir}目录中没有JSON文件")
                return self._load_known_faces_from_images()

            # 处理每个JSON文件
            for json_file in json_files:
                json_path = os.path.join(encoding_dir, json_file)
                logger.info(f"加载JSON文件: {json_file}")

                try:
                    # 加载JSON文件
                    with open(json_path, 'r') as f:
                        import json
                        data = json.load(f)

                    # 提取特征向量
                    for person in data:
                        name = person.get('name', os.path.splitext(json_file)[0])  # 使用文件名作为默认名称
                        features = person.get('features', [])

                        if features:
                            # 添加每个特征向量
                            for feature in features:
                                if isinstance(feature, list) and len(feature) > 0:
                                    self.known_face_encodings.append(np.array(feature, dtype=np.float32))
                                    self.known_face_names.append(name)
                                    self.known_face_files.append(json_file)  # 存储文件名

                    logger.info(f"从{json_file}加载了{len(features)}个特征向量")

                except Exception as e:
                    logger.error(f"加载JSON文件{json_file}失败: {str(e)}")

            # 检查是否成功加载了特征向量
            if len(self.known_face_encodings) > 0:
                logger.info(f"总共加载了{len(self.known_face_encodings)}个特征向量，来自{len(set(self.known_face_files))}个文件")

                # 保存为NPZ格式以加快下次加载
                try:
                    np.savez("face_features.npz",
                             face_encodings=np.array(self.known_face_encodings),
                             face_names=np.array(self.known_face_names),
                             face_files=np.array(self.known_face_files))
                    logger.info("已保存人脸特征为NPZ格式")
                except Exception as e:
                    logger.error(f"保存NPZ格式失败: {str(e)}")

                return True
            else:
                logger.warning("未能从JSON文件加载任何特征向量")
                return self._load_known_faces_from_images()

        except Exception as e:
            logger.error(f"加载人脸数据失败: {str(e)}")
            return self._load_known_faces_from_images()

    def _load_known_faces_from_images(self):
        """从图像文件加载人脸数据"""
        logger.info("从图像文件加载人脸数据...")
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_files = []  # 存储文件名

        # 检查Profile_Pictures目录是否存在
        if not os.path.exists("Profile_Pictures"):
            logger.warning("Profile_Pictures目录不存在，创建空目录")
            os.makedirs("Profile_Pictures")
            return False

        # 遍历目录中的每个子目录
        count = 0
        for person_dir in os.listdir("Profile_Pictures"):
            person_path = os.path.join("Profile_Pictures", person_dir)
            if os.path.isdir(person_path):
                name = person_dir
                logger.info(f"加载 {name} 的人脸图像...")

                # 遍历该人的所有图像
                for filename in os.listdir(person_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(person_path, filename)

                        try:
                            # 使用OpenCV读取图像
                            image = cv2.imread(img_path)
                            if image is None:
                                logger.warning(f"无法加载图像: {filename}")
                                continue

                            # 使用OpenCV的人脸检测器检测人脸
                            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

                            if len(faces) > 0:
                                # 使用第一个检测到的人脸
                                x, y, w, h = faces[0]
                                face_img = image[y:y+h, x:x+w]

                                # 使用MobileFaceNet ONNX提取人脸编码
                                mobilefacenet = MobileFaceNetONNX()
                                encoding = mobilefacenet.get_face_encoding(face_img)

                                self.known_face_encodings.append(encoding)
                                self.known_face_names.append(name)
                                # 使用图像文件名作为JSON文件名
                                self.known_face_files.append(f"{name}.json")
                                logger.info(f"加载人脸: {name}")
                                count += 1
                            else:
                                logger.warning(f"未检测到人脸: {filename}")
                        except Exception as e:
                            logger.error(f"加载人脸 {filename} 错误: {str(e)}")

        logger.info(f"成功加载 {count} 个人脸图像")

        # 保存为NPZ格式以加快下次加载
        if count > 0:
            try:
                np.savez("face_features.npz",
                         face_encodings=np.array(self.known_face_encodings),
                         face_names=np.array(self.known_face_names),
                         face_files=np.array(self.known_face_files))
                logger.info("已保存人脸特征为NPZ格式")
            except Exception as e:
                logger.error(f"保存NPZ格式失败: {str(e)}")

        return count > 0

    def initialize_system(self):
        """初始化整个系统"""
        try:
            # 使用预加载的picamera配置
            time_start = time.time()
            logger.info("开始系统初始化...")

            # 先加载人脸编码数据
            logger.info("先加载人脸编码数据...")
            if not self.load_known_faces():
                logger.error("加载人脸编码数据失败，无法继续初始化")
                return False

            # 创建帧源管理器 - 使用预优化配置
            self.frame_source = FrameSourceManager(
                self.CAMERA_WIDTH,
                self.CAMERA_HEIGHT,
                frame_rate=20
            )

            # 预初始化摄像头
            if not self.frame_source.initialize():
                logger.error("摄像头初始化失败，请检查摄像头连接")
                # 尝试使用模拟摄像头模式
                logger.info("尝试使用模拟摄像头模式...")
                self.frame_source.single_camera_mode = True
                self.frame_source.use_mock_camera = True
                logger.warning("已启用模拟摄像头模式，将使用黑色画面")
                # 继续初始化其他组件

            # 创建人脸检测器 - 使用OpenCV级联分类器
            self.face_detector = FaceDetector()

            # 使用优化的人脸识别器
            self.face_recognizer = OptimizedFaceRecognizer(
                self.known_face_encodings,
                self.known_face_names,
                tolerance=self.TOLERANCE,
                face_files=self.known_face_files if hasattr(self, 'known_face_files') else None
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

            logger.info(f"系统初始化完成，耗时: {time.time() - time_start:.2f} 秒")
            return True
        except Exception as e:
            logger.error(f"系统初始化失败: {str(e)}")
            return False

    def _start_web_server(self):
        """启动 Web 服务器线程"""
        if not self.web_enabled:
            logger.warning("Web 服务器已禁用")
            return

        try:
            self.web_server = WebServer(self)
            self.web_server.start()
            logger.info("Web 服务器已启动")
        except Exception as e:
            logger.error(f"启动 Web 服务器失败: {str(e)}")
            self.web_enabled = False

    def _pipeline_connector(self):
        """管道连接线程"""
        while self.running_event.is_set():
            try:
                # 人脸检测器 -> 人脸识别器
                detection_result = self.face_detector.get_result(block=True, timeout=0.1)
                if detection_result:
                    self.face_recognizer.put_frame(detection_result)

                # 人脸识别器 -> 结果渲染器
                recognition_result = self.face_recognizer.get_result(block=False)
                if recognition_result:
                    self.result_renderer.put_frame(recognition_result)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"管道连接错误: {str(e)}")

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

    def stop_system(self):
        """停止系统"""
        logger.info("正在停止系统...")

        # 停止运行标志
        self.running_event.clear()

        # 停止帧源
        if hasattr(self, 'frame_source'):
            self.frame_source.stop_capture()

        # 停止处理器
        if hasattr(self, 'face_detector'):
            self.face_detector.stop()
        if hasattr(self, 'face_recognizer'):
            self.face_recognizer.stop()
        if hasattr(self, 'result_renderer'):
            self.result_renderer.stop()

        # 停止管道连接线程
        if hasattr(self, 'pipeline_thread') and self.pipeline_thread.is_alive():
            self.pipeline_thread.join(timeout=1.0)

        # 停止 Web 服务器
        if self.web_enabled and hasattr(self, 'web_server'):
            self.web_server.stop()

        # 清理 GPIO
        try:
            cleanup_gpio()
        except:
            pass

        logger.info("系统已停止")

    def display_loop(self):
        """显示循环"""
        logger.info("开始显示循环")

        # 创建窗口
        base_window_name = "5FRAS - 人脸识别考勤系统"

        # 根据摄像头模式添加后缀
        if hasattr(self.frame_source, 'use_mock_camera') and self.frame_source.use_mock_camera:
            window_name = f"{base_window_name} [模拟摄像头模式]"
        elif hasattr(self.frame_source, 'single_camera_mode') and self.frame_source.single_camera_mode:
            window_name = f"{base_window_name} [单摄像头模式]"
        else:
            window_name = f"{base_window_name} [双摄像头模式]"

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.CAMERA_WIDTH, self.CAMERA_HEIGHT)

        try:
            while self.running_event.is_set():
                # 获取渲染结果
                try:
                    frame = self.result_renderer.get_result(block=True, timeout=0.1)
                except queue.Empty:
                    continue

                if frame is None:
                    continue

                # 显示帧
                cv2.imshow(window_name, frame)

                # 处理键盘事件
                key = cv2.waitKey(1) & 0xFF

                # ESC 键退出
                if key == 27:
                    logger.info("用户按下 ESC 键，退出程序")
                    break

                # 'q' 键退出
                elif key == ord('q'):
                    logger.info("用户按下 'q' 键，退出程序")
                    break

                # 's' 键保存当前帧
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"截图已保存: {filename}")

                # 'r' 键重新加载人脸数据
                elif key == ord('r'):
                    logger.info("重新加载人脸数据...")
                    self.load_known_faces()
                    # 更新人脸识别器
                    self.face_recognizer.known_face_encodings = self.known_face_encodings
                    self.face_recognizer.known_face_names = self.known_face_names
                    self.face_recognizer.known_face_files = self.known_face_files
                    # 更新工作线程
                    self.face_recognizer.worker.known_face_encodings = self.known_face_encodings
                    self.face_recognizer.worker.known_face_names = self.known_face_names
                    self.face_recognizer.worker.known_face_files = self.known_face_files
                    logger.info("人脸数据已重新加载")

        finally:
            # 关闭窗口
            cv2.destroyAllWindows()

    def run(self):
        """运行系统"""
        try:
            # 初始化系统
            if not self.initialize_system():
                return False

            # 启动处理
            self.start_processing()

            # 开始显示循环
            self.display_loop()

            return True
        except Exception as e:
            logger.error(f"系统运行时错误: {str(e)}")
            return False
        finally:
            # 停止系统
            self.stop_system()

def main():
    """主函数"""
    try:
        # 输出系统信息
        logger.info(f"OpenCV 版本: {cv2.__version__}")
        logger.info(f"Python 版本: {os.sys.version}")

        # 确保Profile_Pictures目录存在
        if not os.path.exists("Profile_Pictures"):
            logger.info("创建 Profile_Pictures 目录")
            os.makedirs("Profile_Pictures")

        # 创建并运行系统
        system = FaceRecognitionSystem()
        if not system.run():
            logger.error("系统运行失败")
    except KeyboardInterrupt:
        logger.info("用户中断，程序退出")
    except Exception as e:
        logger.error(f"程序异常: {str(e)}")
    finally:
        # 清理 GPIO
        try:
            cleanup_gpio()
        except:
            pass
        logger.info("程序退出")

if __name__ == "__main__":
    main()
