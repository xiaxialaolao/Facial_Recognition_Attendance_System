from picamera2 import Picamera2
import cv2
import face_recognition
import numpy as np
import os
import threading
import time
import logging
import queue
from typing import List, Tuple, Optional
from threading import Event, Thread, Lock
import multiprocessing
import warnings


# 配置日志 - 发布版本调回INFO级别以减少写日志的开销
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FaceRecognition")

# 检查系统CPU核心数
CPU_COUNT = multiprocessing.cpu_count()
logger.info(f"System CPU cores: {CPU_COUNT}")

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
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=2)
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
    """人脸追踪器 - 减少完整识别的频率"""
    def __init__(self):
        self.tracked_faces = {}  # 记录追踪的人脸
        self.next_id = 0
        self.max_lost_frames = 30
        self.match_threshold = 0.6  # IOU匹配阈值
        
    def update(self, faces):
        """更新追踪状态"""
        if not self.tracked_faces and len(faces) == 0:
            return {}
            
        # 初始化新的追踪字典
        new_tracked = {}
        
        # 如果有检测到的人脸
        if len(faces) > 0:
            # 为每个检测到的人脸寻找匹配的追踪ID
            for (x, y, w, h) in faces:
                face_rect = (x, y, w, h)
                best_match = None
                best_iou = 0
                
                # 寻找最佳匹配
                for face_id, face_info in self.tracked_faces.items():
                    old_rect = face_info["rect"]
                    iou = self._calculate_iou(face_rect, old_rect)
                    if iou > self.match_threshold and iou > best_iou:
                        best_match = face_id
                        best_iou = iou
                
                if best_match is not None:
                    # 更新已匹配的人脸
                    face_info = self.tracked_faces[best_match]
                    face_info["rect"] = face_rect
                    face_info["lost_frames"] = 0
                    face_info["total_frames"] += 1
                    new_tracked[best_match] = face_info
                else:
                    # 创建新的追踪记录
                    new_tracked[self.next_id] = {
                        "rect": face_rect,
                        "lost_frames": 0,
                        "total_frames": 1,
                        "need_recognition": True,
                        "name": None,
                        "confidence": 0
                    }
                    self.next_id += 1
        
        # 更新未匹配的追踪记录（丢失帧数增加）
        for face_id, face_info in self.tracked_faces.items():
            if face_id not in new_tracked:
                face_info["lost_frames"] += 1
                if face_info["lost_frames"] <= self.max_lost_frames:
                    new_tracked[face_id] = face_info
        
        self.tracked_faces = new_tracked
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
    """异步人脸识别工作线程"""
    def __init__(self, face_encodings, face_names, tolerance=0.45):
        self.known_face_encodings = face_encodings
        self.known_face_names = face_names
        self.tolerance = tolerance
        self.tasks = queue.Queue(maxsize=3)  # 限制任务队列大小
        self.results = queue.Queue()
        self.running = Event()
        self.running.set()
        self.thread = Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        # 增加批量处理
        self.process_batch_size = 1  # 默认每次处理1个任务
        if CPU_COUNT > 2:
            self.process_batch_size = 2  # 多核CPU时可以增加批处理大小
        
    def submit_task(self, face_id, face_img):
        """提交人脸识别任务"""
        try:
            # 如果队列满，不阻塞，直接返回失败
            if self.tasks.full():
                return False
            self.tasks.put((face_id, face_img), block=False)
            return True
        except:
            return False
            
    def get_result(self):
        """获取识别结果"""
        try:
            return self.results.get(block=False)
        except queue.Empty:
            return None
            
    def stop(self):
        """停止工作线程"""
        self.running.clear()
        
    def _worker_loop(self):
        """工作线程循环"""
        while self.running.is_set():
            try:
                # 获取任务
                face_id, face_img = self.tasks.get(block=True, timeout=0.5)
                
                # 确保图像有效
                if face_img is None or face_img.size == 0 or face_img.shape[0] < 20 or face_img.shape[1] < 20:
                    # 图像无效，返回未知结果
                    self.results.put({
                        "face_id": face_id,
                        "name": "Unknown",
                        "confidence": 0,
                        "timestamp": time.time()
                    })
                    continue
                
                # 确保图像是RGB格式
                if face_img.shape[2] == 3:
                    rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                else:
                    rgb_img = face_img
                
                # 尝试多种尺寸进行处理，提高检测率
                face_locations = []
                face_encoding = None
                
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
                        num_jitters=0,
                        model="small"
                    )[0]
                
                name = "Unknown"
                confidence = 0
                
                # 如果检测到人脸并成功编码
                if face_encoding is not None:
                    # 与已知人脸比较
                    if len(self.known_face_encodings) > 0:
                        face_distances = face_recognition.face_distance(
                            self.known_face_encodings, face_encoding
                        )
                        best_match_index = np.argmin(face_distances)
                        
                        # 采用参考test2.py的tolerance值
                        if face_distances[best_match_index] < self.tolerance:
                            name = self.known_face_names[best_match_index]
                            confidence = 1 - face_distances[best_match_index]
                            
                            # 增加高置信度检测，防止误识别
                            if confidence < 0.55 and face_distances[best_match_index] > 0.4:
                                # 再次尝试更严格的比对
                                second_encoding = face_recognition.face_encodings(
                                    rgb_img,
                                    face_locations,
                                    num_jitters=2,  # 增加抖动采样
                                    model="large"   # 使用大模型提高精度
                                )[0]
                                
                                second_distances = face_recognition.face_distance(
                                    self.known_face_encodings, second_encoding
                                )
                                
                                if second_distances[best_match_index] < face_distances[best_match_index]:
                                    # 第二次识别更好，使用它
                                    confidence = 1 - second_distances[best_match_index]
                                else:
                                    # 置信度太低，可能是误识别，标记为未知
                                    if confidence < 0.53:
                                        name = "Unknown"
                                        confidence = 0
                
                # 将结果放入结果队列
                self.results.put({
                    "face_id": face_id,
                    "name": name,
                    "confidence": confidence,
                    "timestamp": time.time()
                })
                
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
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # 使用更高效的检测参数
        self.scale_factor = 1.1  # 更低的缩放因子，提高检测率
        self.min_neighbors = 4   # 更多的最小邻居，减少误检
        self.min_size = (45, 45) # 更大的最小尺寸，过滤小误检
        
        logger.info("Using OpenCV cascade classifier for face detection")
            
        # 增加自适应处理控制
        self.last_process_time = time.time()
        self.process_interval = 0.05  # 初始处理间隔 - 降低间隔提高响应速度
        self.load_factor = 0.0  # 系统负载因子
        
        # 跟踪上一帧检测到的人脸，用于稳定检测
        self.prev_faces = []
        self.face_stabilization_counter = 0
        
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
                
                # 使用级联分类器进行人脸检测
                faces = self.face_cascade.detectMultiScale(
                    small_gray, 
                    scaleFactor=self.scale_factor,
                    minNeighbors=self.min_neighbors,
                    minSize=self.min_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # 将人脸坐标放大回原始尺寸
                if len(faces) > 0:
                    faces = faces * (1.0 / scale_factor)  # 动态调整放大比例
                    # 确保所有坐标都为整数
                    faces = faces.astype(int)
                
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
    def __init__(self, face_encodings, face_names, tolerance=0.45):
        super().__init__("FaceRecognizer")
        self.known_face_encodings = face_encodings
        self.known_face_names = face_names
        self.tolerance = tolerance
        
    def _process_loop(self):
        while self.running.is_set():
            try:
                # 获取帧和人脸位置
                data = self.input_queue.get(block=True, timeout=0.1)
                
                frame = data["frame"]
                faces = data["faces"]
                
                # 存储识别结果
                recognition_results = []
                
                # 如果没有人脸，返回空结果
                if len(faces) == 0:
                    result = {
                        "frame": frame,
                        "faces": faces,
                        "recognition_results": recognition_results,
                        "timestamp": time.time()
                    }
                    self.output_queue.put(result, block=False)
                    continue

                # 转换为RGB格式，用于face_recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 缩小图像以加快处理速度
                small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)

                # 准备face_recognition需要的位置格式
                face_locations = []
                for (x, y, w, h) in faces:
                    top = int(y * 0.25)
                    right = int((x + w) * 0.25)
                    bottom = int((y + h) * 0.25)
                    left = int(x * 0.25)
                    face_locations.append((top, right, bottom, left))
                
                # 批量进行人脸编码
                face_encodings = face_recognition.face_encodings(small_frame, face_locations, num_jitters=0)
                
                # 对每个人脸进行识别
                for (x, y, w, h), face_encoding in zip(faces, face_encodings):
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
                    recognition_results.append({
                        "bbox": (x, y, w, h),
                        "name": name,
                        "confidence": confidence
                    })
                
                # 输出结果
                result = {
                    "frame": frame,
                    "faces": faces,
                    "recognition_results": recognition_results,
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
    def __init__(self, face_encodings, face_names, tolerance=0.45):
        super().__init__("OptimizedFaceRecognizer")
        self.known_face_encodings = face_encodings
        self.known_face_names = face_names
        self.tolerance = tolerance
        self.face_tracker = FaceTracker()
        self.worker = FaceRecognitionWorker(face_encodings, face_names, tolerance=tolerance)
        # 添加识别稳定性控制
        self.recognition_results_history = {}  # 存储历史识别结果
        self.stability_threshold = 3  # 需要连续几帧保持相同结果才更新显示
        
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
                                # 如果是未知切换到已知，快速接受变化
                                if history["name"] == "Unknown" and face_name != "Unknown":
                                    if confidence > 0.6:  # 高置信度时快速切换
                                        history["name"] = face_name
                                        history["confidence"] = confidence
                                        history["stable_count"] = 2
                                        history["unstable_count"] = 0
                                    else:
                                        history["stable_count"] = 0
                                        history["unstable_count"] += 1
                                # 如果是已知变成未知，需要更多帧才接受变化
                                elif history["name"] != "Unknown" and face_name == "Unknown":
                                    history["unstable_count"] += 1
                                    # 需要更多的未知帧才会切换
                                    if history["unstable_count"] > 5:
                                        history["name"] = face_name
                                        history["confidence"] = confidence
                                        history["stable_count"] = 0
                                # 已知人脸之间的切换，需要一定稳定性
                                else:
                                    history["unstable_count"] += 1
                                    if history["unstable_count"] > 3 and confidence > 0.6:
                                        history["name"] = face_name
                                        history["confidence"] = confidence
                                        history["stable_count"] = 1
                                        history["unstable_count"] = 0
                        
                        # 允许下一次识别
                        tracked_faces[face_id]["need_recognition"] = True
                        bbox = tracked_faces[face_id]["rect"]
                        
                        # 使用稳定的结果
                        stable_result = self.recognition_results_history[face_id]
                        recognition_results.append({
                            "bbox": bbox,
                            "name": stable_result["name"],
                            "confidence": stable_result["confidence"]
                        })
                    else:
                        # 如果人脸已经不在追踪中，创建临时结果
                        bbox = (0, 0, 0, 0)
                        recognition_results.append({
                            "bbox": bbox,
                            "name": face_name,
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
                                "confidence": stable_result["confidence"]
                            })
                
                # 清理已不再追踪的人脸历史
                face_ids_to_remove = []
                for face_id in self.recognition_results_history:
                    if face_id not in tracked_faces:
                        face_ids_to_remove.append(face_id)
                
                for face_id in face_ids_to_remove:
                    del self.recognition_results_history[face_id]
                    
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
                    
                    # 画框 - 使用更清晰的矩形
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

                    # 显示名称和置信度 - 使用带黑边的文字
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
                    text_y = y - 10
                    
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
                instruction_text = "Press 'q' to exit, 's' to screenshot"
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
    """帧源管理器"""
    def __init__(self, width, height, frame_rate=30):
        # 使用高分辨率捕获
        self.width = 1920  # 直接设置为1920 x 1080
        self.height = 1080
        self.display_width = width
        self.display_height = height
        self.frame_rate = frame_rate
        self.frame = None
        self.frame_lock = Lock()
        self.running = Event()
        self.running.set()
        self.picam2 = None
        self.subscribers = []
        
    def initialize(self):
        """初始化摄像头"""
        try:
            logger.info("Initializing camera...")
            self.picam2 = Picamera2()
            
            # 使用高质量预览配置 - 全高清分辨率
            config = self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls={"FrameRate": self.frame_rate, 
                         "AwbEnable": 1,  # 启用自动白平衡
                         "NoiseReductionMode": 1,  # 仍使用一些降噪
                         "FrameDurationLimits": (33333, 33333)}  # 固定帧持续时间约为30fps
            )
            
            self.picam2.configure(config)
            self.picam2.set_controls({"FrameRate": self.frame_rate, "NoiseReductionMode": 0})
            self.picam2.start()
            
            # 测试拍摄
            test_frame = self.picam2.capture_array()
            if test_frame is None:
                raise ValueError("Unable to get test frame")
                
            if len(test_frame.shape) == 3 and test_frame.shape[2] == 4:
                test_frame = cv2.cvtColor(test_frame, cv2.COLOR_RGBA2BGR)
                
            logger.info(f"Camera initialization successful, frame size: {test_frame.shape}")
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
        
        while self.running.is_set():
            try:
                # 捕获帧
                captured_frame = self.picam2.capture_array()
                
                if captured_frame is None:
                    time.sleep(0.01)
                    continue

                # 处理格式
                if len(captured_frame.shape) == 3 and captured_frame.shape[2] == 4:
                    captured_frame = cv2.cvtColor(captured_frame, cv2.COLOR_RGBA2BGR)
                elif len(captured_frame.shape) == 3 and captured_frame.shape[2] == 3:
                    captured_frame = cv2.cvtColor(captured_frame, cv2.COLOR_RGB2BGR)
                
                # 更新帧
                with self.frame_lock:
                    self.frame = captured_frame.copy()
                    
                # 计算帧率
                frame_count += 1
                current_time = time.time()
                if current_time - last_fps_update >= 5.0:  # 每5秒报告一次
                    fps = frame_count / (current_time - last_fps_update)
                    logger.info(f"Frame capture rate: {fps:.1f} FPS")
                    frame_count = 0
                    last_fps_update = current_time
                    
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
        if self.picam2:
            self.picam2.close()

class FaceRecognitionSystem:
    def __init__(self):
        # 系统参数
        self.LOWER_RESOLUTION = False  # 改为显示高分辨率
        self.DISPLAY_FPS = True
        self.CAMERA_WIDTH = 1280 if not self.LOWER_RESOLUTION else 640
        self.CAMERA_HEIGHT = 960 if not self.LOWER_RESOLUTION else 480
        self.RECOGNITION_INTERVAL = 3  # 增加识别间隔以减少处理负担
        self.TOLERANCE = 0.45
        
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
        
        # 处理器组件
        self.frame_source = None
        self.face_detector = None
        self.face_recognizer = None
        self.result_renderer = None
        
    def load_known_faces(self, faces_dir: str = "Profile_Pictures"):
        """加载已知人脸"""
        try:
            logger.info(f"Loading face data: {faces_dir}")
            if not os.path.exists(faces_dir):
                logger.warning(f"Face directory does not exist, creating: {faces_dir}")
                os.makedirs(faces_dir, exist_ok=True)
                
            count = 0
            for filename in os.listdir(faces_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(faces_dir, filename)
                    name = os.path.splitext(filename)[0]
                    
                    try:
                        image = face_recognition.load_image_file(img_path)
                        # 缩小图像加快编码
                        small_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                        encodings = face_recognition.face_encodings(small_image)
                        
                        if encodings:
                            self.known_face_encodings.append(encodings[0])
                            self.known_face_names.append(name)
                            logger.info(f"Loaded face: {name}")
                            count += 1
                        else:
                            logger.warning(f"No face detected: {filename}")
                    except Exception as e:
                        logger.error(f"Error loading face {filename}: {str(e)}")
            
            logger.info(f"Total loaded faces: {count}")
            
            # 如果没有找到人脸，添加测试数据
            if count == 0:
                test_encoding = np.random.rand(128)
                self.known_face_encodings.append(test_encoding)
                self.known_face_names.append("Test User")
                logger.info("Added test user data")
                
            return True
        except Exception as e:
            logger.error(f"Failed to load face data: {str(e)}")
            return False
            
    def initialize_system(self):
        """初始化整个系统"""
        try:
            # 使用预加载的picamera配置
            time_start = time.time()
            logger.info("Starting system initialization...")
            
            # 增加GPU加速配置检测
            try:
                cv_build_info = cv2.getBuildInformation()
                if "OpenCL:                      YES" in cv_build_info:
                    logger.info("OpenCL support detected, attempting to enable GPU acceleration")
                    cv2.ocl.setUseOpenCL(True)
                    if cv2.ocl.useOpenCL():
                        logger.info("Successfully enabled OpenCL GPU acceleration")
                else:
                    logger.info("OpenCL enabling failed, using CPU mode")
            except Exception as e:
                logger.warning(f"Error checking GPU acceleration support: {str(e)}")

            # 创建帧源管理器 - 使用预优化配置
            self.frame_source = FrameSourceManager(
                self.CAMERA_WIDTH, 
                self.CAMERA_HEIGHT, 
                frame_rate=30
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
                tolerance=self.TOLERANCE
            )
            
            # 创建结果渲染器
            self.result_renderer = ResultRenderer(
                self.CAMERA_WIDTH, 
                self.CAMERA_HEIGHT,
                display_fps=self.DISPLAY_FPS
            )
            
            logger.info(f"System initialization complete, time taken: {time.time() - time_start:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            return False
            
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
        cv2.resizeWindow("Face Recognition", 1280, 960)  # 固定显示尺寸
        
        while self.running_event.is_set():
            try:
                # 获取渲染后的帧
                frame = self.result_renderer.get_result()
                
                if frame is not None:
                    # 显示帧
                    cv2.imshow("Face Recognition", frame)
                
                # 检查键盘输入
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("User pressed 'q', exiting")
                    break
                elif key == ord('s'):
                    # 截图功能
                    if frame is not None:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"screenshot_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        logger.info(f"Screenshot saved as: {filename}")
                        
                        # 显示保存提示 - 添加黑色边缘
                        save_frame = frame.copy()
                        save_text = f"Saved: {filename}"
                        text_x, text_y = 10, 120
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_thickness = 2
                        
                        # 使用像素高度设置字体大小 - 保存消息使用36像素高
                        desired_height_px = 36
                        font_size = self.result_renderer._get_font_scale_from_pixels(
                            font, save_text, desired_height_px, font_thickness)
                        
                        # 先绘制黑色描边
                        for offset_x in [-2, 0, 2]:
                            for offset_y in [-2, 0, 2]:
                                if offset_x == 0 and offset_y == 0:
                                    continue
                                cv2.putText(
                                    save_frame, 
                                    save_text, 
                                    (text_x + offset_x, text_y + offset_y), 
                                    font, 
                                    font_size, 
                                    (0, 0, 0),  # 黑色边缘
                                    font_thickness + 1
                                )
                        
                        # 再绘制绿色文字
                        cv2.putText(
                            save_frame, 
                            save_text, 
                            (text_x, text_y), 
                            font, 
                            font_size, 
                            (0, 255, 0),  # 绿色文字
                            font_thickness
                        )
                        
                        cv2.imshow("Face Recognition", save_frame)
                        cv2.waitKey(500)  # 短暂显示保存提示
                    
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
            
        logger.info("System stopped")
        
    def run(self):
        """运行系统"""
        try:
            # 加载人脸数据
            if not self.load_known_faces():
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
        
        # 确保Profile_Pictures目录存在
        if not os.path.exists("Profile_Pictures"):
            logger.info("Creating Profile_Pictures directory")
            os.makedirs("Profile_Pictures")
        
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

if __name__ == "__main__":
    main()
