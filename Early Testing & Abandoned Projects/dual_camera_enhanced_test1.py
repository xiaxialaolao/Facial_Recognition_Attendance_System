from picamera2 import Picamera2
import cv2
import face_recognition
import numpy as np
import RPi.GPIO as GPIO
import time
import os
import logging
from threading import Thread, Event, Lock
import queue

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FaceRecognition")

# =============================================
# GPIO 控制逻辑 - 保持与 test1.py 完全相同
# =============================================
# 初始化 GPIO
# 设置 GPIO 模式为 BCM
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# 定义 GPIO 引脚
GPIO_LED_STANDBY = 23     # 默认亮起的 LED (待机状态)
GPIO_LED_UNKNOWN = 24     # 检测到未知人脸时亮起的 LED
GPIO_LED_RECOGNIZED = 16  # 检测到已知人脸时亮起的 LED

# 设置引脚为输出模式
GPIO.setup(GPIO_LED_STANDBY, GPIO.OUT)
GPIO.setup(GPIO_LED_UNKNOWN, GPIO.OUT)
GPIO.setup(GPIO_LED_RECOGNIZED, GPIO.OUT)

# 初始化状态：关闭所有 LED
GPIO.output(GPIO_LED_STANDBY, GPIO.LOW)
GPIO.output(GPIO_LED_UNKNOWN, GPIO.LOW)
GPIO.output(GPIO_LED_RECOGNIZED, GPIO.LOW)
time.sleep(0.3)  # 短暂延时确保状态稳定

# 默认状态：待机 LED 亮起
GPIO.output(GPIO_LED_STANDBY, GPIO.HIGH)

# =============================================
# 双摄像头支持的高级功能
# =============================================

class DualCameraFrameSource:
    """双摄像头帧源管理器"""
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        
        # 两个摄像头实例
        self.picam2_entrance = None  # 入口摄像头
        self.picam2_exit = None      # 出口摄像头
        
        # 运行状态
        self.running = Event()
        
        # 帧队列
        self.entrance_frame_queue = queue.Queue(maxsize=2)
        self.exit_frame_queue = queue.Queue(maxsize=2)
        
        # 线程
        self.entrance_thread = None
        self.exit_thread = None
    
    def initialize(self):
        """初始化双摄像头"""
        try:
            # 初始化入口摄像头 (摄像头 0)
            self.picam2_entrance = Picamera2(0)
            self.picam2_entrance.configure(self.picam2_entrance.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            ))
            
            # 初始化出口摄像头 (摄像头 1)
            self.picam2_exit = Picamera2(1)
            self.picam2_exit.configure(self.picam2_exit.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            ))
            
            # 启动摄像头
            self.picam2_entrance.start()
            self.picam2_exit.start()
            
            # 设置运行状态
            self.running.set()
            
            logger.info("Dual cameras initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Dual camera initialization error: {str(e)}")
            return False
    
    def start(self):
        """启动帧捕获线程"""
        # 启动入口摄像头线程
        self.entrance_thread = Thread(target=self._entrance_capture_loop, daemon=True)
        self.entrance_thread.start()
        
        # 启动出口摄像头线程
        self.exit_thread = Thread(target=self._exit_capture_loop, daemon=True)
        self.exit_thread.start()
        
        logger.info("Dual camera capture threads started")
    
    def _entrance_capture_loop(self):
        """入口摄像头帧捕获循环"""
        while self.running.is_set():
            try:
                frame = self.picam2_entrance.capture_array()
                if not self.entrance_frame_queue.full():
                    self.entrance_frame_queue.put(frame, block=False)
            except Exception as e:
                logger.error(f"Entrance camera capture error: {str(e)}")
                time.sleep(0.01)
    
    def _exit_capture_loop(self):
        """出口摄像头帧捕获循环"""
        while self.running.is_set():
            try:
                frame = self.picam2_exit.capture_array()
                if not self.exit_frame_queue.full():
                    self.exit_frame_queue.put(frame, block=False)
            except Exception as e:
                logger.error(f"Exit camera capture error: {str(e)}")
                time.sleep(0.01)
    
    def get_entrance_frame(self):
        """获取入口摄像头最新帧"""
        try:
            return self.entrance_frame_queue.get(block=True, timeout=1.0)
        except queue.Empty:
            return None
    
    def get_exit_frame(self):
        """获取出口摄像头最新帧"""
        try:
            return self.exit_frame_queue.get(block=True, timeout=1.0)
        except queue.Empty:
            return None
    
    def stop(self):
        """停止帧源"""
        self.running.clear()
        
        # 停止线程
        if self.entrance_thread:
            self.entrance_thread.join(timeout=1.0)
        if self.exit_thread:
            self.exit_thread.join(timeout=1.0)
        
        # 关闭摄像头
        if self.picam2_entrance:
            self.picam2_entrance.close()
        if self.picam2_exit:
            self.picam2_exit.close()
        
        logger.info("Dual cameras stopped")

class FaceDetector:
    """人脸检测器"""
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
    
    def detect_faces(self, frame):
        """检测人脸"""
        if frame is None:
            return []
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        # 转换为列表格式
        face_locations = []
        for (x, y, w, h) in faces:
            face_locations.append((y, x+w, y+h, x))  # top, right, bottom, left
        
        return face_locations

class FaceRecognizer:
    """人脸识别器"""
    def __init__(self, known_face_encodings=None, known_face_names=None, tolerance=0.6):
        self.known_face_encodings = known_face_encodings or []
        self.known_face_names = known_face_names or []
        self.tolerance = tolerance
    
    def recognize_faces(self, frame, face_locations):
        """识别人脸"""
        if frame is None or not face_locations:
            return []
        
        # 转换为RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 计算人脸编码
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # 识别结果
        recognition_results = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # 默认为未知
            name = "Unknown"
            confidence = 0.0
            
            if self.known_face_encodings:
                # 计算与已知人脸的距离
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    best_match_distance = face_distances[best_match_index]
                    
                    # 转换距离为置信度 (0-1)
                    confidence = 1.0 - best_match_distance
                    
                    # 如果置信度高于阈值，认为是已知人脸
                    if confidence > self.tolerance:
                        name = self.known_face_names[best_match_index]
            
            # 添加结果
            recognition_results.append({
                "name": name,
                "confidence": confidence,
                "location": face_location
            })
        
        return recognition_results

class DualCameraFaceRecognitionSystem:
    """双摄像头人脸识别系统"""
    def __init__(self):
        # 配置参数
        self.width = 640
        self.height = 480
        self.fps = 30
        self.tolerance = 0.6
        self.display_width = 1920
        self.display_height = 720
        
        # 组件
        self.frame_source = None
        self.face_detector = None
        self.face_recognizer = None
        
        # 运行状态
        self.running = True
        
        # 已知人脸数据
        self.known_face_encodings = []
        self.known_face_names = []
    
    def load_known_faces(self, faces_dir="Profile_Pictures"):
        """加载已知人脸"""
        try:
            logger.info(f"Loading face data from: {faces_dir}")
            
            # 确保目录存在
            if not os.path.exists(faces_dir):
                logger.warning(f"Face directory does not exist, creating: {faces_dir}")
                os.makedirs(faces_dir, exist_ok=True)
                return True
            
            # 获取所有图片文件
            image_files = [f for f in os.listdir(faces_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                logger.warning("No face images found")
                return True
            
            # 加载每个图片
            for image_file in image_files:
                person_name = os.path.splitext(image_file)[0]
                image_path = os.path.join(faces_dir, image_file)
                
                try:
                    # 加载图片并计算编码
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(person_name)
                        logger.info(f"Loaded face: {person_name}")
                    else:
                        logger.warning(f"No face found in image: {image_file}")
                except Exception as e:
                    logger.error(f"Error loading face {image_file}: {str(e)}")
            
            logger.info(f"Loaded {len(self.known_face_names)} faces")
            return True
        except Exception as e:
            logger.error(f"Error loading known faces: {str(e)}")
            return False
    
    def initialize_system(self):
        """初始化系统组件"""
        try:
            # 创建双摄像头帧源
            self.frame_source = DualCameraFrameSource(self.width, self.height, self.fps)
            if not self.frame_source.initialize():
                logger.error("Failed to initialize dual camera frame source")
                return False
            
            # 创建人脸检测器
            self.face_detector = FaceDetector()
            
            # 创建人脸识别器
            self.face_recognizer = FaceRecognizer(
                self.known_face_encodings,
                self.known_face_names,
                self.tolerance
            )
            
            # 启动帧源
            self.frame_source.start()
            
            logger.info("System initialized successfully")
            return True
        except Exception as e:
            logger.error(f"System initialization error: {str(e)}")
            return False
    
    def process_frame(self, frame, camera_label):
        """处理单个摄像头帧"""
        if frame is None:
            return frame, False
        
        # 检测人脸
        face_locations = self.face_detector.detect_faces(frame)
        
        # 识别人脸
        recognition_results = self.face_recognizer.recognize_faces(frame, face_locations)
        
        # 标记是否识别到已知人脸
        recognized_face = False
        
        # 处理识别结果
        for result in recognition_results:
            name = result["name"]
            confidence = result["confidence"]
            top, right, bottom, left = result["location"]
            
            # 检查是否是已知人脸
            if name != "Unknown" and confidence > self.tolerance:
                recognized_face = True
                color = (0, 255, 0)  # 绿色 - 已知人脸
            else:
                color = (0, 0, 255)  # 红色 - 未知人脸
            
            # 绘制人脸框
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # 计算置信度文本
            confidence_text = f"{confidence:.2f}" if confidence > 0 else ""
            label = f"{name} {confidence_text}"
            
            # 绘制名称
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 添加摄像头标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 2
        text = camera_label
        
        # 计算文本大小
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 计算文字位置（中间下方）
        x = int((frame.shape[1] - text_width) // 2)
        y = int(frame.shape[0] - 50)  # 距离底部50像素
        
        # 绘制文本
        cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness + 2)  # 白色描边
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 255), thickness)  # 红色文字
        
        return frame, recognized_face
    
    def run(self):
        """运行系统"""
        try:
            # 加载已知人脸
            if not self.load_known_faces():
                return False
            
            # 初始化系统
            if not self.initialize_system():
                return False
            
            # 创建显示窗口
            cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Face Recognition", self.display_width, self.display_height)
            
            # 主循环
            while self.running:
                # 获取两个摄像头的帧
                entrance_frame = self.frame_source.get_entrance_frame()
                exit_frame = self.frame_source.get_exit_frame()
                
                # 处理入口摄像头帧
                entrance_frame, entrance_recognized = self.process_frame(entrance_frame, "ENTRANCE")
                
                # 处理出口摄像头帧
                exit_frame, exit_recognized = self.process_frame(exit_frame, "EXIT")
                
                # =============================================
                # GPIO 控制逻辑 - 保持与 test1.py 相同的逻辑
                # =============================================
                # 如果任一摄像头识别到已知人脸，点亮已知人脸LED
                if entrance_recognized or exit_recognized:
                    # 切换 LED 状态：已知人脸 LED 亮起，其他 LED 熄灭
                    GPIO.output(GPIO_LED_RECOGNIZED, GPIO.HIGH)
                    GPIO.output(GPIO_LED_UNKNOWN, GPIO.LOW)
                    GPIO.output(GPIO_LED_STANDBY, GPIO.LOW)
                # 如果有人脸但都是未知人脸，点亮未知人脸LED
                elif (entrance_frame is not None and self.face_detector.detect_faces(entrance_frame)) or \
                     (exit_frame is not None and self.face_detector.detect_faces(exit_frame)):
                    # 切换 LED 状态：未知人脸 LED 亮起，其他 LED 熄灭
                    GPIO.output(GPIO_LED_UNKNOWN, GPIO.HIGH)
                    GPIO.output(GPIO_LED_RECOGNIZED, GPIO.LOW)
                    GPIO.output(GPIO_LED_STANDBY, GPIO.LOW)
                # 如果没有检测到人脸，保持待机状态
                else:
                    # 切换到待机状态
                    GPIO.output(GPIO_LED_STANDBY, GPIO.HIGH)
                    GPIO.output(GPIO_LED_UNKNOWN, GPIO.LOW)
                    GPIO.output(GPIO_LED_RECOGNIZED, GPIO.LOW)
                
                # 创建并排显示的组合帧
                if entrance_frame is not None and exit_frame is not None:
                    # 确保两个帧的大小相同
                    entrance_frame = cv2.resize(entrance_frame, (self.display_width // 2, self.display_height))
                    exit_frame = cv2.resize(exit_frame, (self.display_width // 2, self.display_height))
                    
                    # 水平拼接两个帧
                    combined_frame = np.hstack((entrance_frame, exit_frame))
                    
                    # 显示组合帧
                    cv2.imshow("Face Recognition", combined_frame)
                
                # 检查键盘输入
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("User pressed 'q', exiting")
                    break
            
            return True
        except Exception as e:
            logger.error(f"System runtime error: {str(e)}")
            return False
        finally:
            # 停止系统
            self.stop_system()
    
    def stop_system(self):
        """停止并清理系统"""
        logger.info("Stopping system...")
        
        # 停止帧源
        if self.frame_source:
            self.frame_source.stop()
        
        # 关闭窗口
        cv2.destroyAllWindows()
        # 确保窗口关闭
        for i in range(3):
            cv2.waitKey(1)

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
        system = DualCameraFaceRecognitionSystem()
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
    try:
        main()
    except KeyboardInterrupt:
        # 处理Ctrl+C中断
        print("\nProgram terminated by user")
    finally:
        # 清理GPIO资源 - 保持与test1.py相同的逻辑
        GPIO.output(GPIO_LED_STANDBY, GPIO.LOW)
        GPIO.output(GPIO_LED_UNKNOWN, GPIO.LOW)
        GPIO.output(GPIO_LED_RECOGNIZED, GPIO.LOW)
        GPIO.cleanup()
        print("程序已退出，GPIO 资源已释放")
