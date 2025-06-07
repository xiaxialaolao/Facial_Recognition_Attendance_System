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
# 从 FRAS.py 移植的高级功能
# =============================================

class FrameSource:
    """帧源管理器"""
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.picam2 = None
        self.running = Event()
        self.frame_queue = queue.Queue(maxsize=2)
        self.thread = None
    
    def initialize(self):
        """初始化摄像头"""
        try:
            self.picam2 = Picamera2()
            self.picam2.configure(self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            ))
            self.picam2.start()
            self.running.set()
            return True
        except Exception as e:
            logger.error(f"Camera initialization error: {str(e)}")
            return False
    
    def start(self):
        """启动帧捕获线程"""
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def _capture_loop(self):
        """帧捕获循环"""
        while self.running.is_set():
            try:
                frame = self.picam2.capture_array()
                if not self.frame_queue.full():
                    self.frame_queue.put(frame, block=False)
            except Exception as e:
                logger.error(f"Frame capture error: {str(e)}")
                time.sleep(0.01)
    
    def get_frame(self):
        """获取最新帧"""
        try:
            return self.frame_queue.get(block=True, timeout=1.0)
        except queue.Empty:
            return None
    
    def stop(self):
        """停止帧源"""
        self.running.clear()
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.picam2:
            self.picam2.close()

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

class EnhancedFaceRecognitionSystem:
    """增强的人脸识别系统"""
    def __init__(self):
        # 配置参数
        self.width = 640
        self.height = 480
        self.fps = 30
        self.tolerance = 0.6
        
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
            # 创建帧源
            self.frame_source = FrameSource(self.width, self.height, self.fps)
            if not self.frame_source.initialize():
                logger.error("Failed to initialize frame source")
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
            
            return True
        except Exception as e:
            logger.error(f"System initialization error: {str(e)}")
            return False
    
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
            cv2.resizeWindow("Face Recognition", 800, 600)
            
            # 主循环
            while self.running:
                # 获取帧
                frame = self.frame_source.get_frame()
                if frame is None:
                    continue
                
                # 检测人脸
                face_locations = self.face_detector.detect_faces(frame)
                
                # 识别人脸
                recognition_results = self.face_recognizer.recognize_faces(frame, face_locations)
                
                # =============================================
                # GPIO 控制逻辑 - 保持与 test1.py 相同的逻辑
                # =============================================
                # 默认状态：如果没有检测到人脸，保持待机状态
                if len(face_locations) == 0:
                    # 切换到待机状态
                    GPIO.output(GPIO_LED_STANDBY, GPIO.HIGH)
                    GPIO.output(GPIO_LED_UNKNOWN, GPIO.LOW)
                    GPIO.output(GPIO_LED_RECOGNIZED, GPIO.LOW)
                
                recognized_face = False  # 标记是否识别到已知人脸
                
                for result in recognition_results:
                    name = result["name"]
                    confidence = result["confidence"]
                    top, right, bottom, left = result["location"]
                    
                    if name != "Unknown" and confidence > self.tolerance:
                        recognized_face = True  # 标记为已识别到已知人脸
                        
                        # 切换 LED 状态：已知人脸 LED 亮起，其他 LED 熄灭
                        GPIO.output(GPIO_LED_RECOGNIZED, GPIO.HIGH)
                        GPIO.output(GPIO_LED_UNKNOWN, GPIO.LOW)
                        GPIO.output(GPIO_LED_STANDBY, GPIO.LOW)
                    elif not recognized_face:  # 如果是未知人脸且没有识别到已知人脸
                        # 切换 LED 状态：未知人脸 LED 亮起，其他 LED 熄灭
                        GPIO.output(GPIO_LED_UNKNOWN, GPIO.HIGH)
                        GPIO.output(GPIO_LED_RECOGNIZED, GPIO.LOW)
                        GPIO.output(GPIO_LED_STANDBY, GPIO.LOW)
                    
                    # 绘制人脸框和名称
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # 计算置信度文本
                    confidence_text = f"{confidence:.2f}" if confidence > 0 else ""
                    label = f"{name} {confidence_text}"
                    
                    # 绘制名称
                    cv2.putText(frame, label, (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # 显示帧
                cv2.imshow("Face Recognition", frame)
                
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
        system = EnhancedFaceRecognitionSystem()
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
