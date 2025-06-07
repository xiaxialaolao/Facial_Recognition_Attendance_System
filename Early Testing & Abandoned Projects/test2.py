from picamera2 import Picamera2
import cv2
import face_recognition
import numpy as np
import os
import RPi.GPIO as GPIO
import time
from threading import Timer, Lock

# 初始化 GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# 定义 GPIO 引脚
GPIO_LED_RECOGNIZED = 16  # 识别到已知人脸时亮起的 LED (状态C)
GPIO_LED_UNKNOWN = 24     # 检测到未知人脸时亮起的 LED (状态B)
GPIO_LED_STANDBY = 23     # 无人脸状态时亮起的 LED (状态A)

# 设置为输出模式
GPIO.setup(GPIO_LED_RECOGNIZED, GPIO.OUT)
GPIO.setup(GPIO_LED_UNKNOWN, GPIO.OUT)
GPIO.setup(GPIO_LED_STANDBY, GPIO.OUT)

# 确保所有 LED 都是关闭状态，然后再设置初始状态
GPIO.output(GPIO_LED_RECOGNIZED, GPIO.LOW)
GPIO.output(GPIO_LED_UNKNOWN, GPIO.LOW)
GPIO.output(GPIO_LED_STANDBY, GPIO.LOW)
time.sleep(0.3)  # 延长延时，确保状态稳定

# 初始状态A：无人脸状态，GPIO 23常亮
GPIO.output(GPIO_LED_STANDBY, GPIO.HIGH)
time.sleep(0.3)  # 延长延时，确保状态稳定

# 定义 GPIO 控制变量
_gpio_state = "standby"  # 可能的状态: "standby", "unknown", "recognized"
_recognized_timer = None  # 定时器，用于延时熄灯
_unknown_timer = None     # 定时器，用于延时熄灯
_hold_time = 5.0          # LED保持亮灯的时间，单位秒（5秒）
_gpio_lock = Lock()       # 添加锁，防止并发访问GPIO

# 定义 GPIO 控制函数
def _set_gpio_pins(recognized=False, unknown=False, standby=False):
    """直接设置GPIO引脚状态，使用锁防止并发访问"""
    with _gpio_lock:  # 使用锁确保GPIO操作的原子性
        try:
            # 先关闭所有LED，防止闪烁
            GPIO.output(GPIO_LED_RECOGNIZED, GPIO.LOW)
            GPIO.output(GPIO_LED_UNKNOWN, GPIO.LOW)
            GPIO.output(GPIO_LED_STANDBY, GPIO.LOW)
            time.sleep(0.1)  # 短暂延时，确保状态稳定

            if recognized:
                # 状态C：识别到已知人脸，GPIO 16亮
                GPIO.output(GPIO_LED_RECOGNIZED, GPIO.HIGH)
                print("GPIO set: Recognized face LED (GPIO 16) ON")
            elif unknown:
                # 状态B：检测到未知人脸，GPIO 24亮
                GPIO.output(GPIO_LED_UNKNOWN, GPIO.HIGH)
                print("GPIO set: Unknown face LED (GPIO 24) ON")
            elif standby:
                # 状态A：无人脸状态，GPIO 23亮
                GPIO.output(GPIO_LED_STANDBY, GPIO.HIGH)
                print("GPIO set: Standby LED (GPIO 23) ON")
        except Exception as e:
            print(f"GPIO pin control error: {str(e)}")

def _delayed_turn_off_recognized():
    """延时关闭已识别LED的定时器回调函数"""
    global _gpio_state, _recognized_timer

    try:
        # 关闭已识别LED，打开待机LED（状态A）
        _set_gpio_pins(standby=True)
        print("GPIO delayed turn off: Recognized face LED, switching to Standby")

        # 更新状态
        _gpio_state = "standby"
        _recognized_timer = None
    except Exception as e:
        print(f"GPIO delayed turn off recognized error: {str(e)}")

def _delayed_turn_off_unknown():
    """延时关闭未知人脸LED的定时器回调函数"""
    global _gpio_state, _unknown_timer

    try:
        # 关闭未知人脸LED，打开待机LED（状态A）
        _set_gpio_pins(standby=True)
        print("GPIO delayed turn off: Unknown face LED, switching to Standby")

        # 更新状态
        _gpio_state = "standby"
        _unknown_timer = None
    except Exception as e:
        print(f"GPIO delayed turn off unknown error: {str(e)}")

def set_gpio_state(recognized=False, unknown=False):
    """设置GPIO状态，根据人脸识别状态控制LED"""
    global _gpio_state, _recognized_timer, _unknown_timer

    try:
        # 如果识别到已知人脸（状态C）
        if recognized:
            # 如果当前状态不是"recognized"
            if _gpio_state != "recognized":
                # 设置GPIO: GPIO 16亮
                _set_gpio_pins(recognized=True)
                print("GPIO set: Recognized face (状态C)")

                # 更新状态
                _gpio_state = "recognized"

                # 取消之前的定时器（如果存在）
                if _recognized_timer is not None:
                    _recognized_timer.cancel()
                    _recognized_timer = None
                if _unknown_timer is not None:
                    _unknown_timer.cancel()
                    _unknown_timer = None

                # 启动新的定时器，5秒后关闭
                _recognized_timer = Timer(_hold_time, _delayed_turn_off_recognized)
                _recognized_timer.daemon = True
                _recognized_timer.start()
                print(f"GPIO set: Scheduled turn off recognized LED in {_hold_time} seconds")

        # 如果检测到未知人脸（状态B）
        elif unknown:
            # 如果当前状态不是"unknown"
            if _gpio_state != "unknown":
                # 设置GPIO: GPIO 24亮
                _set_gpio_pins(unknown=True)
                print("GPIO set: Unknown face (状态B)")

                # 更新状态
                _gpio_state = "unknown"

                # 取消之前的定时器（如果存在）
                if _recognized_timer is not None:
                    _recognized_timer.cancel()
                    _recognized_timer = None
                if _unknown_timer is not None:
                    _unknown_timer.cancel()
                    _unknown_timer = None

                # 启动新的定时器，5秒后关闭
                _unknown_timer = Timer(_hold_time, _delayed_turn_off_unknown)
                _unknown_timer.daemon = True
                _unknown_timer.start()
                print(f"GPIO set: Scheduled turn off unknown LED in {_hold_time} seconds")

        # 如果没有检测到人脸（状态A）
        else:
            # 如果当前没有定时器在运行，或者状态不是standby
            if (_recognized_timer is None and _unknown_timer is None) or _gpio_state != "standby":
                # 设置GPIO: GPIO 23亮（状态A）
                _set_gpio_pins(standby=True)
                print("GPIO set: No face detected (状态A)")

                # 更新状态
                _gpio_state = "standby"
    except Exception as e:
        print(f"GPIO control error: {str(e)}")

# 初始化 picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (480, 270)},
    controls={"FrameRate": 15}
    ))
picam2.start()

# 加载 OpenCV 自带的 Haar 级联分类器（用于初步人脸检测）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 预加载已知人脸数据
known_face_encodings = []
known_face_names = []

# 加载已知人脸（自动从文件夹读取）
known_faces_dir = "Profile_Pictures"  # 存放已知人脸图片的文件夹

# 遍历文件夹中的所有图片
for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(known_faces_dir, filename)
        name = os.path.splitext(filename)[0]  # 使用文件名作为人名

        try:
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:  # 确保编码成功
                # 每张图片只取第一个检测到的人脸编码
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"Loaded: {name}")
            else:
                print(f"Warning: No face detected in {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")

if not known_face_encodings:
    print("Error: No face data loaded!")
    exit()

# 提高精度的参数
TOLERANCE = 0.45  # 匹配容忍度，值越小越严格
FRAME_SKIP = 2    # 跳帧处理，减少计算量
frame_count = 0

while True:
    frame = picam2.capture_array()  # 读取摄像头画面
    frame_count += 1

    # 跳帧处理，减少计算量
    if frame_count % FRAME_SKIP != 0:
        continue

    # 转换为RGB格式
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 缩小图像以加快处理速度
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)

    # OpenCV 进行初步人脸检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),  # 增大最小尺寸，过滤小误检
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # 如果没有检测到人脸，设置为状态A（无脸状态）
    if len(faces) == 0:
        # 设置GPIO状态A
        set_gpio_state(recognized=False, unknown=False)

        # 显示画面
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # 进一步用 face_recognition 进行人脸比对
    face_locations = face_recognition.face_locations(small_frame, model="cnn")  # 使用更精确的CNN模型
    face_encodings = face_recognition.face_encodings(small_frame, face_locations, num_jitters=2)  # 多次采样提高精度

    # 用于跟踪识别结果
    known_face_detected = False
    unknown_face_detected = False
    best_confidence = 0

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 恢复原始坐标（因为我们缩小了图像）
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # 计算人脸编码与已知人脸的距离
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        # 检查是否匹配
        if face_distances[best_match_index] < TOLERANCE:
            name = known_face_names[best_match_index]
            confidence = 1 - face_distances[best_match_index]

            # 更新已知人脸检测状态
            if confidence > best_confidence:
                best_confidence = confidence
            known_face_detected = True
        else:
            name = "Unknown"
            confidence = 0

            # 更新未知人脸检测状态
            if not known_face_detected:
                unknown_face_detected = True

        # 画框和标签
        if name != "Unknown":
            # 已知人脸用绿色
            color = (0, 255, 0)
            thickness = 3
        else:
            # 未知人脸用红色
            color = (0, 0, 255)
            thickness = 2

        cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)

        # 显示名称和置信度
        label = f"{name} ({confidence:.2f})"
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness)

    # 根据识别结果控制 GPIO 状态
    if known_face_detected:
        # 状态C：识别到已知人脸，GPIO 16亮起5秒后关闭
        print(f"已知人脸检测到，置信度: {best_confidence:.2f}")
        set_gpio_state(recognized=True, unknown=False)
    elif unknown_face_detected:
        # 状态B：检测到未知人脸，GPIO 24亮起5秒后关闭
        print("未知人脸检测到")
        set_gpio_state(recognized=False, unknown=True)
    else:
        # 状态A：无人脸状态，GPIO 23常亮
        print("没有检测到人脸")
        set_gpio_state(recognized=False, unknown=False)

    # 显示画面
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

try:
    # 清理资源
    cv2.destroyAllWindows()
    picam2.close()

    # 清理 GPIO 资源
    print("清理 GPIO 资源...")

    # 取消所有定时器
    if _recognized_timer is not None:
        _recognized_timer.cancel()
        _recognized_timer = None
    if _unknown_timer is not None:
        _unknown_timer.cancel()
        _unknown_timer = None

    # 关闭所有 LED
    with _gpio_lock:
        GPIO.output(GPIO_LED_RECOGNIZED, GPIO.LOW)
        GPIO.output(GPIO_LED_UNKNOWN, GPIO.LOW)
        GPIO.output(GPIO_LED_STANDBY, GPIO.LOW)
        time.sleep(0.3)
        # 清理 GPIO
        GPIO.cleanup()

    print("程序正常退出")
except Exception as e:
    print(f"清理资源时出错: {str(e)}")
