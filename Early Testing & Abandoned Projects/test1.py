from picamera2 import Picamera2
import cv2
import face_recognition
import numpy as np
import RPi.GPIO as GPIO
import time

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

# 初始化 picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (480, 270)}))
picam2.start()

# 加载 OpenCV 自带的 Haar 级联分类器（用于初步人脸检测）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 预加载已知人脸数据
known_face_encodings = []
known_face_names = []

# 加载已知人脸（示例：已存的两个人脸图片）
known_images = ["Profile_Pictures/BLACK MAN.jpg"] # 替换为你的图片路径
names = ["BLACK MAN "]  # 替换为对应的人名

for img_path, name in zip(known_images, names):
    image = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(image)
    if encoding:  # 确保编码成功
        known_face_encodings.append(encoding[0])
        known_face_names.append(name)
    elif not encoding:
        print(f"Warning: No face detected in {img_path}")

while True:
    frame = picam2.capture_array()  # 读取摄像头画面
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # face_recognition 需要 RGB 图像

    # OpenCV 进行初步人脸检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 进一步用 face_recognition 进行人脸比对
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # 默认状态：如果没有检测到人脸，保持待机状态
    if len(face_locations) == 0:
        # 切换到待机状态
        GPIO.output(GPIO_LED_STANDBY, GPIO.HIGH)
        GPIO.output(GPIO_LED_UNKNOWN, GPIO.LOW)
        GPIO.output(GPIO_LED_RECOGNIZED, GPIO.LOW)

    recognized_face = False  # 标记是否识别到已知人脸

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 计算人脸编码与已知人脸的距离
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # 计算欧氏距离，找到最相近的人脸
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:  # 如果匹配成功
            name = known_face_names[best_match_index]
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

        # 画框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)  # 显示图像

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.close()

# 清理 GPIO 资源
GPIO.output(GPIO_LED_STANDBY, GPIO.LOW)
GPIO.output(GPIO_LED_UNKNOWN, GPIO.LOW)
GPIO.output(GPIO_LED_RECOGNIZED, GPIO.LOW)
GPIO.cleanup()
print("程序已退出，GPIO 资源已释放")
