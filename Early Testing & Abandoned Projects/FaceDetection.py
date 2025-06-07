from picamera2 import Picamera2
import cv2

# 初始化 picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (800, 450)}))
picam2.start()

# 加载 OpenCV 自带的 Haar 级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    frame = picam2.capture_array()  # 读取摄像头画面
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图（提高识别率）

    # 进行人脸检测
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 在检测到的人脸位置画框
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)  # 显示图像

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.close()
