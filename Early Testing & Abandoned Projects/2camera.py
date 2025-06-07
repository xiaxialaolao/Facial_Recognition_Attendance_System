from picamera2 import Picamera2
import cv2
import threading
import time

# 初始化两个摄像头
cam1 = Picamera2(0)
cam2 = Picamera2(1)

cam1.configure(cam1.create_preview_configuration(main={"size": (640, 480)}))
cam2.configure(cam2.create_preview_configuration(main={"size": (640, 480)}))

cam1.start()
cam2.start()

frame1 = None
frame2 = None

# 捕捉线程函数
def update_camera(cam, frame_holder, index):
    while True:
        try:
            frame_holder[index] = cam.capture_array()
        except Exception as e:
            print(f"Camera {index} error: {e}")
            break

# 启动两个线程
frames = [None, None]
threading.Thread(target=update_camera, args=(cam1, frames, 0), daemon=True).start()
threading.Thread(target=update_camera, args=(cam2, frames, 1), daemon=True).start()

print("按 ESC 键退出")

# 主显示循环
while True:
    if frames[0] is not None:
        cv2.imshow("Camera 1", frames[0])
    if frames[1] is not None:
        cv2.imshow("Camera 2", frames[1])

    if cv2.waitKey(1) == 27:  # ESC 键
        break

cv2.destroyAllWindows()
cam1.close()
cam2.close()
