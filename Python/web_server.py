from flask import Flask, Response, render_template
import cv2
import threading
import time
import logging
import os
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WebServer")

# 创建 Flask 应用
app = Flask(__name__)

# 全局变量，存储最新的帧
frame_lock = threading.Lock()
current_frame = None

def set_frame(frame):
    """设置当前帧，供 FRAS.py 调用"""
    global current_frame
    with frame_lock:
        if frame is not None:
            current_frame = frame.copy()

def generate_frames():
    """生成视频流的帧"""
    global current_frame
    
    # 创建一个黑色的初始帧
    blank_frame = np.zeros((720, 1920, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_frame, "Waiting for camera feed...", (50, 360), 
                font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    while True:
        # 获取当前帧
        with frame_lock:
            if current_frame is None:
                # 如果没有帧，使用黑色帧
                output_frame = blank_frame.copy()
            else:
                output_frame = current_frame.copy()
        
        # 将帧编码为 JPEG
        ret, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        
        # 生成 multipart 响应
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # 控制帧率
        time.sleep(0.03)  # 约 30 FPS

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """视频流路由"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def start_server(host='0.0.0.0', port=5000):
    """启动 Web 服务器"""
    try:
        logger.info(f"Starting web server on http://{host}:{port}/video_feed")
        app.run(host=host, port=port, threaded=True)
    except Exception as e:
        logger.error(f"Web server error: {str(e)}")

if __name__ == "__main__":
    # 如果直接运行此文件，启动服务器
    start_server()
