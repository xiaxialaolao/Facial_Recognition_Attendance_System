from picamera2 import Picamera2
import cv2
import numpy as np
import time
import os
import sys

def verify_model_files(model_path):
    required_files = [
        "deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel"
    ]
    
    missing_files = []
    for file in required_files:
        full_path = os.path.join(model_path, file)
        if not os.path.exists(full_path):
            missing_files.append(file)
        else:
            # 检查文件大小
            size = os.path.getsize(full_path)
            if size == 0:
                missing_files.append(f"{file} (空文件)")
            
    return len(missing_files) == 0, missing_files

# 初始化 picamera2
picam2 = Picamera2()

# 设置相机格式和分辨率（使用更保守的设置）
preview_config = picam2.create_preview_configuration(
    main={
        "size": (640, 480),  # 降低分辨率
        "format": "RGB888"
    }
)
picam2.configure(preview_config)
picam2.start()

# 加载 OpenCV DNN 模型
model_path = "/home/xiaxialaolao/FRAS_env/model/"

# 验证模型文件
models_ok, missing = verify_model_files(model_path)
if not models_ok:
    print(f"错误：缺少或损坏的模型文件: {', '.join(missing)}")
    print("正在尝试重新下载模型文件...")
    
    # 这里可以添加自动下载模型的代码
    sys.exit(1)

try:
    # 加载模型
    print("正在加载DNN模型...")
    prototxt_path = os.path.join(model_path, "deploy.prototxt")
    model_path = os.path.join(model_path, "res10_300x300_ssd_iter_140000.caffemodel")
    
    # 验证文件内容
    with open(prototxt_path, 'r') as f:
        if len(f.read().strip()) == 0:
            raise Exception("deploy.prototxt 文件为空")
            
    # 加载模型
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    # 确保模型加载成功
    if net.empty():
        raise Exception("模型加载失败")
    
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    sys.exit(1)

# 设置后端为CPU（更稳定）
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

print("开始视频捕获...")

frame_count = 0
start_time = time.time()

while True:
    try:
        # 捕获帧
        frame = picam2.capture_array()
        
        # 确保帧格式正确
        if frame.dtype != np.uint8:
            frame = cv2.convertScaleAbs(frame)
        
        if len(frame.shape) == 2:  # 如果是灰度图
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
        h, w = frame.shape[:2]

        # 准备输入数据（修改均值参数）
        blob = cv2.dnn.blobFromImage(
            frame, 
            1.0,  # scalefactor
            (300, 300),  # size
            [104, 117, 123],  # mean
            swapRB=True,  # OpenCV使用BGR，而模型期望RGB
            crop=False
        )
        
        # 设置网络输入
        net.setInput(blob)
        
        try:
            # 执行前向传播
            detections = net.forward()
            
            # 计算并显示FPS
            frame_count += 1
            if frame_count % 30 == 0:  # 每30帧更新一次FPS
                fps = frame_count / (time.time() - start_time)
                print(f"FPS: {fps:.2f}")
                
        except cv2.error as e:
            print(f"DNN推理错误: {str(e)}")
            continue

        # 设置置信度阈值
        confidence_threshold = 0.5

        # 处理检测结果
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # 确保坐标在图像范围内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # 绘制边界框和置信度
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{confidence * 100:.1f}%"
                y1 = max(y1 - 10, 0)
                cv2.putText(frame, text, (x1, y1),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示FPS
        cv2.putText(frame, 
                    f"FPS: {frame_count/(time.time()-start_time):.1f}", 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)

        # 显示图像
        cv2.imshow("Face Detection", frame)
        
        # 检查退出条件
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    except Exception as e:
        print(f"错误: {str(e)}")
        continue

# 清理资源
print("正在清理资源...")
cv2.destroyAllWindows()
picam2.stop()
picam2.close()
print("程序已退出")
