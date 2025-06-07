#!/usr/bin/env python3
"""
人脸图像采集脚本
功能：采集25张人脸图像，用于3FRAS.py人脸识别系统
使用方法：
1. 运行脚本: python face_capture.py
2. 输入姓名（将作为图像文件名和识别名称）
3. 面对摄像头，保持面部在框内
4. 按空格键拍摄照片，共需拍摄25张
5. 完成后，图像将保存到Profile_Pictures目录
"""

from picamera2 import Picamera2
import cv2
import numpy as np
import os
import time
import argparse

# 参数设置
DEFAULT_OUTPUT_DIR = "Profile_Pictures"  # 默认输出目录，与3FRAS.py使用的目录相同
DEFAULT_IMAGE_COUNT = 25  # 默认采集图像数量
DEFAULT_RESOLUTION = (1920, 1080)  # 提高默认分辨率以获取更清晰的图像
DEFAULT_PREVIEW_SCALE = 0.6  # 预览窗口缩放比例

def create_output_directory(directory):
    """创建输出目录（如果不存在）"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")
    return directory

def initialize_camera(resolution):
    """初始化PiCamera2"""
    try:
        picam2 = Picamera2()
        # 使用更高质量的配置，提高清晰度
        config = picam2.create_still_configuration(
            main={"size": resolution, "format": "RGB888"},
            controls={
                "FrameRate": 30,
                "AwbEnable": 1,
                "NoiseReductionMode": 2,  # 增强降噪
                "Sharpness": 10,          # 增加锐度
                "Contrast": 1.2,          # 增加对比度
                "Saturation": 1.1,        # 增加饱和度
                "ExposureValue": 0,       # 自动曝光
                "Brightness": 0.1         # 轻微增加亮度
            }
        )
        picam2.configure(config)
        picam2.start()

        # 等待自动曝光和白平衡稳定
        time.sleep(2)
        print("摄像头初始化成功")
        return picam2
    except Exception as e:
        print(f"摄像头初始化失败: {str(e)}")
        return None

def detect_face(frame, face_cascade):
    """检测人脸，使用优化参数提高检测质量"""
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 应用直方图均衡化提高对比度
    gray = cv2.equalizeHist(gray)

    # 使用级联分类器检测人脸，优化参数
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,  # 降低缩放因子，提高检测率
        minNeighbors=6,    # 增加最小邻居数，减少误检
        minSize=(60, 60),  # 增加最小人脸尺寸
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # 如果没有检测到人脸，尝试使用更宽松的参数
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    return faces

def enhance_image(image):
    """增强图像质量，提高清晰度"""
    # 确保图像尺寸合适
    if image.shape[0] < 200 or image.shape[1] < 200:
        # 放大小图像
        image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 应用双边滤波 - 保留边缘细节的同时减少噪点
    denoised = cv2.bilateralFilter(image, 9, 75, 75)

    # 转换为YUV颜色空间进行处理
    yuv = cv2.cvtColor(denoised, cv2.COLOR_BGR2YUV)

    # 自适应直方图均衡化 - 仅应用于Y通道（亮度）
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    yuv[:,:,0] = clahe.apply(yuv[:,:,0])

    # 转回BGR
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # 应用更强的锐化
    kernel = np.array([[-1, -1, -1],
                      [-1, 9.5, -1],
                      [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    # 增加对比度
    alpha = 1.2  # 对比度增强因子
    beta = 5     # 亮度增强因子
    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)

    # 应用轻微的高斯模糊去除过度锐化产生的噪点
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)

    return enhanced

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='人脸图像采集工具')
    parser.add_argument('--name', type=str, help='人脸图像的名称（不含扩展名）')
    parser.add_argument('--count', type=int, default=DEFAULT_IMAGE_COUNT, help=f'要采集的图像数量（默认：{DEFAULT_IMAGE_COUNT}）')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR, help=f'输出目录（默认：{DEFAULT_OUTPUT_DIR}）')
    args = parser.parse_args()

    # 获取人名
    person_name = args.name
    while not person_name:
        person_name = input("请输入姓名（将用作识别名称）: ")

    # 创建输出目录
    output_dir = create_output_directory(args.output)

    # 初始化摄像头
    picam2 = initialize_camera(DEFAULT_RESOLUTION)
    if picam2 is None:
        return

    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

    # 初始化变量
    captured_count = 0
    target_count = args.count

    print(f"\n准备为 {person_name} 采集 {target_count} 张人脸图像")
    print("按空格键拍照，按ESC键退出")

    while captured_count < target_count:
        # 捕获帧
        frame = picam2.capture_array()

        # 转换为BGR格式（OpenCV格式）
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 检测人脸
        faces = detect_face(frame, face_cascade)

        # 在帧上绘制信息
        display_frame = frame.copy()

        # 显示进度
        cv2.putText(display_frame, f"已采集: {captured_count}/{target_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示操作提示
        cv2.putText(display_frame, "空格键: 拍照  ESC: 退出", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 如果检测到人脸，绘制人脸框
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # 绘制人脸框
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # 显示人脸质量提示
                if w < 100 or h < 100:
                    cv2.putText(display_frame, "请靠近一点", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif w > 300 or h > 300:
                    cv2.putText(display_frame, "请后退一点", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, "位置合适", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # 未检测到人脸
            cv2.putText(display_frame, "未检测到人脸", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 调整预览窗口大小
        preview_width = int(display_frame.shape[1] * DEFAULT_PREVIEW_SCALE)
        preview_height = int(display_frame.shape[0] * DEFAULT_PREVIEW_SCALE)
        preview_frame = cv2.resize(display_frame, (preview_width, preview_height))

        # 显示预览
        cv2.imshow("人脸采集", preview_frame)

        # 处理按键
        key = cv2.waitKey(1) & 0xFF

        # 按ESC键退出
        if key == 27:
            break

        # 按空格键拍照
        if key == 32:
            # 确保检测到人脸
            if len(faces) > 0:
                # 使用最大的人脸
                if len(faces) > 1:
                    # 按面积排序，选择最大的人脸
                    face_areas = [w*h for (x, y, w, h) in faces]
                    largest_face_idx = face_areas.index(max(face_areas))
                    x, y, w, h = faces[largest_face_idx]
                else:
                    x, y, w, h = faces[0]

                # 提取人脸区域（更大范围以包含完整面部特征）
                expand_factor = 0.5  # 增加扩展因子以包含更多面部周围区域
                ex = max(0, int(x - w * expand_factor / 2))
                ey = max(0, int(y - h * expand_factor / 2))
                ew = min(frame.shape[1] - ex, int(w * (1 + expand_factor)))
                eh = min(frame.shape[0] - ey, int(h * (1 + expand_factor)))

                face_img = frame[ey:ey+eh, ex:ex+ew]

                # 检查图像质量
                if face_img.shape[0] < 100 or face_img.shape[1] < 100:
                    print("图像太小，请靠近摄像头")
                    continue

                # 检查亮度
                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray_face)
                if brightness < 50:
                    print("图像太暗，请增加光线")
                    continue
                elif brightness > 200:
                    print("图像太亮，请减少光线")
                    continue

                # 增强图像质量
                enhanced_face = enhance_image(face_img)

                # 生成文件名（格式：姓名_序号.jpg）
                # 注意：系统已修改为能够自动提取"_"前的部分作为真正的识别名称
                filename = f"{person_name}_{captured_count+1}.jpg"
                filepath = os.path.join(output_dir, filename)

                # 显示文件命名信息
                print(f"文件将保存为: {filename}")
                print(f"系统将使用 '{person_name}' 作为识别名称")

                # 保存高质量图像
                save_params = [cv2.IMWRITE_JPEG_QUALITY, 95]  # 设置JPEG质量为95%
                cv2.imwrite(filepath, enhanced_face, save_params)
                print(f"已保存: {filepath}")

                # 更新计数
                captured_count += 1

                # 显示已捕获的图像，并保持更长时间以便查看
                display_img = cv2.resize(enhanced_face, (400, int(400 * enhanced_face.shape[0] / enhanced_face.shape[1])))
                cv2.putText(display_img, f"已保存 {captured_count}/{target_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("已捕获", display_img)
                cv2.waitKey(800)  # 显示更长时间

                # 等待一段时间，避免连续拍摄
                time.sleep(0.5)
            else:
                print("未检测到人脸，请调整位置")

    # 清理
    picam2.stop()
    cv2.destroyAllWindows()

    if captured_count > 0:
        print(f"\n采集完成! 已保存 {captured_count} 张图像到 {output_dir} 目录")
        print(f"图像将被用作 {person_name} 的人脸识别数据")
        print(f"3FRAS.py 将自动加载这些图像用于人脸识别")
    else:
        print("\n未采集任何图像")

if __name__ == "__main__":
    main()
