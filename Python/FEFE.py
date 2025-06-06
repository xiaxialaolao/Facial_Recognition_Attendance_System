#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据收集器 - 处理图像并提取人脸编码（Face Extraction Feature Encoding）
此脚本监控Image_DataSet目录中的新图像，提取人脸编码，并将其保存为JSON文件
改进版：筛选出8个最高质量的图片进行编码
"""

import os
import cv2
import json
import numpy as np
import face_recognition
import time
import logging
import argparse
import signal
import sys
from datetime import datetime
from pathlib import Path
import pytz
import dlib

# 设置时区为亚洲/吉隆坡（马来西亚时间）
kl_tz = pytz.timezone('Asia/Kuala_Lumpur')

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 可以设置为logging.DEBUG以查看更详细的日志
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("FEFE.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FEFE")

# 设置日志级别，可以在这里调整为DEBUG以查看更详细的质量评估信息
# logger.setLevel(logging.DEBUG)

class FEFE:
    def __init__(self, image_dir="Image_DataSet", encoding_dir="Encoding_DataSet"):
        """初始化数据收集器"""
        self.image_dir = image_dir
        self.encoding_dir = encoding_dir

        # 设置最大编码数量为8（只保留8个最高质量的图片）
        self.MAX_ENCODINGS = 8

        # 初始化dlib人脸检测器（用于获取置信度）
        self.face_detector = dlib.get_frontal_face_detector()

        # 确保目录存在
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.encoding_dir, exist_ok=True)

        logger.info(f"Data collector initialized. Image directory: {self.image_dir}, Encoding directory: {self.encoding_dir}, Max encodings per person: {self.MAX_ENCODINGS}")

    def calculate_blur_score(self, image):
        """
        计算图像的模糊度分数（拉普拉斯方差）
        分数越高，图像越清晰
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        else:
            gray = image

        # 计算拉普拉斯方差
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def calculate_face_size_score(self, image, face_location):
        """
        计算人脸大小得分
        人脸越大，分数越高
        """
        if not face_location:
            return 0

        # 获取图像尺寸
        img_height, img_width = image.shape[:2]

        # 获取人脸框
        top, right, bottom, left = face_location
        face_width = right - left
        face_height = bottom - top

        # 计算人脸占图像的比例
        width_ratio = face_width / img_width
        height_ratio = face_height / img_height

        # 人脸面积占比
        area_ratio = width_ratio * height_ratio

        # 如果人脸太小（小于50像素），给予较低分数
        if face_width < 50 or face_height < 50:
            return area_ratio * 0.5

        return area_ratio

    def calculate_face_position_score(self, image):
        """
        计算人脸位置评分
        评估人脸在图像中的位置是否居中
        居中的人脸通常质量更好
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            gray = image

        # 检测人脸，使用不同的上采样次数
        faces_level1 = self.face_detector(gray, 1)  # 上采样1次

        if not faces_level1:
            return 0  # 没有检测到人脸

        # 如果在第一级别检测到人脸，给予较高的分数
        # 我们可以通过检查人脸是否居中来增加评分
        face = faces_level1[0]
        img_height, img_width = gray.shape[:2]
        face_center_x = (face.left() + face.right()) / 2
        face_center_y = (face.top() + face.bottom()) / 2

        # 计算人脸中心点与图像中心点的距离（归一化）
        center_dist_x = abs(face_center_x - img_width/2) / (img_width/2)
        center_dist_y = abs(face_center_y - img_height/2) / (img_height/2)
        center_dist = (center_dist_x + center_dist_y) / 2

        # 居中程度评分（1为完全居中，0为边缘）
        center_score = 1 - center_dist

        # 综合评分：基础分0.7 + 居中加分0.3
        return 0.7 + (center_score * 0.3)

    def evaluate_image_quality(self, image_path):
        """
        Evaluate image quality, returning a composite quality score
        """
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Unable to load image: {image_path}")
            return 0

        # 转换为RGB（face_recognition需要RGB格式）
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 检测人脸
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        if not face_locations:
            return 0

        # 获取第一个人脸位置
        face_location = face_locations[0]

        # 计算各项指标
        blur_score = self.calculate_blur_score(image)
        face_size_score = self.calculate_face_size_score(image, face_location)
        face_position_score = self.calculate_face_position_score(image)

        # 归一化模糊度分数（通常在0-2000之间）
        normalized_blur = min(blur_score / 2000.0, 1.0)

        # 综合评分（可以根据需要调整权重）
        # 模糊度权重0.5，人脸大小权重0.3，人脸位置权重0.2
        quality_score = (normalized_blur * 0.5) + (face_size_score * 0.3) + (face_position_score * 0.2)

        logger.debug(f"Image quality assessment - {os.path.basename(image_path)}: Blur={blur_score:.2f}, Face size={face_size_score:.2f}, Face position={face_position_score:.2f}, Total score={quality_score:.4f}")

        return quality_score

    def scan_directories(self, force_regenerate=False):
        """
        扫描图像目录，查找新图像

        参数:
            force_regenerate: 如果为True，将重新生成所有JSON文件，即使它们已经存在
        """
        if not os.path.exists(self.image_dir):
            logger.warning(f"Image directory does not exist: {self.image_dir}")
            return

        # 获取所有人员目录
        person_dirs = [d for d in os.listdir(self.image_dir)
                      if os.path.isdir(os.path.join(self.image_dir, d))]

        if not person_dirs:
            logger.info("No person directories found")
            return

        logger.info(f"Found {len(person_dirs)} person directories")

        # 处理每个人员目录
        for person_name in person_dirs:
            person_dir = os.path.join(self.image_dir, person_name)
            json_path = os.path.join(self.encoding_dir, f"{person_name}.json")

            # 获取目录中的图像文件
            image_files = [f for f in os.listdir(person_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not image_files:
                logger.info(f"No images found in directory {person_dir}, skipping")
                continue

            # 如果强制重新生成，直接处理所有图像
            if force_regenerate:
                logger.info(f"Force regenerating encoding file for {person_name}")
                # 如果存在旧的JSON文件，先删除
                if os.path.exists(json_path):
                    try:
                        os.remove(json_path)
                        logger.info(f"Deleted old encoding file: {json_path}")
                    except Exception as e:
                        logger.error(f"Error deleting file {json_path}: {str(e)}")
                self.process_person_directory(person_name)
                continue

            # 检查是否有新图像需要处理
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)

                    # 获取已处理的文件列表
                    processed_files = data.get("processed_files", [])

                    # 检查是否有新图像
                    new_images = [f for f in image_files if f not in processed_files]

                    if new_images:
                        logger.info(f"Found {len(new_images)} new images in directory {person_dir}, processing")
                        self.process_person_directory(person_name)
                    else:
                        logger.info(f"No new images to process in directory {person_dir}")
                except Exception as e:
                    logger.error(f"Error reading JSON file {json_path}: {str(e)}")
                    logger.info(f"Will reprocess all images for {person_name}")
                    self.process_person_directory(person_name)
            else:
                logger.info(f"Encoding file for {person_name} not found, processing all images")
                self.process_person_directory(person_name)

    def process_person_directory(self, person_name):
        """处理单个人员目录中的所有图像，筛选最高质量的图片"""
        person_dir = os.path.join(self.image_dir, person_name)
        json_path = os.path.join(self.encoding_dir, f"{person_name}.json")

        # 获取所有图像文件
        image_files = [f for f in os.listdir(person_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            logger.info(f"No images found in directory {person_dir}")
            return

        # 获取已处理的文件列表
        processed_files = []
        existing_encodings = []
        existing_encoding_files = []

        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    if "processed_files" in data:
                        processed_files = data["processed_files"]
                    if "encodings" in data and "encoding_files" in data:
                        existing_encodings = [np.array(enc) for enc in data.get("encodings", [])]
                        existing_encoding_files = data.get("encoding_files", [])
            except Exception as e:
                logger.error(f"Error reading JSON file {json_path}: {str(e)}")

        # 过滤出未处理的图像
        new_image_files = [f for f in image_files if f not in processed_files]

        if not new_image_files and not existing_encodings:
            logger.info(f"No new images to process in directory {person_dir}")
            return

        # 如果有新图像，或者强制重新生成，则评估所有图像质量
        logger.info(f"Evaluating quality of {len(image_files)} images for {person_name}")

        # 存储图像质量评分和对应的编码
        image_quality_scores = []

        # 处理所有图像（包括已处理和未处理的）
        for image_file in image_files:
            image_path = os.path.join(person_dir, image_file)
            try:
                # 评估图像质量
                quality_score = self.evaluate_image_quality(image_path)

                if quality_score > 0:  # 只有检测到人脸的图像才有效
                    # 提取人脸编码
                    face_encoding = self.extract_face_encoding(image_path)
                    if face_encoding is not None:
                        # 保存图像文件名、质量分数和编码
                        image_quality_scores.append({
                            'file': image_file,
                            'score': quality_score,
                            'encoding': face_encoding
                        })
                        logger.info(f"Image quality assessment: {image_file}, score: {quality_score:.4f}")
                    else:
                        logger.warning(f"Failed to extract face encoding: {image_file}")
                else:
                    logger.warning(f"Image quality too low or no face detected: {image_file}")
            except Exception as e:
                logger.error(f"Error processing image {image_file}: {str(e)}")

        # 按质量分数降序排序
        image_quality_scores.sort(key=lambda x: x['score'], reverse=True)

        # 选择最高质量的8张图片（或更少，如果图片不足8张）
        top_images = image_quality_scores[:self.MAX_ENCODINGS]

        if not top_images:
            logger.warning(f"Could not extract any valid face encodings from {person_name}'s images")
            return

        # 提取编码和文件名
        top_encodings = [item['encoding'] for item in top_images]
        top_files = [item['file'] for item in top_images]

        logger.info(f"Selected {len(top_images)} highest quality images for encoding for {person_name}")
        for i, item in enumerate(top_images):
            logger.info(f"  {i+1}. {item['file']} - quality score: {item['score']:.4f}")

        # 更新编码文件，使用新的top_files参数
        self.update_encodings_with_files(person_name, top_encodings, top_files, image_files)

    def extract_face_encoding(self, image_path):
        """从图像中提取人脸编码"""
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Unable to load image: {image_path}")
            return None

        # 转换为RGB（face_recognition需要RGB格式）
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 检测人脸
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        if not face_locations:
            return None

        # 提取编码
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations, model="large")
        if not face_encodings:
            return None

        # 返回第一个人脸的编码
        return face_encodings[0]

    def update_encodings_with_files(self, name, encodings, encoding_files, processed_files):
        """
        更新对应名字的JSON编码文件，保存编码对应的文件名

        参数:
            name: 人名
            encodings: 人脸编码列表
            encoding_files: 编码对应的图像文件名列表
            processed_files: 所有处理过的图像文件名列表
        """
        json_path = os.path.join(self.encoding_dir, f"{name}.json")

        # 获取当前时间（吉隆坡时区）
        now = datetime.now(kl_tz)
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        # 初始化数据结构
        data = {
            "name": name,
            "encodings": [],
            "encoding_files": [],  # 添加编码对应的文件名列表
            "processed_files": [],  # 添加已处理文件列表
            "last_updated": timestamp,  # 添加最后更新时间（吉隆坡时区）
            "timezone": "Asia/Kuala_Lumpur",  # 记录使用的时区
            "encoding_timestamps": []  # 添加编码时间戳列表，用于跟踪每个编码的添加时间
        }

        # 将编码转换为列表格式
        encodings_list = [e.tolist() for e in encodings]

        # 直接使用新的编码和文件名（替换而非追加）
        data["encodings"] = encodings_list
        data["encoding_files"] = encoding_files
        data["encoding_timestamps"] = [timestamp] * len(encodings_list)

        # 添加所有处理过的文件名
        data["processed_files"] = processed_files

        # 保存更新后的数据
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Updated encoding file: {json_path} (Current encodings: {len(data['encodings'])}/{self.MAX_ENCODINGS}, Processed files: {len(data['processed_files'])}, Update time: {timestamp})")

    def update_encodings(self, name, new_encodings, processed_files):
        """
        更新对应名字的JSON编码文件（兼容旧版本）

        参数:
            name: 人名
            new_encodings: 新的人脸编码列表
            processed_files: 处理过的图像文件名列表
        """
        json_path = os.path.join(self.encoding_dir, f"{name}.json")

        # 获取当前时间（吉隆坡时区）
        now = datetime.now(kl_tz)
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        # 初始化数据结构
        data = {
            "name": name,
            "encodings": [],
            "processed_files": [],  # 添加已处理文件列表
            "last_updated": timestamp,  # 添加最后更新时间（吉隆坡时区）
            "timezone": "Asia/Kuala_Lumpur",  # 记录使用的时区
            "encoding_timestamps": []  # 添加编码时间戳列表，用于跟踪每个编码的添加时间
        }

        # 如果文件已存在则加载已有数据
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                # 确保数据结构中有processed_files字段
                if "processed_files" not in data:
                    data["processed_files"] = []

                # 确保数据结构中有encoding_timestamps字段
                if "encoding_timestamps" not in data:
                    # 如果没有时间戳字段，为现有编码创建默认时间戳（当前时间）
                    data["encoding_timestamps"] = [timestamp] * len(data["encodings"])
            except json.JSONDecodeError:
                logger.error(f"JSON file format error: {json_path}, will create a new file")

        # 添加新编码（转换为列表格式）和对应的时间戳
        new_encodings_list = [e.tolist() for e in new_encodings]
        data["encodings"].extend(new_encodings_list)
        data["encoding_timestamps"].extend([timestamp] * len(new_encodings_list))

        # 检查是否超过最大编码数量限制
        if len(data["encodings"]) > self.MAX_ENCODINGS:
            # 计算需要删除的编码数量
            excess_count = len(data["encodings"]) - self.MAX_ENCODINGS
            logger.info(f"Encoding count {len(data['encodings'])} exceeds limit {self.MAX_ENCODINGS}, will delete the oldest {excess_count} encodings")

            # 删除最早的编码和对应的时间戳
            data["encodings"] = data["encodings"][excess_count:]
            data["encoding_timestamps"] = data["encoding_timestamps"][excess_count:]

        # 添加新处理的文件名
        for file in processed_files:
            if file not in data["processed_files"]:
                data["processed_files"].append(file)

        # 更新时间戳（吉隆坡时区）
        now = datetime.now(kl_tz)
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        data["last_updated"] = timestamp
        data["timezone"] = "Asia/Kuala_Lumpur"

        # 保存更新后的数据
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Updated encoding file: {json_path} (Current encodings: {len(data['encodings'])}/{self.MAX_ENCODINGS}, Processed files: {len(data['processed_files'])}, Update time: {timestamp})")

    def run_once(self, force_regenerate=False):
        """
        执行一次扫描和处理

        参数:
            force_regenerate: 如果为True，将重新生成所有JSON文件，即使它们已经存在
        """
        logger.info("Starting to scan image directory...")
        self.scan_directories(force_regenerate)
        logger.info("Scan completed")

    def run_continuous(self, interval=60, force_regenerate=False):
        """
        持续运行，定期扫描新图像

        参数:
            interval: 扫描间隔（秒）
            force_regenerate: 如果为True，将重新生成所有JSON文件，即使它们已经存在
        """
        logger.info(f"Starting continuous monitoring, scan interval: {interval} seconds")
        logger.info("Press Ctrl+C to stop the program at any time")
        print("\n")
        print("="*80)
        print("Data collector started, monitoring for new images...")
        print("Press Ctrl+C to stop the program at any time")
        print("="*80)
        print("\n")

        try:
            while True:
                self.run_once(force_regenerate)
                logger.info(f"Waiting {interval} seconds before next scan...")

                # 使用小的时间间隔来检查中断，使程序能更快响应Ctrl+C
                for _ in range(interval):
                    time.sleep(1)

        except KeyboardInterrupt:
            print("\n")
            print("="*80)
            print("Received Ctrl+C interrupt signal, safely exiting program...")
            print("="*80)
            logger.info("Received Ctrl+C interrupt signal, program safely exited")
            # 这里可以添加清理代码
            print("Program has safely exited")

        except Exception as e:
            logger.error(f"Runtime error: {str(e)}")
            print(f"\nError: {str(e)}")
            raise

# 定义信号处理函数
def signal_handler(sig, frame):
    """处理 Ctrl+C 信号"""
    print("\n")
    print("="*80)
    print("Received Ctrl+C interrupt signal, safely exiting program...")
    print("="*80)
    logger.info("Received Ctrl+C interrupt signal, program safely exited")
    sys.exit(0)

def main():
    """主函数"""
    # 注册信号处理函数
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="Process images and extract face encodings, selecting the highest quality images")
    parser.add_argument("--image-dir", default="Image_DataSet", help="Image directory path")
    parser.add_argument("--encoding-dir", default="Encoding_DataSet", help="Encoding directory path")
    parser.add_argument("--interval", type=int, default=60, help="Scan interval (seconds)")
    parser.add_argument("--once", action="store_true", help="Run only once")
    parser.add_argument("--force", action="store_true", help="Force regeneration of all JSON files, even if they already exist")

    args = parser.parse_args()

    # 打印启动信息
    print("\nData Collector - Process images and extract face encodings (Improved Version)")
    print(f"Will select 8 highest quality images for encoding for each person")
    print(f"Image directory: {args.image_dir}")
    print(f"Encoding directory: {args.encoding_dir}")

    if not args.once:
        print(f"Continuous monitoring mode, scan interval: {args.interval} seconds")
        print("You can use Ctrl+C to stop the program at any time")
    else:
        print("Single run mode")

    if args.force:
        print("Force regeneration of all encoding files")

    print("\nInitializing...\n")

    collector = FEFE(args.image_dir, args.encoding_dir)

    try:
        if args.once:
            collector.run_once(args.force)
            print("\nProgram execution completed")
        else:
            collector.run_continuous(args.interval, args.force)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nProgram error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
