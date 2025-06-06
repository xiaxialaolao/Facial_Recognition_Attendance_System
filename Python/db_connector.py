#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mysql.connector
import logging
import time
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DBConnector")

class DBConnector:
    """数据库连接器，用于处理与MySQL数据库的连接和操作"""

    def __init__(self, host="127.0.0.1", user="xiaxialaolao", password="xiaxialaolao",
                 database="Facial_Recognition_Attendance_System"):
        """初始化数据库连接参数

        Args:
            host (str): 数据库主机地址
            user (str): 数据库用户名
            password (str): 数据库密码
            database (str): 数据库名称
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None

        # 记录最近打卡记录的缓存，用于防止重复打卡
        # 格式: {employee_id: {'cam0': timestamp, 'cam1': timestamp}}
        self.recent_attendance = {}

        # 重复打卡的时间间隔（15分钟）
        self.attendance_interval = 15 * 60  # 15分钟 = 900秒
        logger.info(f"Attendance interval set to {self.attendance_interval} seconds ({self.attendance_interval/60} minutes)")

    def connect(self):
        """连接到MySQL数据库"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            self.cursor = self.connection.cursor()
            logger.info("Successfully connected to database")
            return True
        except mysql.connector.Error as err:
            logger.error(f"Database connection error: {err}")
            return False

    def disconnect(self):
        """断开与数据库的连接"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("Database connection closed")

    def is_connected(self):
        """检查是否已连接到数据库"""
        if self.connection:
            return self.connection.is_connected()
        return False

    def record_attendance(self, employee_id, camera_source, session_type=None):
        """记录员工的打卡信息

        Args:
            employee_id (int): 员工ID
            camera_source (str): 摄像头来源，'cam0'表示入口摄像头，'cam1'表示出口摄像头
            session_type (str, optional): 打卡类型，'in'表示上班打卡，'out'表示下班打卡。
                                         如果为None，则根据camera_source自动确定：
                                         cam0 -> in（上班打卡）
                                         cam1 -> out（下班打卡）

        Returns:
            bool: 操作是否成功
            str: 结果消息
        """
        if not self.is_connected():
            if not self.connect():
                return False, "Database connection failed"

        # 检查是否是重复打卡（15分钟内同一个用户在同一个摄像头）
        current_time = time.time()

        # 如果员工ID不在缓存中，初始化
        if employee_id not in self.recent_attendance:
            self.recent_attendance[employee_id] = {}

        # 检查是否在15分钟内已经打过卡
        if camera_source in self.recent_attendance[employee_id]:
            last_time = self.recent_attendance[employee_id][camera_source]
            time_diff = current_time - last_time
            if time_diff < self.attendance_interval:
                # 添加更详细的日志，帮助调试
                minutes_ago = int(time_diff / 60)
                seconds_ago = int(time_diff % 60)
                logger.debug(f"Repeat attendance detected for employee {employee_id} at {camera_source}. "
                           f"Last record was {minutes_ago} minutes and {seconds_ago} seconds ago. "
                           f"Must wait {int((self.attendance_interval - time_diff) / 60)} more minutes.")
                return False, "repeat_attendance"

        # 如果没有提供session_type，根据camera_source自动确定
        if session_type is None:
            session_type = "in" if camera_source == "cam0" else "out"

        # 不再需要判断是否迟到

        try:
            # 插入打卡记录
            query = """
            INSERT INTO attendance (employee_id, session_type)
            VALUES (%s, %s)
            """
            self.cursor.execute(query, (employee_id, session_type))
            self.connection.commit()

            # 更新缓存
            self.recent_attendance[employee_id][camera_source] = current_time

            # 格式化当前时间为可读格式
            current_time_str = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')

            # 记录详细的打卡信息
            logger.info(f"Successfully recorded attendance for employee {employee_id} at {camera_source} with type {session_type}")
            logger.info(f"Next attendance for employee {employee_id} at {camera_source} will be allowed after: "
                       f"{datetime.fromtimestamp(current_time + self.attendance_interval).strftime('%Y-%m-%d %H:%M:%S')}")

            return True, f"Successfully recorded attendance at {current_time_str}"

        except mysql.connector.Error as err:
            logger.error(f"Error recording attendance: {err}")
            return False, f"Database error: {err}"

    def get_employee_id_by_name(self, name):
        """根据员工姓名获取员工ID

        Args:
            name (str): 员工姓名

        Returns:
            int or None: 员工ID，如果未找到则返回None
        """
        if not self.is_connected():
            if not self.connect():
                return None

        try:
            query = "SELECT employee_id FROM users WHERE fullname = %s"
            self.cursor.execute(query, (name,))
            result = self.cursor.fetchone()

            if result:
                return result[0]
            else:
                return None

        except mysql.connector.Error as err:
            logger.error(f"Error querying employee ID: {err}")
            return None

# 单例模式，确保只有一个数据库连接实例
db_connector = DBConnector()
