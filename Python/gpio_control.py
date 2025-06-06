#!/usr/bin/env python3
"""
GPIO控制模块
功能：控制与人脸识别系统相关的GPIO引脚
支持两个摄像头的GPIO控制
"""

import RPi.GPIO as GPIO
import time
import logging
from threading import Timer, Lock

# 配置日志
logger = logging.getLogger("GPIO_Control")

# 初始化 GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# 定义 GPIO 引脚 - 摄像头 0 (cam0)
GPIO_LED_RECOGNIZED_CAM0 =13   # 识别到已知人脸时亮起的 LED (状态C)
GPIO_LED_UNKNOWN_CAM0 = 19     # 检测到未知人脸时亮起的 LED (状态B)
GPIO_LED_STANDBY_CAM0 = 26     # 无人脸状态时亮起的 LED (状态A)

# 定义 GPIO 引脚 - 摄像头 1 (cam1)
GPIO_LED_RECOGNIZED_CAM1 = 14  # 识别到已知人脸时亮起的 LED (状态C)
GPIO_LED_UNKNOWN_CAM1 = 15     # 检测到未知人脸时亮起的 LED (状态B)
GPIO_LED_STANDBY_CAM1 = 18     # 无人脸状态时亮起的 LED (状态A)

# 为了向后兼容，保留原始变量名
GPIO_LED_RECOGNIZED = GPIO_LED_RECOGNIZED_CAM0
GPIO_LED_UNKNOWN = GPIO_LED_UNKNOWN_CAM0
GPIO_LED_STANDBY = GPIO_LED_STANDBY_CAM0

# 定义 GPIO 控制变量 - 每个摄像头一组
_gpio_state = {"cam0": "standby", "cam1": "standby"}  # 可能的状态: "standby", "unknown", "recognized"
_standby_timer = {"cam0": None, "cam1": None}         # 定时器，用于延时返回待机状态
_delay_time = 0.3                                     # 无人脸检测后延迟返回待机状态的时间，单位秒
_last_face_time = {"cam0": 0, "cam1": 0}              # 上次检测到人脸的时间
_gpio_lock = Lock()                                   # 添加锁，防止并发访问GPIO

# GPIO初始化标志
GPIO_INITIALIZED = {"cam0": False, "cam1": False}

def initialize_gpio(camera_id="cam0"):
    """初始化GPIO引脚设置

    Args:
        camera_id: 摄像头ID，可以是 "cam0" 或 "cam1"

    Returns:
        bool: 初始化是否成功
    """
    global GPIO_INITIALIZED

    # 根据摄像头ID选择对应的GPIO引脚
    if camera_id == "cam0":
        recognized_pin = GPIO_LED_RECOGNIZED_CAM0
        unknown_pin = GPIO_LED_UNKNOWN_CAM0
        standby_pin = GPIO_LED_STANDBY_CAM0
    elif camera_id == "cam1":
        recognized_pin = GPIO_LED_RECOGNIZED_CAM1
        unknown_pin = GPIO_LED_UNKNOWN_CAM1
        standby_pin = GPIO_LED_STANDBY_CAM1
    else:
        logger.error(f"Invalid camera ID: {camera_id}")
        return False

    try:
        # 设置为输出模式
        GPIO.setup(recognized_pin, GPIO.OUT)
        GPIO.setup(unknown_pin, GPIO.OUT)
        GPIO.setup(standby_pin, GPIO.OUT)

        # 确保所有 LED 都是关闭状态，然后再设置初始状态
        GPIO.output(recognized_pin, GPIO.LOW)
        GPIO.output(unknown_pin, GPIO.LOW)
        GPIO.output(standby_pin, GPIO.LOW)
        time.sleep(0.05)  # 减少延时，加速灯光变化

        # 初始状态A：无人脸状态，待机LED常亮
        GPIO.output(standby_pin, GPIO.HIGH)
        time.sleep(0.05)  # 减少延时，加速灯光变化

        GPIO_INITIALIZED[camera_id] = True
        logger.info(f"Camera {camera_id} GPIO initialization successful")
        return True
    except Exception as e:
        logger.error(f"Camera {camera_id} GPIO initialization failed: {str(e)}")
        GPIO_INITIALIZED[camera_id] = False
        return False

def initialize_all_gpio():
    """初始化所有摄像头的GPIO引脚设置"""
    result_cam0 = initialize_gpio("cam0")
    result_cam1 = initialize_gpio("cam1")
    return result_cam0 and result_cam1

def _set_gpio_pins(recognized=False, unknown=False, standby=False, camera_id="cam0"):
    """直接设置GPIO引脚状态，使用锁防止并发访问

    Args:
        recognized: 是否设置已知人脸状态
        unknown: 是否设置未知人脸状态
        standby: 是否设置待机状态
        camera_id: 摄像头ID，可以是 "cam0" 或 "cam1"
    """
    global GPIO_INITIALIZED

    # 根据摄像头ID选择对应的GPIO引脚
    if camera_id == "cam0":
        recognized_pin = GPIO_LED_RECOGNIZED_CAM0
        unknown_pin = GPIO_LED_UNKNOWN_CAM0
        standby_pin = GPIO_LED_STANDBY_CAM0
    elif camera_id == "cam1":
        recognized_pin = GPIO_LED_RECOGNIZED_CAM1
        unknown_pin = GPIO_LED_UNKNOWN_CAM1
        standby_pin = GPIO_LED_STANDBY_CAM1
    else:
        logger.error(f"Invalid camera ID: {camera_id}")
        return

    # 确保GPIO已初始化
    if not GPIO_INITIALIZED[camera_id]:
        initialize_gpio(camera_id)

    # 确保设置了GPIO模式
    if GPIO.getmode() is None:
        GPIO.setmode(GPIO.BCM)

    with _gpio_lock:  # 使用锁确保GPIO操作的原子性
        try:
            # 先关闭所有LED，防止闪烁
            GPIO.output(recognized_pin, GPIO.LOW)
            GPIO.output(unknown_pin, GPIO.LOW)
            GPIO.output(standby_pin, GPIO.LOW)
            time.sleep(0.02)  # 极短延时，加速灯光变化

            if recognized:
                # 状态C：识别到已知人脸
                GPIO.output(recognized_pin, GPIO.HIGH)
                # 减少日志输出
                # logger.info(f"GPIO set: 摄像头 {camera_id} Recognized face LED ON")
            elif unknown:
                # 状态B：检测到未知人脸
                GPIO.output(unknown_pin, GPIO.HIGH)
                # 减少日志输出
                # logger.info(f"GPIO set: 摄像头 {camera_id} Unknown face LED ON")
            elif standby:
                # 状态A：无人脸状态
                GPIO.output(standby_pin, GPIO.HIGH)
                # 减少日志输出
                # logger.info(f"GPIO set: 摄像头 {camera_id} Standby LED ON")
        except Exception as e:
            logger.error(f"Camera {camera_id} GPIO pin control error: {str(e)}")

def _delayed_return_to_standby(camera_id="cam0"):
    """延时返回待机状态的定时器回调函数

    Args:
        camera_id: 摄像头ID，可以是 "cam0" 或 "cam1"
    """
    global _gpio_state, _standby_timer

    try:
        # 关闭当前LED，打开待机LED（状态A）
        _set_gpio_pins(standby=True, camera_id=camera_id)
        logger.info(f"Camera {camera_id} GPIO state change: Delay ended, returning to standby LED")

        # 更新状态
        _gpio_state[camera_id] = "standby"
        _standby_timer[camera_id] = None
    except Exception as e:
        logger.error(f"Camera {camera_id} GPIO delayed return to standby error: {str(e)}")

def set_gpio_state(recognized=False, unknown=False, camera_id="cam0"):
    """设置GPIO状态，根据人脸识别状态控制LED

    Args:
        recognized: 是否识别到已知人脸
        unknown: 是否检测到未知人脸
        camera_id: 摄像头ID，可以是 "cam0" 或 "cam1"
    """
    global _gpio_state, _standby_timer, _last_face_time, GPIO_INITIALIZED

    # 如果GPIO未初始化，先初始化
    if not GPIO_INITIALIZED[camera_id]:
        initialize_gpio(camera_id)

    try:
        # 如果检测到人脸（已知或未知）
        if recognized or unknown:
            # 更新最后检测到人脸的时间
            _last_face_time[camera_id] = time.time()

            # 如果有待机定时器在运行，取消它
            if _standby_timer[camera_id] is not None:
                _standby_timer[camera_id].cancel()
                _standby_timer[camera_id] = None
                # 减少日志输出
                # logger.info(f"摄像头 {camera_id} GPIO: Canceled standby timer due to face detection")

            # 如果识别到已知人脸（状态C）
            if recognized:
                # 如果当前状态不是"recognized"
                if _gpio_state[camera_id] != "recognized":
                    # 设置GPIO
                    _set_gpio_pins(recognized=True, camera_id=camera_id)
                    logger.info(f"Camera {camera_id} GPIO state change: Known face detected - switching to Recognized LED")

                    # 更新状态
                    _gpio_state[camera_id] = "recognized"

            # 如果检测到未知人脸（状态B）
            elif unknown:
                # 如果当前状态不是"unknown"
                if _gpio_state[camera_id] != "unknown":
                    # 设置GPIO
                    _set_gpio_pins(unknown=True, camera_id=camera_id)
                    logger.info(f"Camera {camera_id} GPIO state change: Unknown face detected - switching to Unknown LED")

                    # 更新状态
                    _gpio_state[camera_id] = "unknown"

        # 如果没有检测到人脸（状态A）
        else:
            # 获取对应的待机LED引脚
            standby_pin = GPIO_LED_STANDBY_CAM0 if camera_id == "cam0" else GPIO_LED_STANDBY_CAM1

            # 如果当前状态不是standby且没有定时器在运行
            if _gpio_state[camera_id] != "standby" and _standby_timer[camera_id] is None:
                # 创建定时器回调函数，传递camera_id参数
                def callback():
                    _delayed_return_to_standby(camera_id)

                # 启动定时器，延迟返回待机状态
                _standby_timer[camera_id] = Timer(_delay_time, callback)
                _standby_timer[camera_id].daemon = True
                _standby_timer[camera_id].start()
                logger.info(f"Camera {camera_id} GPIO state change: No face detected - returning to standby state in {_delay_time} seconds")

            # 如果已经是待机状态但没有LED亮起
            elif _gpio_state[camera_id] == "standby" and not GPIO.input(standby_pin):
                # 设置GPIO: 待机LED亮（状态A）
                _set_gpio_pins(standby=True, camera_id=camera_id)
                logger.info(f"Camera {camera_id} GPIO state change: Ensuring standby LED is on")
    except Exception as e:
        logger.error(f"Camera {camera_id} GPIO control error: {str(e)}")

def cleanup_gpio(camera_id=None):
    """清理GPIO资源

    Args:
        camera_id: 摄像头ID，可以是 "cam0" 或 "cam1"，如果为None则清理所有摄像头的GPIO

    Returns:
        bool: 如果GPIO已初始化并成功清理则返回True，否则返回False
    """
    try:
        logger.info("Cleaning up GPIO resources...")
        global GPIO_INITIALIZED

        # 如果未指定摄像头ID，则清理所有摄像头
        camera_ids = ["cam0", "cam1"] if camera_id is None else [camera_id]

        success = True
        for cam_id in camera_ids:
            # 取消定时器
            if _standby_timer[cam_id] is not None:
                _standby_timer[cam_id].cancel()
                _standby_timer[cam_id] = None

            # 只有在GPIO已初始化的情况下才进行清理
            if GPIO_INITIALIZED[cam_id]:
                # 确保设置了GPIO模式
                if GPIO.getmode() is None:
                    GPIO.setmode(GPIO.BCM)

                # 获取对应的GPIO引脚
                if cam_id == "cam0":
                    recognized_pin = GPIO_LED_RECOGNIZED_CAM0
                    unknown_pin = GPIO_LED_UNKNOWN_CAM0
                    standby_pin = GPIO_LED_STANDBY_CAM0
                else:
                    recognized_pin = GPIO_LED_RECOGNIZED_CAM1
                    unknown_pin = GPIO_LED_UNKNOWN_CAM1
                    standby_pin = GPIO_LED_STANDBY_CAM1

                # 关闭所有 LED
                with _gpio_lock:
                    GPIO.output(recognized_pin, GPIO.LOW)
                    GPIO.output(unknown_pin, GPIO.LOW)
                    GPIO.output(standby_pin, GPIO.LOW)
                    time.sleep(0.02)  # 极短延时，加速灯光变化

                GPIO_INITIALIZED[cam_id] = False
                logger.info(f"Camera {cam_id} GPIO resources cleaned up")
            else:
                logger.info(f"Camera {cam_id} GPIO was not initialized, nothing to clean up")
                success = success and False

        # 如果所有摄像头的GPIO都已清理，则清理整个GPIO系统
        if camera_id is None and not any(GPIO_INITIALIZED.values()):
            GPIO.cleanup()
            logger.info("All GPIO resources cleaned up")

        return success
    except Exception as e:
        logger.error(f"Error cleaning up GPIO resources: {str(e)}")
        return False

# 为了兼容现有代码，提供一些别名函数
def set_recognized_face(camera_id="cam0"):
    """设置为已知人脸状态"""
    set_gpio_state(recognized=True, unknown=False, camera_id=camera_id)

def set_unknown_face(camera_id="cam0"):
    """设置为未知人脸状态"""
    set_gpio_state(recognized=False, unknown=True, camera_id=camera_id)

def set_standby_state(camera_id="cam0"):
    """设置为待机状态"""
    set_gpio_state(recognized=False, unknown=False, camera_id=camera_id)

# 初始化所有摄像头的GPIO
initialize_all_gpio()