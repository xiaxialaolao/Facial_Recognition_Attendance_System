#!/usr/bin/env python3
"""
GPIO控制模块
功能：控制与人脸识别系统相关的GPIO引脚
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

# 定义 GPIO 引脚
GPIO_LED_RECOGNIZED = 16  # 识别到已知人脸时亮起的 LED (状态C)
GPIO_LED_UNKNOWN = 24     # 检测到未知人脸时亮起的 LED (状态B)
GPIO_LED_STANDBY = 23     # 无人脸状态时亮起的 LED (状态A)

# 定义 GPIO 控制变量
_gpio_state = "standby"  # 可能的状态: "standby", "unknown", "recognized"
_standby_timer = None    # 定时器，用于延时返回待机状态
_delay_time = 1.0        # 无人脸检测后延迟返回待机状态的时间，单位秒（1秒）
_last_face_time = 0      # 上次检测到人脸的时间
_gpio_lock = Lock()      # 添加锁，防止并发访问GPIO

# GPIO初始化标志
GPIO_INITIALIZED = False

def initialize_gpio():
    """初始化GPIO引脚设置"""
    global GPIO_INITIALIZED

    try:
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

        GPIO_INITIALIZED = True
        logger.info("GPIO初始化成功")
        return True
    except Exception as e:
        logger.error(f"GPIO初始化失败: {str(e)}")
        GPIO_INITIALIZED = False
        return False

def _set_gpio_pins(recognized=False, unknown=False, standby=False):
    """直接设置GPIO引脚状态，使用锁防止并发访问"""
    global GPIO_INITIALIZED

    # 确保GPIO已初始化
    if not GPIO_INITIALIZED:
        initialize_gpio()

    # 确保设置了GPIO模式
    if GPIO.getmode() is None:
        GPIO.setmode(GPIO.BCM)

    with _gpio_lock:  # 使用锁确保GPIO操作的原子性
        try:
            # 先关闭所有LED，防止闪烁
            GPIO.output(GPIO_LED_RECOGNIZED, GPIO.LOW)
            GPIO.output(GPIO_LED_UNKNOWN, GPIO.LOW)
            GPIO.output(GPIO_LED_STANDBY, GPIO.LOW)
            time.sleep(0.3)  # 短暂延时，确保状态稳定

            if recognized:
                # 状态C：识别到已知人脸，GPIO 16亮
                GPIO.output(GPIO_LED_RECOGNIZED, GPIO.HIGH)
                # 减少日志输出
                # logger.info("GPIO set: Recognized face LED (GPIO 16) ON")
            elif unknown:
                # 状态B：检测到未知人脸，GPIO 24亮
                GPIO.output(GPIO_LED_UNKNOWN, GPIO.HIGH)
                # 减少日志输出
                # logger.info("GPIO set: Unknown face LED (GPIO 24) ON")
            elif standby:
                # 状态A：无人脸状态，GPIO 23亮
                GPIO.output(GPIO_LED_STANDBY, GPIO.HIGH)
                # 减少日志输出
                # logger.info("GPIO set: Standby LED (GPIO 23) ON")
        except Exception as e:
            logger.error(f"GPIO pin control error: {str(e)}")

def _delayed_return_to_standby():
    """延时返回待机状态的定时器回调函数"""
    global _gpio_state, _standby_timer

    try:
        # 关闭当前LED，打开待机LED（状态A）
        _set_gpio_pins(standby=True)
        logger.info("GPIO state change: Delay ended, returning to RED LED (GPIO 23)")

        # 更新状态
        _gpio_state = "standby"
        _standby_timer = None
    except Exception as e:
        logger.error(f"GPIO delayed return to standby error: {str(e)}")

def set_gpio_state(recognized=False, unknown=False):
    """设置GPIO状态，根据人脸识别状态控制LED"""
    global _gpio_state, _standby_timer, _last_face_time, GPIO_INITIALIZED

    # 如果GPIO未初始化，先初始化
    if not GPIO_INITIALIZED:
        initialize_gpio()

    try:
        # 如果检测到人脸（已知或未知）
        if recognized or unknown:
            # 更新最后检测到人脸的时间
            _last_face_time = time.time()

            # 如果有待机定时器在运行，取消它
            if _standby_timer is not None:
                _standby_timer.cancel()
                _standby_timer = None
                # 减少日志输出
                # logger.info("GPIO: Canceled standby timer due to face detection")

            # 如果识别到已知人脸（状态C）
            if recognized:
                # 如果当前状态不是"recognized"
                if _gpio_state != "recognized":
                    # 设置GPIO: GPIO 16亮
                    _set_gpio_pins(recognized=True)
                    logger.info("GPIO state change: Known face detected - switching to Green LED (GPIO 16)")

                    # 更新状态
                    _gpio_state = "recognized"

            # 如果检测到未知人脸（状态B）
            elif unknown:
                # 如果当前状态不是"unknown"
                if _gpio_state != "unknown":
                    # 设置GPIO: GPIO 24亮
                    _set_gpio_pins(unknown=True)
                    logger.info("GPIO state change: Unknown face detected - switching to Yellow LED (GPIO 24)")

                    # 更新状态
                    _gpio_state = "unknown"

        # 如果没有检测到人脸（状态A）
        else:
            # 如果当前状态不是standby且没有定时器在运行
            if _gpio_state != "standby" and _standby_timer is None:
                # 启动定时器，延迟返回待机状态
                _standby_timer = Timer(_delay_time, _delayed_return_to_standby)
                _standby_timer.daemon = True
                _standby_timer.start()
                logger.info(f"GPIO state change: No face detected - returning to standby state in {_delay_time} seconds")

            # 如果已经是待机状态但没有LED亮起
            elif _gpio_state == "standby" and not GPIO.input(GPIO_LED_STANDBY):
                # 设置GPIO: GPIO 23亮（状态A）
                _set_gpio_pins(standby=True)
                logger.info("GPIO state change: Ensuring standby LED is on (GPIO 23)")
    except Exception as e:
        logger.error(f"GPIO control error: {str(e)}")

def cleanup_gpio():
    """清理GPIO资源

    返回:
        bool: 如果GPIO已初始化并成功清理则返回True，否则返回False
    """
    try:
        logger.info("Cleaning up GPIO resources...")
        global GPIO_INITIALIZED

        # 取消所有定时器
        global _standby_timer
        if _standby_timer is not None:
            _standby_timer.cancel()
            _standby_timer = None

        # 只有在GPIO已初始化的情况下才进行清理
        if GPIO_INITIALIZED:
            # 确保设置了GPIO模式
            if GPIO.getmode() is None:
                GPIO.setmode(GPIO.BCM)

            # 关闭所有 LED
            with _gpio_lock:
                GPIO.output(GPIO_LED_RECOGNIZED, GPIO.LOW)
                GPIO.output(GPIO_LED_UNKNOWN, GPIO.LOW)
                GPIO.output(GPIO_LED_STANDBY, GPIO.LOW)
                time.sleep(0.3)
                # 清理 GPIO
                GPIO.cleanup()

            GPIO_INITIALIZED = False
            logger.info("GPIO resources cleaned up")
            return True
        else:
            logger.info("GPIO was not initialized, nothing to clean up")
            return False
    except Exception as e:
        logger.error(f"Error cleaning up GPIO resources: {str(e)}")
        return False

# 为了兼容现有代码，提供一些别名函数
def set_recognized_face():
    """设置为已知人脸状态"""
    set_gpio_state(recognized=True, unknown=False)

def set_unknown_face():
    """设置为未知人脸状态"""
    set_gpio_state(recognized=False, unknown=True)

def set_standby_state():
    """设置为待机状态"""
    set_gpio_state(recognized=False, unknown=False)

# 初始化GPIO（自动初始化）
initialize_gpio()