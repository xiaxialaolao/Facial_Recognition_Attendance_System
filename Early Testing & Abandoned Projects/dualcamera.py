import subprocess
import threading
import time
import signal
import os
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DualCamera")

def run_camera_process(script_name):
    """运行摄像头进程"""
    try:
        logger.info(f"Starting {script_name}")
        # 添加环境变量来优化性能
        env = os.environ.copy()
        env['OPENCV_OPENCL_RUNTIME'] = '1'  # 启用OpenCL加速
        env['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'  # 优化视频IO
        
        # 启动进程并设置较低的优先级
        process = subprocess.Popen(
            [sys.executable, script_name],
            env=env,
            bufsize=-1  # 使用系统默认缓冲区大小
        )
        
        return process
    except Exception as e:
        logger.error(f"Error starting {script_name}: {str(e)}")
        return None

def signal_handler(signum, frame):
    """信号处理函数"""
    logger.info("Received signal to terminate")
    global running
    running = False

def main():
    global running
    running = True
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 启动两个摄像头程序
        process1 = run_camera_process("entrance.py")
        time.sleep(2)  # 等待第一个程序初始化
        process2 = run_camera_process("exit.py")

        if not process1 or not process2:
            logger.error("Failed to start one or both camera processes")
            return

        # 等待程序运行
        while running:
            time.sleep(0.1)
            
            # 检查进程是否还在运行
            if process1.poll() is not None:
                logger.error("test3.py has terminated unexpectedly")
                break
            if process2.poll() is not None:
                logger.error("test3cp.py has terminated unexpectedly")
                break

    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
    finally:
        # 清理进程
        logger.info("Cleaning up processes...")
        if process1:
            process1.terminate()
            try:
                process1.wait(timeout=5)
            except:
                process1.kill()
                
        if process2:
            process2.terminate()
            try:
                process2.wait(timeout=5)
            except:
                process2.kill()

if __name__ == "__main__":
    logger.info("Starting dual camera system")
    main()
    logger.info("Dual camera system terminated")