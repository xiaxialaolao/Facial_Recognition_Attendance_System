import RPi.GPIO as GPIO
import time

# 设置物理引脚编号模式（BOARD）
GPIO.setmode(GPIO.BCM)

# 定义实际可用的输出引脚（物理18=BCM24，物理15=BCM25，物理14=BCM8）
led_pins = [18, 15, 14, 26, 19, 13]

# 初始化引脚
for pin in led_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)  # 初始关闭

try:
    print("LED交替闪烁中，按 Ctrl+C 停止")
    while True:
        # 点亮第一个LED
        GPIO.output(18, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(18, GPIO.LOW)
        
        # 点亮第二个LED
        GPIO.output(15, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(15, GPIO.LOW)

	    # 点亮第3个LED
        GPIO.output(14, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(14, GPIO.LOW)

        GPIO.output(26, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(26, GPIO.LOW)

        GPIO.output(19, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(19, GPIO.LOW)

        GPIO.output(13, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(13, GPIO.LOW)

except KeyboardInterrupt:
    print("\n程序终止")
finally:
    GPIO.cleanup()
