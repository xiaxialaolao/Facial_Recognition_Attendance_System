import RPi.GPIO as GPIO
import time

# 设置物理引脚编号模式（BOARD）
GPIO.setmode(GPIO.BOARD)

# 定义实际可用的输出引脚（物理16=BCM23，物理18=BCM24）
led_pins = [16, 18, 36]

# 初始化引脚
for pin in led_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)  # 初始关闭

try:
    print("LED交替闪烁中，按 Ctrl+C 停止")
    while True:
        # 点亮第一个LED
        GPIO.output(16, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(16, GPIO.LOW)
        
        # 点亮第二个LED
        GPIO.output(18, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(18, GPIO.LOW)

	# 点亮第3个LED
        GPIO.output(36, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(36, GPIO.LOW)



except KeyboardInterrupt:
    print("\n程序终止")
finally:
    GPIO.cleanup()
