import lgpio
import time

# Mở GPIO chip (mặc định là 0)
h = lgpio.gpiochip_open(0)

# Định nghĩa các chân LED
LED1 = 4   # GPIO4
LED2 = 5   # GPIO5
LED3 = 17  # GPIO17

# Thiết lập các chân làm OUTPUT
lgpio.gpio_claim_output(h, LED1)
lgpio.gpio_claim_output(h, LED2)
lgpio.gpio_claim_output(h, LED3)

def gpio_on(pin):
    lgpio.gpio_write(h, pin, 1)
    print(f"GPIO {pin} is ON")

def gpio_off(pin):
    lgpio.gpio_write(h, pin, 0)
    print(f"GPIO {pin} is OFF")

try:
    while True:
        gpio_on(LED1)
        gpio_off(LED2)
        gpio_off(LED3)
        time.sleep(2)

        gpio_off(LED1)
        gpio_on(LED2)
        gpio_off(LED3)
        time.sleep(2)

        gpio_off(LED1)
        gpio_off(LED2)
        gpio_on(LED3)
        time.sleep(2)

except KeyboardInterrupt:
    print("Program terminated")

finally:
    lgpio.gpiochip_close(h)
