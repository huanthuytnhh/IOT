from gpiozero import LED
from time import sleep

class Led:
    def __init__(self, signal_pin, ground_pin=None):
        self.signal_pin = LED(signal_pin)  # Điều khiển LED với gpiozero
        if ground_pin is not None:
            self.ground_pin = ground_pin
            print(f"Ground pin {self.ground_pin} not required with gpiozero.")

    def on(self):
        self.signal_pin.on()
        print(f"GPIO {self.signal_pin.pin} is ON")

    def off(self):
        self.signal_pin.off()
        print(f"GPIO {self.signal_pin.pin} is OFF")

    def blink(self, duration=2):
        self.on()
        sleep(duration)
        self.off()
        sleep(duration)

    def cleanup(self):
        print("Cleanup not necessary in gpiozero for LED.")


# Sử dụng lớp Led
# try:
#     led_living = Led(4)
#     led_kitchen = Led(17)
#     led_children = Led(10)
#     led_parent = Led(11)
#     led_garage = Led(5)
    
#     while True:
#         led_living.blink()  # Bật và tắt LED
#         led_kitchen.blink()  # Bật và tắt LED
#         led_children.blink()  # Bật và tắt LED
#         led_parent.blink()  # Bật và tắt LED
#         led_garage.blink()  # Bật và tắt LED

# except KeyboardInterrupt:
#     print("Program terminated")
# finally:
#     # Cleanup không cần thiết trong gpiozero đối với LED
#     print("Exiting program.")
