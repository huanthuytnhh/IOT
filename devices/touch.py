from gpiozero import Button
import time

class TouchSensor:
    def __init__(self, pin):
        self.pin = pin
        # Sử dụng gpiozero Button để thay thế GPIO.IN với pull-up
        self.sensor = Button(self.pin, pull_up=True)

    def is_touched(self):
        # Kiểm tra xem cảm biến có bị chạm hay không
        return self.sensor.is_pressed

# Ví dụ sử dụng class TouchSensor
if __name__ == "__main__":
    touch_sensor_pin = 22
    touch_sensor = TouchSensor(touch_sensor_pin)

    try:
        while True:
            print(f"Sensor state: {touch_sensor.is_touched()}")  # In ra trạng thái của cảm biến
            if touch_sensor.is_touched():
                print("Cảm biến chạm đã được chạm")
            else:
                print("Cảm biến chạm không được chạm")
            # Chờ 0.1 giây trước khi đọc lại
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Kết thúc chương trình.")
