from gpiozero import Servo
from time import sleep

class ServoController:
    def __init__(self, pin=23):
        """Khởi tạo GPIO pin cho servo với gpiozero."""
        self.servo = Servo(pin)  # Kết nối servo với pin GPIO được chỉ định

    def set_servo_angle(self, angle):
        """Điều khiển servo đến góc cụ thể."""
        if 0 <= angle <= 180:
            # Chuyển góc thành giá trị từ -1 đến 1 cho servo
            duty_cycle = (angle - 90) / 90  # Chuyển đổi góc thành giá trị từ -1 đến 1
            self.servo.value = duty_cycle
        else:
            print("Lỗi: Góc phải trong khoảng từ 0 đến 180 độ.")

    def open_door(self, angle):
        """Mở cửa đến góc mở (angle degrees)."""
        self.set_servo_angle(angle)

    def close_door(self):
        """Đóng cửa (quay về vị trí 0 độ)."""
        self.set_servo_angle(0)

    def custom_angle(self, angle):
        """Điều khiển servo đến góc tùy chỉnh."""
        self.set_servo_angle(angle)

    def open_door_close_door(self, angle, time_to_wait):
        """Mở và đóng cửa."""
        self.open_door(angle)  # Mở cửa đến góc đã chỉ định
        sleep(time_to_wait)  # Giữ cửa mở trong khoảng thời gian đã chỉ định
        self.close_door()  # Đóng cửa về vị trí 0 độ

# Ví dụ sử dụng
if __name__ == "__main__":
    door_controller = ServoController(pin=23)  # Kết nối servo với pin GPIO 23
    
    # Mở và đóng cửa
    door_controller.open_door_close_door(90, 3)  # Mở cửa đến 90 độ và đợi 3 giây
    door_controller.open_door_close_door(180, 3)  # Mở cửa đến 90 độ và đợi 3 giây
    door_controller.close_door()  # Đóng cửa về vị trí 0 độ
    door_controller.open_door_close_door(0, 3)  # Đóng cửa về vị trí 0 độ và đợi 3 giây
    # Tùy chọn điều chỉnh góc
    # door_controller.custom_angle(45)  # Điều chỉnh cửa đến 45 độ

    # Cleanup GPIO sau khi sử dụng
    print("Hoàn tất chương trình.")
