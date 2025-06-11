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
    servo_pin = 7
    door_controller = None

    try:
        print("Đang khởi tạo ServoController...")
        door_controller = ServoController(pin=servo_pin)
        print("ServoController đã sẵn sàng.")
        print("\nĐiều khiển Servo bằng bàn phím (sử dụng gpiozero):")
        print("  'o [angle]' - Mở cửa đến góc (mặc định 90, ví dụ: o 120)")
        print("  'c'         - Đóng cửa (về 0 độ)")
        print("  'a <angle>' - Đặt cửa ở góc tùy chỉnh (ví dụ: a 45)")
        print("  'q'         - Thoát chương trình")
        print("Nhấn Ctrl+C để thoát bất cứ lúc nào.\n")

        while True:
            command_input = input("Nhập lệnh: ").strip().lower()

            if command_input == 'q':
                break
            elif command_input == 'c':
                door_controller.close_door()
            elif command_input.startswith('o'):
                parts = command_input.split()
                angle = 90
                if len(parts) > 1:
                    try:
                        angle = int(parts[1])
                    except ValueError:
                        print("Góc không hợp lệ. Sử dụng giá trị số.")
                        continue
                door_controller.open_door(angle)
            elif command_input.startswith('a '):
                try:
                    angle_str = command_input.split(' ')[1]
                    angle = int(angle_str)
                    door_controller.custom_angle(angle)
                except (IndexError, ValueError):
                    print("Lệnh không hợp lệ. Sử dụng 'a <angle>', ví dụ: 'a 45'.")
            else:
                print("Lệnh không xác định. Vui lòng thử lại.")

    except KeyboardInterrupt:
        print("\nĐã nhận Ctrl+C. Đang thoát...")
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn: {e}")
    finally:
        if door_controller:
            door_controller.cleanup()
        print("Hoàn tất chương trình.")