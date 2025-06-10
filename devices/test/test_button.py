from gpiozero import Button
from signal import pause # Dùng để giữ chương trình chạy

# Khởi tạo một đối tượng Button trên chân GPIO 17 (BCM).
# Nút bấm được kết nối giữa GPIO 17 và GND.
# pull_up=True nghĩa là khi không nhấn, chân sẽ ở mức HIGH.
# Khi nhấn, nút bấm sẽ kéo chân về mức LOW.
button = Button(22, pull_up=True) # Hoặc Button(pin_number)

# Hàm sẽ được gọi khi nút bấm được nhấn
def nut_da_nhan():
    print("Nút bấm đã được nhấn!")

# Hàm sẽ được gọi khi nút bấm được nhả
def nut_da_nha():
    print("Nút bấm đã được nhả.")

# Gán hàm xử lý sự kiện
button.when_pressed = nut_da_nhan
button.when_released = nut_da_nha

print("Đang chờ nút bấm trên GPIO 17. Nhấn Ctrl+C để thoát.")
pause() # Giữ chương trình chạy vô thời hạn để chờ sự kiện