import sounddevice as sd
import soundfile as sf
import os
# Liệt kê toàn bộ thiết bị âm thanh
# devices = sd.query_devices()

# # In danh sách thiết bị
# print("Danh sách thiết bị âm thanh có sẵn:\n")
# for i, device in enumerate(devices):
#     print(f"{i}: {device} \n")


# Cấu hình ghi âm
device_index = 1 # ⚠️ QUAN TRỌNG: Kiểm tra lại device_index trên Raspberry Pi của bạn
                  # Chạy `python -m sounddevice` trong terminal để xem danh sách thiết bị
duration = 4   # Thời gian ghi âm (giây). Ví dụ: 5 giây hoặc 180 giây (3 phút)
fs = 16000        # Tần số lấy mẫu 16kHz, chuẩn cho voice identification

# --- THAY ĐỔI ĐƯỜNG DẪN LƯU FILE TẠI ĐÂY NẾU CẦN ---
# Đường dẫn thư mục cố định để lưu file
# Bạn có thể thay đổi "Thanh" thành tên người dùng khác nếu muốn
# ví dụ: user_folder_name = "Quan"
user_folder_name = "Thanh"
save_dir_base = "/home/pi/Desktop/09_06/IOT/devices/test/audio-test-latest"
save_dir = os.path.join(save_dir_base, user_folder_name)
# ----------------------------------------------------

# Lấy tên user từ tên thư mục cuối cùng (ví dụ: "Thanh")
# Điều này hữu ích để tự động đặt tên file theo tên thư mục người dùng
user_name_for_file = os.path.basename(save_dir)

# Tạo thư mục nếu chưa có
os.makedirs(save_dir, exist_ok=True)
print(f"Sẽ lưu file vào thư mục: {save_dir}")

# Tìm số thứ tự file tiếp theo
# Chỉ đếm các file có dạng user_name_for_file_X.wav (ví dụ: Thanh_1.wav)
existing_files_for_user = [
    f for f in os.listdir(save_dir)
    if f.startswith(f"{user_name_for_file}_") and f.endswith(".wav")
]
file_count = len(existing_files_for_user) + 1
output_filename = os.path.join(save_dir, f"{user_name_for_file}_{file_count}.wav")

# Bắt đầu ghi âm
print(f"Bắt đầu ghi âm cho {user_name_for_file}, lưu vào {output_filename}...")
print(f"Thời gian ghi âm: {duration} giây.")
try:
    recording = sd.rec(
        int(duration * fs), samplerate=fs, channels=1, dtype="int16", device=device_index
    )
    sd.wait()  # Chờ ghi âm hoàn tất
    print("Ghi âm xong!")

    # Lưu file WAV
    sf.write(output_filename, recording, fs)
    print(f"Đã lưu file: {output_filename}")

except Exception as e:
    print(f"Có lỗi xảy ra trong quá trình ghi âm hoặc lưu file: {e}")
    print("Hãy kiểm tra lại 'device_index' và quyền ghi vào thư mục.")
    print("Bạn có thể chạy 'python -m sounddevice' để xem danh sách thiết bị âm thanh.")