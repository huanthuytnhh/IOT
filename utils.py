import subprocess
import re
from gtts import gTTS
import pygame
import os
from pydub import AudioSegment
import tempfile # Thêm import này

def convert_sample_rate(input_path, output_path, target_sample_rate=16000):
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'panic', '-y', '-i', input_path, '-ar', str(target_sample_rate), output_path]
    subprocess.run(command, check=True)

def speech2text(wav_file, model):
    return model(wav_file)['text']

def extend_audio(audio, times=5):
    return audio * times

def extract_action_and_device(sentence):
    # Định nghĩa các hành động
    actions = r"(mở|đóng|bật|tắt|xem)"

    # Định nghĩa các thiết bị với các biến thể có thể được nhận dạng
    # Sắp xếp từ cụm từ dài nhất đến ngắn nhất để ưu tiên khớp chính xác hơn
    devices = r"(cửa nhà xe|đèn nhà xe|cửa phòng khách|đèn phòng khách|cửa phòng ngủ ba mẹ|cửa phòng ngủ con cái|đèn phòng ngủ ba mẹ|đèn phòng ngủ con cái|đèn phòng bếp|cảm biến|phòng bố mẹ|phòng ba mẹ|phòng khách|nhà xe|phòng bếp|phòng ngủ con cái)"

    # Kết hợp thành mẫu regex chính
    pattern = rf"{actions}\s*(?:cửa\s*)?(?:phòng\s*ngủ\s*)?({devices})\b" # Điều chỉnh pattern để linh hoạt hơn

    match = re.search(pattern, sentence)

    if match:
        action = match.group(1).strip()
        device_raw = match.group(2).strip()

        # Ánh xạ các biến thể được nhận dạng về tên thiết bị chuẩn
        if "nhà xe" in device_raw:
            if action in ["mở", "đóng"]:
                device = "cửa nhà xe"
            elif action in ["bật", "tắt"]:
                device = "đèn nhà xe"
            else: # Nếu không phải hành động cụ thể, mặc định là cửa
                device = "cửa nhà xe" 
        elif "phòng khách" in device_raw:
            if action in ["mở", "đóng"]:
                device = "cửa phòng khách"
            elif action in ["bật", "tắt"]:
                device = "đèn phòng khách"
            else:
                device = "đèn phòng khách" # Mặc định là đèn
        elif "phòng bố mẹ" in device_raw or "phòng ba mẹ" in device_raw:
            device = "cửa phòng ngủ ba mẹ" # Ánh xạ về tên chuẩn
            if action in ["bật", "tắt"]:
                device = "đèn phòng ngủ ba mẹ"
        elif "phòng ngủ con cái" in device_raw:
            if action in ["mở", "đóng"]:
                device = "cửa phòng ngủ con cái"
            elif action in ["bật", "tắt"]:
                device = "đèn phòng ngủ con cái"
            else:
                device = "đèn phòng ngủ con cái" # Mặc định là đèn
        elif "phòng bếp" in device_raw:
            device = "đèn phòng bếp" # Chỉ có đèn phòng bếp
        else:
            device = device_raw # Giữ nguyên nếu đã khớp chính xác

        print(f"Hành động được nhận dạng: {action}, Thiết bị được nhận dạng: {device}")
        return action, device
    else:
        print("Không tìm thấy khớp.")
        print(f"Mẫu regex đã dùng: {pattern}")
        print(f"Câu lệnh đầu vào: {sentence}")
        return None, None

def speak_text(text, lang='vi', volume=1.0):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save("temp.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load("temp.mp3")
    pygame.mixer.music.set_volume(volume)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    os.remove("temp.mp3")

def count_files_in_folder(folder_path, extension=None):
    if extension:
        return len([file for file in os.listdir(folder_path) if file.endswith(extension)])
    else:
        return len(os.listdir(folder_path))

def split_audio(input_path, output_dir, first_segment_duration, segment_duration, num_segments):
    os.makedirs(output_dir, exist_ok=True)
    audio = AudioSegment.from_wav(input_path)
    first_segment = audio[:first_segment_duration]
    first_segment.export(os.path.join(output_dir, "Dat_base_100s.wav"), format="wav")
    for i in range(1, num_segments + 1):
        start_time = first_segment_duration + (i - 1) * segment_duration
        end_time = start_time + segment_duration
        segment = audio[start_time:end_time]
        segment.export(os.path.join(output_dir, f"Dat_segment_3s_{i}.wav"), format="wav")
    print("Các phân đoạn âm thanh đã được tạo thành công.")

def merge_audio_files(folder_path, output_file):
    combined_audio = AudioSegment.empty()
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            audio = AudioSegment.from_wav(file_path)
            combined_audio += audio
    combined_audio.export(output_file, format="wav")
    print(f"Tất cả các file âm thanh đã được hợp nhất vào {output_file}")

# Ví dụ sử dụng:
# Câu lệnh từ bạn: Văn bản được nhận dạng: xin chào tôi là lê ngọc thanh xin chào tôi là lê ngọc thanh mở cửa mở cửa phòng bố mẹ
recognized_text = "xin chào tôi là lê ngọc thanh xin chào tôi là lê ngọc thanh mở cửa mở cửa phòng bố mẹ"
action, device = extract_action_and_device(recognized_text)
print(f"Kết quả cuối cùng: Hành động: {action}, Thiết bị: {device}")