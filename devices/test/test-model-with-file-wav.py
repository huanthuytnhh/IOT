import os
import time
import wave
import tempfile
import numpy as np
import torch
import pyaudio
from pydub import AudioSegment
from gpiozero import Button
import speaker_recognition.inference as inference
import speaker_recognition.neural_net as neural_net

# --- Định nghĩa các hằng số ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Tần số lấy mẫu mặc định của micro, sẽ resample về 16000Hz sau
CHUNK = 512
RECORD_SECONDS = 3  # Thời lượng ghi âm
RAW_RECORDING_FILENAME = "test-new-model-mic/recorded_audio_raw.wav"
RESAMPLED_PROCESSING_FILENAME = "test-new-model-mic/recorded_audio_resampled_for_processing.wav"
N_TIMES_DUPLICATE = 1  # Nhân bản âm thanh để đủ thời lượng nếu cần

# --- Cấu hình đầu vào ---
USE_MICROPHONE_INPUT = False
# STATIC_WAV_FILE_PATH = "/home/pi/Desktop/PBL5/ai_model/data/user_C/user_C_50.wav"
# STATIC_WAV_FILE_PATH = "/home/pi/Desktop/PBL5/ai_model/data/user_E/user_E_17.wav"
STATIC_WAV_FILE_PATH = "/home/pi/Desktop/09_06/IOT/temp_recorded_audio/recording_resampled.wav"
# STATIC_WAV_FILE_PATH = "/home/pi/Desktop/PBL5/ai_model/data/user_D/user_D_50.wav"
# STATIC_WAV_FILE_PATH = "/home/pi/Desktop/PBL5/ai_model/data/user_A/user_A_20.wav"
# STATIC_WAV_FILE_PATH = "/home/pi/Desktop/PBL05_smart_home_with_voice_print/AI Module/speaker_recognition_using_lstm/Data Tiếng nói để điều khiển nhà/Đạt/Dat_tat_den_garage.wav"
# STATIC_WAV_FILE_PATH = "/home/pi/Desktop/PBL05_smart_home_with_voice_print/AI Module/speaker_recognition_using_lstm/Data Tiếng nói để điều khiển nhà/Phát/phat_bat-den-phong-bep.wav"
# STATIC_WAV_FILE_PATH = "/home/pi/Desktop/PBL05_smart_home_with_voice_print/AI Module/speaker_recognition_using_lstm/Data Tiếng nói để điều khiển nhà/Trí/tri_tat_den_phong_ngu_con_cai.wav"
# Tạo thư mục nếu chưa có
os.makedirs("test-new-model-mic", exist_ok=True)

# Cảm biến chạm GPIO
touch_sensor = Button(22)

# --- Hàm hỗ trợ ---
def get_unique_filename(base_filename):
    """Tạo tên file duy nhất bằng cách thêm timestamp."""
    base, ext = os.path.splitext(base_filename)
    return f"{base}_{int(time.time())}{ext}"

def convert_sample_rate(input_filename, output_filename, target_sample_rate):
    """Chuyển đổi tần số lấy mẫu của file WAV."""
    # print(f"Đang chuyển đổi sample rate của '{input_filename}' sang {target_sample_rate} Hz...")
    try:
        sound = AudioSegment.from_file(input_filename)
        sound = sound.set_frame_rate(target_sample_rate)
        sound.export(output_filename, format="wav")
        # print(f"Đã lưu file chuyển đổi: '{output_filename}'")
    except Exception as e:
        print(f"Lỗi khi chuyển đổi sample rate: {e}")
        raise

def extend_audio(audio_segment, times=N_TIMES_DUPLICATE):
    """Nhân bản đoạn âm thanh để kéo dài thời lượng."""
    if times > 1:
        print(f"Nhân bản âm thanh {times} lần để kéo dài thời lượng...")
    return audio_segment * times

def get_embedding_from_audiosegment(audio_segment, encoder_model):
    """Lấy embedding từ một đối tượng AudioSegment."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        audio_segment.export(tmpfile.name, format="wav")
        tmp_filename = tmpfile.name
    try:
        embedding = inference.get_embedding(tmp_filename, encoder_model)
    except Exception as e:
        print(f"Lỗi khi tính embedding cho '{tmp_filename}': {e}")
        raise
    finally:
        os.remove(tmp_filename)  # Luôn xóa file tạm
    return embedding

def record_audio(filename=RAW_RECORDING_FILENAME, record_seconds=RECORD_SECONDS):
    """Ghi âm từ micro và lưu vào file WAV."""
    print(f"🎤 Đang ghi âm trong {record_seconds} giây...")
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    frames = []
    for _ in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    print("✅ Kết thúc ghi âm.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    full_filename = get_unique_filename(filename)
    wf = wave.open(full_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)  # Sửa lỗi typo từ 'setframezrate' thành 'setframerate'
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Đã lưu bản ghi âm thô tại: '{full_filename}'")
    return full_filename

def process_and_compare_audio(input_audio_path, encoder_model, base_embeddings):
    """Xử lý file âm thanh và so sánh với các embedding cơ sở."""
    if not os.path.exists(input_audio_path):
        print(f"❌ Lỗi: Không tìm thấy file âm thanh đầu vào: '{input_audio_path}'")
        return

    resampled_filename = get_unique_filename(RESAMPLED_PROCESSING_FILENAME)
    convert_sample_rate(input_audio_path, resampled_filename, 16000)
    
    audio_test = AudioSegment.from_wav(resampled_filename)
    extended_audio = extend_audio(audio_test)
    
    try:
        audio_embedding = get_embedding_from_audiosegment(extended_audio, encoder_model)
    except Exception as e:
        print(f"Không thể lấy embedding từ file ghi âm: {e}")
        return

    users = ["Trí","Sum", "Đạt", "Quân", "Quang", "Phát", "Thanh"]
    distances = [
        inference.compute_cosine_similarity(base_embeddings["Tri"], audio_embedding),
        inference.compute_cosine_similarity(base_embeddings["Sum"], audio_embedding),
        inference.compute_cosine_similarity(base_embeddings["Dat"], audio_embedding),
        inference.compute_cosine_similarity(base_embeddings["Quan"], audio_embedding),
        inference.compute_cosine_similarity(base_embeddings["Quang"], audio_embedding),
        inference.compute_cosine_similarity(base_embeddings["Phat"], audio_embedding),
        inference.compute_cosine_similarity(base_embeddings["Thanh"], audio_embedding)
    ]

    print("\n--- Khoảng cách Cosine ---")
    for user, distance in zip(users, distances):
        print(f"  - {user}: {distance:.4f}")
    
    min_distance = min(distances)
    min_distance_index = distances.index(min_distance)
    recognized_user = users[min_distance_index]
    print(f"\n✨ Người nói được nhận dạng là: {recognized_user} (khoảng cách: {min_distance:.4f})")

    os.remove(resampled_filename)
    if USE_MICROPHONE_INPUT:
        os.remove(input_audio_path)
    print("✅ Đã xử lý xong và dọn dẹp file tạm.")

def prepare_base_embedding(file_path, encoder_model):
    """Tính toán embedding cơ sở cho một người dùng."""
    if not os.path.exists(file_path):
        print(f"❌ Lỗi: Không tìm thấy file mẫu '{file_path}'.")
        return None

    try:
        audio = AudioSegment.from_wav(file_path)
        print(f"Sample rate của file mẫu '{os.path.basename(file_path)}': {audio.frame_rate} Hz")
    except Exception as e:
        print(f"Lỗi khi đọc file mẫu '{file_path}': {e}")
        return None

    resampled_path = get_unique_filename("test-new-model-mic/temp_resampled_base.wav")
    convert_sample_rate(file_path, resampled_path, 16000)
    
    audio = AudioSegment.from_wav(resampled_path)
    extended = extend_audio(audio)
    
    embedding = None
    try:
        embedding = get_embedding_from_audiosegment(extended, encoder_model)
    except Exception as e:
        print(f"Lỗi khi tính embedding cho file mẫu '{resampled_path}': {e}")
    finally:
        os.remove(resampled_path)
    return embedding

# --- Chương trình chính ---
print("🔄 Đang load model Speaker Encoder...")
encoder_path = "/home/pi/Desktop/09_06/IOT/saved_model/best-model-train-clean-360-hours-50000-epochs-specaug-80-mfcc-100-seq-len-full-inference-8-batch-3-stacks/saved_model_20240606215142.pt"
encoder = neural_net.get_speaker_encoder(encoder_path)
print("✅ Đã load model xong!")

print("\n🔄 Đang load mẫu giọng nói cơ sở của người dùng...")
base_embeddings = {}
base_embeddings["Tri"] = prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Tri-Merge_Audio.wav", encoder)
base_embeddings["Sum"] = prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Sum-Merge_Audio.wav", encoder)
base_embeddings["Dat"] = prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Dat-Merge_Audio.wav", encoder)
base_embeddings["Quan"] = prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Quan-Merge_Audio.wav", encoder)
base_embeddings["Quang"] = prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Quang-Merge_Audio.wav", encoder)
base_embeddings["Phat"] = prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Phat-Merge_Audio.wav", encoder)
base_embeddings["Thanh"] = prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Thanh-Merge_Audio.wav", encoder)

if any(e is None for e in base_embeddings.values()):
    print("❌ Lỗi: Không thể load đầy đủ các mẫu giọng nói cơ sở. Vui lòng kiểm tra lại đường dẫn file.")
    exit()
print("✅ Đã load xong các mẫu giọng nói cơ sở!")

try:
    if USE_MICROPHONE_INPUT:
        print("\n💡 Chạm vào cảm biến (GPIO 22) để bắt đầu GHI ÂM và nhận dạng giọng nói...")
    else:
        print(f"\n💡 Chạm vào cảm biến (GPIO 22) để xử lý file: '{STATIC_WAV_FILE_PATH}' và nhận dạng giọng nói...")
    
    while True:
        touch_sensor.wait_for_press()
        print("\n--- Cảm biến được chạm! ---")
        
        input_audio_for_processing = record_audio() if USE_MICROPHONE_INPUT else STATIC_WAV_FILE_PATH
        if not USE_MICROPHONE_INPUT:
            print(f"Đang sử dụng file đã định nghĩa: '{input_audio_for_processing}'")
        
        process_and_compare_audio(input_audio_for_processing, encoder, base_embeddings)
        time.sleep(1)

except KeyboardInterrupt:
    print("\n⛔ Dừng chương trình.")
except Exception as e:
    print(f"Đã xảy ra lỗi không mong muốn: {e}")
finally:
    touch_sensor.close()
    print("Đã đóng tài nguyên GPIO Zero.")