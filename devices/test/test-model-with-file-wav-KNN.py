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
from collections import Counter

# --- Định nghĩa các hằng số ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Match model input
CHUNK = 256
RECORD_SECONDS = 3  # 3 seconds for robust embeddings
RAW_RECORDING_FILENAME = "test-new-model-mic/recorded_audio.wav"
RESAMPLED_PROCESSING_FILENAME = "test-new-model-mic/processed_audio.wav"
N_TIMES_DUPLICATE = 1  # Dynamic extension in function
K_NEAREST_NEIGHBOURS = 5  # KNN parameter from second script

# --- Cấu hình đầu vào ---
USE_MICROPHONE_INPUT = True
STATIC_WAV_FILE_PATH = "09_06/IOT/devices/test/test-new-model-mic/temp_audio_recording_resampled_raw_1749555457.wav"
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
    try:
        sound = AudioSegment.from_file(input_filename)
        sound = sound.set_frame_rate(target_sample_rate)
        sound.export(output_filename, format="wav")
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
        os.remove(tmp_filename)
    return embedding

def record_audio(filename=RAW_RECORDING_FILENAME, record_seconds=RECORD_SECONDS):
    """Ghi âm từ micro và lưu vào file WAV."""
    print(f"🎤 Đang ghi âm trong {record_seconds} giây...")
    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"❌ Lỗi mở stream âm thanh: {e}")
        audio.terminate()
        return None

    frames = []
    try:
        for _ in range(0, int(RATE / CHUNK * record_seconds)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
    except Exception as e:
        print(f"❌ Lỗi khi ghi âm: {e}")
        return None
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    full_filename = get_unique_filename(filename)
    try:
        wf = wave.open(full_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    except Exception as e:
        print(f"❌ Lỗi lưu file âm thanh: {e}")
        return None

    audio_segment = AudioSegment.from_wav(full_filename)
    normalized_audio = audio_segment.normalize()
    normalized_audio.export(full_filename, format="wav")
    print(f"Đã chuẩn hóa âm lượng và lưu bản ghi âm tại: '{full_filename}'")
    
    if len(audio_segment) < 3000:  # Less than 3 seconds
        print("⚠️ Cảnh báo: Bản ghi âm nên dài ít nhất 3 giây để đảm bảo độ chính xác.")
    return full_filename

def prepare_base_embedding(folder_path, encoder_model):
    """Tính toán nhiều embedding cơ sở từ thư mục chứa các file WAV của một người dùng."""
    if not os.path.exists(folder_path):
        print(f"❌ Lỗi: Không tìm thấy thư mục '{folder_path}'.")
        return None

    audio_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    if not audio_files:
        print(f"❌ Lỗi: Không tìm thấy file WAV trong '{folder_path}'.")
        return None

    embeddings = []
    for audio_file in audio_files[:10]:  # Limit to 10 files per speaker for efficiency
        file_path = os.path.join(folder_path, audio_file)
        try:
            audio = AudioSegment.from_wav(file_path)
            if len(audio) < 3000:
                print(f"⚠️ File '{file_path}' ngắn hơn 3 giây ({len(audio)} ms). Nên dùng file 3–5 giây.")
            audio = audio.normalize()
            resampled_path = get_unique_filename("test-new-model-mic/temp_resampled_base.wav")
            convert_sample_rate(file_path, resampled_path, 16000)
            audio = AudioSegment.from_wav(resampled_path)
            audio = audio.normalize()
            extended = extend_audio(audio)
            embedding = get_embedding_from_audiosegment(extended, encoder_model)
            embeddings.append((embedding, os.path.basename(folder_path)))  # Store embedding and speaker label
            os.remove(resampled_path)
        except Exception as e:
            print(f"Lỗi xử lý file '{file_path}': {e}")
            continue
    if not embeddings:
        print(f"❌ Không tạo được embedding nào cho '{folder_path}'.")
        return None
    return embeddings

def process_and_compare_audio(input_audio_path, encoder_model, base_embeddings):
    """Xử lý file âm thanh và dự đoán người nói bằng KNN."""
    if not os.path.exists(input_audio_path):
        print(f"❌ Lỗi: Không tìm thấy file âm thanh đầu vào: '{input_audio_path}'")
        return

    try:
        audio_test = AudioSegment.from_wav(input_audio_path)
        audio_test = audio_test.normalize()
        print(f"Đã chuẩn hóa âm lượng cho file đầu vào '{input_audio_path}'")
    except Exception as e:
        print(f"❌ Lỗi đọc file âm thanh: {e}")
        return

    if audio_test.frame_rate != 16000:
        resampled_filename = get_unique_filename(RESAMPLED_PROCESSING_FILENAME)
        try:
            audio_test.export(resampled_filename, format="wav")
            audio_test = AudioSegment.from_wav(resampled_filename)
            audio_test = audio_test.normalize()
        except Exception as e:
            print(f"❌ Lỗi resampling: {e}")
            return
    else:
        resampled_filename = None

    min_duration_ms = 3000  # 3 seconds recommended
    if len(audio_test) < min_duration_ms:
        times = int(np.ceil(min_duration_ms / len(audio_test)))
        print(f"Nhân bản âm thanh {times} lần để đạt độ dài tối thiểu.")
        extended_audio = extend_audio(audio_test, times)
        extended_audio = extended_audio.normalize()
    else:
        extended_audio = audio_test

    try:
        audio_embedding = get_embedding_from_audiosegment(extended_audio, encoder_model)
    except Exception as e:
        print(f"❌ Không thể lấy embedding: {e}")
        return

    # KNN prediction
    all_distances = []
    for speaker, embeddings in base_embeddings.items():
        for embedding, _ in embeddings:
            print(f"Đang tính khoảng cách Cosine với {speaker}...")
            print(f"  - Số lượng embedding: {len(embeddings)}")
            print(f"  - Kích thước embedding: {embedding}")
            distance = inference.compute_cosine_similarity(embedding, audio_embedding)
            all_distances.append((distance, speaker))
    
    # Sort by distance and select K nearest neighbors
    sorted_distances = sorted(all_distances, key=lambda x: x[0])
    knn_predictions = [speaker for _, speaker in sorted_distances[:K_NEAREST_NEIGHBOURS]]
    
    # Compute mean distances per speaker (for comparison with original method)
    mean_distances = {}
    for speaker in base_embeddings:
        distances = [inference.compute_cosine_similarity(emb, audio_embedding) for emb, _ in base_embeddings[speaker]]
        mean_distances[speaker] = np.mean(distances) if distances else float('inf')

    print("\n--- Khoảng cách Cosine Trung Bình ---")
    for speaker, distance in mean_distances.items():
        print(f"  - {speaker}: {distance:.4f}")

    # KNN prediction result
    predicted_speaker = Counter(knn_predictions).most_common(1)[0][0]
    print(f"\n--- K-Nearest Neighbors Prediction (K={K_NEAREST_NEIGHBOURS}) ---")
    print(f"Top {K_NEAREST_NEIGHBOURS} nearest neighbors: {knn_predictions}")
    print(f"\033[94mNgười được dự đoán (KNN): {predicted_speaker}\033[0m")

    # Original prediction for comparison
    min_distance = min(mean_distances.values())
    original_predicted_speaker = min(mean_distances, key=mean_distances.get)
    print(f"\nOriginal Prediction (Min Mean Distance): {original_predicted_speaker} (khoảng cách: {min_distance:.4f})")

    if resampled_filename:
        try:
            os.remove(resampled_filename)
        except Exception:
            print(f"Lỗi xóa: {resampled_filename}")
    if USE_MICROPHONE_INPUT:
        try:
            os.remove(input_audio_path)
        except Exception:
            print(f"Lỗi xóa: {input_audio_path}")
    print("✅")

def record_audio_until_ctrl(filename=RAW_RECORDING_FILENAME):
    """Ghi âm từ micro cho đến khi nhấn Ctrl+C."""
    print("🎤 Đang ghi âm... Nhấn Ctrl+C để dừng.")
    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
        frames = []
        while True:
            try:
                if touch_sensor.is_pressed:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
            except KeyboardInterrupt:
                print("\n✅ Dừng ghi âm.")
                break

        full_filename = get_unique_filename(filename)
        with wave.open(full_filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        print(f"Đã lưu bản ghi âm tại: {full_filename}")
        return full_filename
    except Exception as e:
        print(f"❌ Lỗi khi ghi âm: {e}")
        return None
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

# --- Chương trình chính ---
print("🔄 Đang tải Speaker Encoder...")
encoder_path = "/home/pi/Desktop/09_06/IOT/saved_model/best-model-train-clean-360-hours-50000-epochs-specaug-80-mfcc-100-seq-len-full-inference-8-batch-3-stacks/saved_model_20240606215142.pt"
encoder = neural_net.get_speaker_encoder(encoder_path)
print("✅ Đã tải model xong!")

print("\n🔄 Đang tải mẫu giọng nói cơ sở...")
base_embeddings = {
    "Trí": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples_folder/Tri", encoder),
    "Sum": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples_folder/Sum", encoder),
    "Dat": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples_folder/Dat", encoder),
    "Quan": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples_folder/Quan", encoder),
    "Quang": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples_folder/Quang", encoder),
    "Phat": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples_folder/Phat", encoder),
    "Thanh": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples_folder/Thanh", encoder)
}

if any(e is None for e in base_embeddings.values()):
    print("❌ Lỗi: Không thể tải đầy đủ các mẫu giọng nói cơ sở.")
    exit()
print("✅ Đã tải xong các mẫu giọng nói cơ sở!")

try:
    while True:
        print("\n💡 Nhấn Enter để nhập WAV path hoặc chạm cảm biến (GPIO 22) để ghi âm/xử lý file tĩnh.")
        print("Nhập 'q' để thoát.")
        
        touch_sensor.wait_for_press(timeout=0.1)
        if touch_sensor.is_pressed:
            print("\n--- Cảm biến được chạm! ---")
            if USE_MICROPHONE_INPUT:
                input_audio_for_processing = record_audio_until_ctrl()
            else:
                input_audio_for_processing = STATIC_WAV_FILE_PATH
                print(f"Đang sử dụng file: '{input_audio_for_processing}'")
            process_and_compare_audio(input_audio_for_processing, encoder, base_embeddings)
        else:
            user_input = input("Nhập đường dẫn WAV hoặc nhấn Enter để dùng mặc định, 'q' để thoát: ")
            if user_input.lower() == 'q':
                break
            elif user_input.strip():
                input_audio_for_processing = user_input.strip()
                print(f"Đang xử lý file WAV: '{input_audio_for_processing}'")
            else:
                input_audio_for_processing = record_audio() if USE_MICROPHONE_INPUT else STATIC_WAV_FILE_PATH
                if not USE_MICROPHONE_INPUT:
                    print(f"Đang sử dụng file mặc định: '{input_audio_for_processing}'")
            
            process_and_compare_audio(input_audio_for_processing, encoder, base_embeddings)
        time.sleep(1)

except KeyboardInterrupt:
    print("\n⛔ Dừng chương trình.")
except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")
finally:
    touch_sensor.close()
    print("Đã đóng tài nguyên GPIO Zero.")