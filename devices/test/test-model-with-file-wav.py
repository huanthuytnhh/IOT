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

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 256
RECORD_SECONDS = 3
RAW_RECORDING_FILENAME = "test-new-model-mic/recorded_audio_raw.wav"
RESAMPLED_PROCESSING_FILENAME = "test-new-model-mic/recorded_audio_resampled_for_processing.wav"
N_TIMES_DUPLICATE = 3

USE_MICROPHONE_INPUT = True
STATIC_WAV_FILE_PATH = "/home/pi/Desktop/09_06/IOT/temp_recorded_audio/recording_resampled.wav"

os.makedirs("test-new-model-mic", exist_ok=True)
touch_sensor = Button(22)

def get_unique_filename(base_filename):
    base, ext = os.path.splitext(base_filename)
    return f"{base}_{int(time.time())}{ext}"

def convert_sample_rate(input_filename, output_filename, target_sample_rate):
    sound = AudioSegment.from_file(input_filename)
    sound = sound.set_frame_rate(target_sample_rate)
    sound.export(output_filename, format="wav")

def extend_audio(audio_segment, times=N_TIMES_DUPLICATE):
    if times > 1:
        print(f"Nhân bản âm thanh {times} lần để kéo dài thời lượng...")
    return audio_segment * times

def get_embedding_from_audiosegment(audio_segment, encoder_model):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        audio_segment.export(tmpfile.name, format="wav")
        tmp_filename = tmpfile.name
    try:
        embedding = inference.get_embedding(tmp_filename, encoder_model)
    finally:
        os.remove(tmp_filename)
    return embedding

def record_audio(filename=RAW_RECORDING_FILENAME, record_seconds=RECORD_SECONDS):
    print(f"🎤 Mời nói vào micro trong {record_seconds} giây...")
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
    with wave.open(full_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    print(f"Đã lưu bản ghi âm tại: '{full_filename}'")
    return full_filename

def process_and_compare_audio(input_audio_path, encoder_model, base_embeddings):
    if not os.path.exists(input_audio_path):
        print(f"❌ Lỗi: Không tìm thấy file âm thanh đầu vào: '{input_audio_path}'")
        return

    resampled_filename = get_unique_filename(RESAMPLED_PROCESSING_FILENAME)
    
    convert_sample_rate(input_audio_path, resampled_filename, 16000)
    audio_test = AudioSegment.from_wav(resampled_filename)
    audio_test = audio_test.normalize()
    
    min_duration_ms = 3000
    if len(audio_test) < min_duration_ms:
        times_to_extend = int(np.ceil(min_duration_ms / len(audio_test)))
        extended_audio = extend_audio(audio_test, times=times_to_extend)
    else:
        extended_audio = audio_test

    audio_embedding = get_embedding_from_audiosegment(extended_audio, encoder_model)

    distances = {}
    for user, base_emb in base_embeddings.items():
        distance = inference.compute_cosine_similarity(base_emb, audio_embedding)
        distances[user] = distance
        print(f"Khoảng cách Cosine với {user}: {distance:.4f}")

    print("\n--- Khoảng cách Cosine ---")
    for user, distance in sorted(distances.items(), key=lambda item: item[1]):
        print(f"  - {user}: {distance:.4f}")
    
    recognized_user = min(distances, key=distances.get)
    min_distance = distances[recognized_user]
    
    SIMILARITY_THRESHOLD = - 0.5 
    
    if min_distance < SIMILARITY_THRESHOLD:
        print(f"\n✨ Người nói được nhận dạng là: \033[94m{recognized_user}\033[0m (khoảng cách: {min_distance:.4f})")
    else:
        print(f"\n🚫 Không nhận dạng được. Người gần nhất là {recognized_user} (khoảng cách: {min_distance:.4f})")

    if os.path.exists(resampled_filename):
        os.remove(resampled_filename)
    if USE_MICROPHONE_INPUT and os.path.exists(input_audio_path):
        os.remove(input_audio_path)
    print("✅ Xử lý xong.")

def prepare_base_embedding(file_path, encoder_model):
    if not os.path.exists(file_path):
        print(f"❌ Lỗi: Không tìm thấy file mẫu '{file_path}'.")
        return None

    resampled_path = get_unique_filename("test-new-model-mic/temp_resampled_base.wav")
    convert_sample_rate(file_path, resampled_path, 16000)
    
    audio = AudioSegment.from_wav(resampled_path)
    extended = extend_audio(audio.normalize())
    
    try:
        embedding = get_embedding_from_audiosegment(extended, encoder_model)
    finally:
        os.remove(resampled_path)
    return embedding

print("🔄 Đang load model Speaker Encoder...")
encoder_path = "/home/pi/Desktop/09_06/IOT/saved_model/best-model-train-clean-360-hours-50000-epochs-specaug-80-mfcc-100-seq-len-full-inference-8-batch-3-stacks/saved_model_20240606215142.pt"
encoder = neural_net.get_speaker_encoder(encoder_path)
print("✅ Đã load model xong!")

print("\n🔄 Đang load mẫu giọng nói cơ sở...")
base_embeddings = {
    "Tri": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Tri-Merge_Audio.wav", encoder),
    "Sum": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Sum-Merge_Audio.wav", encoder),
    "Dat": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Dat-Merge_Audio.wav", encoder),
    "Quan": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Quan-Merge_Audio.wav", encoder),
    "Quang": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Quang-Merge_Audio.wav", encoder),
    "Phat": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Phat-Merge_Audio.wav", encoder),
    "Thanh": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Thanh-Merge_Audio.wav", encoder)
}

if any(e is None for e in base_embeddings.values()):
    print("❌ Lỗi: Không thể load đầy đủ các mẫu giọng nói. Vui lòng kiểm tra lại đường dẫn file.")
    exit()
print("✅ Đã load xong các mẫu giọng nói cơ sở!")

try:
    print("\n💡 Chương trình đã sẵn sàng. Mời bạn bấm nút cảm biến (GPIO 22) để bắt đầu.")
    while True:
        touch_sensor.wait_for_press()
        print("\n--- Cảm biến được chạm! ---")
        
        if USE_MICROPHONE_INPUT:
            input_audio_for_processing = record_audio()
        else:
            input_audio_for_processing = STATIC_WAV_FILE_PATH
            print(f"Đang sử dụng file tĩnh: '{input_audio_for_processing}'")
        
        if input_audio_for_processing:
            process_and_compare_audio(input_audio_for_processing, encoder, base_embeddings)
        
        time.sleep(1)

except KeyboardInterrupt:
    print("\n⛔ Dừng chương trình.")
finally:
    touch_sensor.close()
    print("Đã đóng tài nguyên GPIO.")