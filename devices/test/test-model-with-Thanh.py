import os
import time
import wave
import tempfile
import numpy as np
import torch
import pyaudio
from pydub import AudioSegment
from gpiozero import Button
import speech_recognition as sr
import speaker_recognition.inference as inference
import speaker_recognition.neural_net as neural_net

# Cấu hình ghi âm
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
RECORD_SECONDS = 3
RAW_FILENAME = "test-new-model-mic/temp_audio_recording_raw.wav"
RESAMPLED_FILENAME = "test-new-model-mic/temp_audio_recording_resampled_raw.wav"
N_TIMES_DUPLICATE = 1  # Nhân bản âm thanh để đủ thời lượng nếu cần

# Tạo thư mục nếu chưa có
os.makedirs("test-new-model-mic", exist_ok=True)

# Cảm biến chạm GPIO sử dụng gpiozero (Button mặc định pull-up, pressed == chạm)
touch_sensor = Button(22)

# PyAudio setup
audio = pyaudio.PyAudio()
frames = []

def get_unique_filename(filename):
    base, ext = os.path.splitext(filename)
    return f"{base}_{int(time.time())}{ext}"

def convert_sample_rate(input_filename, output_filename, target_sample_rate):
    sound = AudioSegment.from_file(input_filename)
    sound = sound.set_frame_rate(target_sample_rate)
    sound.export(output_filename, format="wav")

def extend_audio(audio, times=5):
    return audio * times

def get_embedding_from_audiosegment(audio, encoder):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        audio.export(tmpfile.name, format="wav")
        embedding = inference.get_embedding(tmpfile.name, encoder)
    os.remove(tmpfile.name)
    return embedding

def save_and_convert_audio():
    raw_filename = get_unique_filename(RAW_FILENAME)
    resampled_filename = get_unique_filename(RESAMPLED_FILENAME)
    
    with wave.open(raw_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    convert_sample_rate(raw_filename, resampled_filename, 16000)
    print(f"Đã lưu và resample file: {resampled_filename}")
    
    audio_test = AudioSegment.from_wav(resampled_filename)
    extended_audio = extend_audio(audio_test, times=N_TIMES_DUPLICATE)
    audio_embedding = get_embedding_from_audiosegment(extended_audio, encoder)

    # So sánh với từng người
    users = ["Trí", "Phát", "Thanh"]
    distances = [
        inference.compute_cosine_similarity(tri_base_embedding, audio_embedding),
        # inference.compute_cosine_similarity(dat_base_embedding, audio_embedding),
        inference.compute_cosine_similarity(phat_base_embedding, audio_embedding),
        inference.compute_cosine_similarity(thanh_base_embedding, audio_embedding)
    ]

    print("Khoảng cách cosine:", distances)
    print("👉 Người nói là:", users[distances.index(min(distances))])

def record_audio():
    global frames
    frames.clear()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("🎙️ Bắt đầu ghi âm... (giữ cảm biến)")

    start_time = time.time()
    while touch_sensor.is_pressed:
        data = stream.read(CHUNK)
        frames.append(data)
        if time.time() - start_time > RECORD_SECONDS:
            break

    stream.stop_stream()
    stream.close()
    print("✅ Ghi âm xong.")
    save_and_convert_audio()

# Load model
encoder_path = "/home/pi/Desktop/09_06/IOT/saved_model/best-model-train-clean-360-hours-50000-epochs-specaug-80-mfcc-100-seq-len-full-inference-8-batch-3-stacks/saved_model_20240606215142.pt"
encoder = neural_net.get_speaker_encoder(encoder_path)

def prepare_base_embedding(file_path):
    audio = AudioSegment.from_wav(file_path)
    extended = extend_audio(audio, times=N_TIMES_DUPLICATE)
    return get_embedding_from_audiosegment(extended, encoder)

# Load embedding của các người dùng
print("🔄 Đang load mẫu giọng nói...")
tri_base_embedding = prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Tri-Merge_Audio.wav")
# dat_base_embedding = prepare_base_embedding("/home/tranductri2003/Code/PBL05_smart_home_with_voice_print_and_antifraud_ai/Dat-Merge_Audio.wav")
phat_base_embedding = prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Phat-Merge_Audio.wav")
thanh_base_embedding = prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Thanh-Merge_Audio.wav")
print("✅ Đã load xong!")

# Vòng lặp chính
try:
    print("💡 Chạm vào cảm biến để bắt đầu ghi âm...")
    while True:
        touch_sensor.wait_for_press()
        record_audio()
except KeyboardInterrupt:
    print("\n⛔ Dừng chương trình.")
finally:
    audio.terminate()
