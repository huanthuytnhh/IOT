import os
from dotenv import load_dotenv
from collections import defaultdict, Counter
import pyaudio
import wave
from pydub import AudioSegment
import speech_recognition as sr
import numpy as np
import time
# from flask import Flask, request, jsonify # Commented out Flask imports
import threading
# import requests # Commented out requests as it's used for web communication

# Import các lớp tùy chỉnh
from devices.servo import ServoController
from devices.motor import MotorController
from devices.stepper import StepperController
from devices.led import Led
from devices.dht11 import DHTSensor
from devices.touch import TouchSensor
import speaker_recognition.neural_net as neural_net
import speaker_recognition.inference as inference
from db.db_helper import query_members, query_permissions, connect_db
from utils import convert_sample_rate, extract_action_and_device, speak_text, extend_audio

# Load biến môi trường từ .env
load_dotenv()
import tempfile
# Cấu hình biến môi trường
SPEAKER_RECOGNITION_MODEL_PATH = os.getenv("SPEAKER_RECOGNITION_MODEL_PATH")
DB_PATH = os.getenv("DB_PATH")
N_TIMES_DUPLICATE = int(os.getenv("N_TIMES_DUPLICATE"))
N_TAKEN_AUDIO = int(os.getenv("N_TAKEN_AUDIO"))
K_NEAREST_NEIGHBOURS = int(os.getenv("K_NEAREST_NEIGHBOURS"))
FORMAT = eval(os.getenv("FORMAT"))
CHANNELS = int(os.getenv("CHANNELS"))
RATE = int(os.getenv("RATE"))
CHUNK = int(os.getenv("CHUNK"))
RAW_RECORDING_PATH = os.getenv("RAW_RECORDING_PATH")
RESAMPLED_RATE = int(os.getenv("RESAMPLED_RATE"))
WAVE_OUTPUT_RAW_FILENAME = os.getenv("WAVE_OUTPUT_RAW_FILENAME")
WAVE_OUTPUT_RESAMPLED_FILENAME = os.getenv("WAVE_OUTPUT_RESAMPLED_FILENAME")
MAC_ADDRESS = os.getenv("MAC_ADDRESS")

# Khởi tạo mô hình nhận diện giọng nói
SPEAKER_RECOGNITION_MODEL = neural_net.get_speaker_encoder(SPEAKER_RECOGNITION_MODEL_PATH)

# Kết nối database
CONN = connect_db(DB_PATH)

# Khởi tạo các thiết bị tùy chỉnh
motor = MotorController(14,15,18)
stepper = StepperController(21, 20, 16, 12)
servo_parent = ServoController(7)
# servo_children = ServoController(8) # Commented out as it's commented in your original
led_living = Led(4)
led_kitchen = Led(17)
led_children = Led(10)
led_parent = Led(11)
led_garage = Led(5)
dht = DHTSensor(13, 19, 26)
touch_sensor = TouchSensor(22)

# Từ điển trạng thái thiết bị
status_data = {
    "Garage Led": 0,
    "Garage Door": 0,
    "Living Led": 0,
    "Kitchen Led": 0,
    "Parent Led": 0,
    "Children Led": 0,
    "Temperature": 0,
    "Humidity": 0,
}
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

# Redefinition from original code, keeping it as is per request
def convert_sample_rate(input_filename, output_filename, target_sample_rate):
    sound = AudioSegment.from_file(input_filename)
    sound = sound.set_frame_rate(target_sample_rate)
    sound.export(output_filename, format="wav")

# Redefinition from original code, keeping it as is per request
def extend_audio(audio, times=5):
    return audio * times

def get_embedding_from_audiosegment(audio, encoder):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        audio.export(tmpfile.name, format="wav")
        embedding = inference.get_embedding(tmpfile.name, encoder)
    os.remove(tmpfile.name)
    return embedding

# Hàm chuẩn bị embedding cho mẫu giọng nói
def prepare_base_embedding(file_path):
    try:
        audio = AudioSegment.from_wav(file_path)
        extended = extend_audio(audio, times=N_TIMES_DUPLICATE)
        return get_embedding_from_audiosegment(extended, SPEAKER_RECOGNITION_MODEL)
    except Exception as e:
        print(f"Lỗi khi xử lý file âm thanh {file_path}: {e}")
        return None

# Load embedding của các người dùng
print("🔄 Đang load mẫu giọng nói...")
user_embeddings = {}
try:
    user_embeddings = {
        "Trí": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Tri-Merge_Audio.wav"),
        "Phát": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Phat-Merge_Audio.wav"),
        "Thanh": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Thanh-Merge_Audio.wav"),
        "Quang": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Quang-Merge_Audio.wav"),
        "Quân": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Quan-Merge_Audio.wav"),
        "Sum": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Sum-Merge_Audio.wav"),
        "Đạt": prepare_base_embedding("/home/pi/Desktop/09_06/IOT/audio_samples/Dat-Merge_Audio.wav"),
    }
    # Loại bỏ các embedding None (nếu có lỗi)
    user_embeddings = {k: v for k, v in user_embeddings.items() if v is not None}
    if not user_embeddings:
        raise Exception("Không load được embedding nào!")
    print("✅ Đã load xong tất cả các mẫu giọng nói!")
except Exception as e:
    print(f"Lỗi khi load mẫu giọng nói: {e}")
    exit(1)

# Khởi tạo ứng dụng Flask
# app = Flask(__name__) # Commented out Flask app initialization

# Hàm điều khiển thiết bị
def control_device(device, action):
    try:
        if device == "cửa nhà xe":
            if action == "mở":
                stepper.rotate("forward", 5)
                status_data["Garage Door"] = 1
            elif action == "đóng":
                stepper.rotate("backward", 5)
                status_data["Garage Door"] = 0
        # elif device == "cửa phòng ngủ con cái": # Commented out as it's commented in your original
        #     servo_children.open_door_close_door(0, 6)
        elif device == "cửa phòng ngủ ba mẹ":
            servo_parent.open_door_close_door(0, 6)
        elif device == "đèn phòng khách":
            if action == "bật":
                led_living.on()
                status_data["Living Led"] = 1
            elif action == "tắt":
                led_living.off()
                status_data["Living Led"] = 0
        elif device == "đèn phòng bếp":
            if action == "bật":
                led_kitchen.on()
                status_data["Kitchen Led"] = 1
            elif action == "tắt":
                led_kitchen.off()
                status_data["Kitchen Led"] = 0
        elif device == "đèn phòng ngủ ba mẹ":
            if action == "bật":
                led_parent.on()
                status_data["Parent Led"] = 1
            elif action == "tắt":
                led_parent.off()
                status_data["Parent Led"] = 0
        elif device == "đèn phòng ngủ con cái":
            if action == "bật":
                led_children.on()
                status_data["Children Led"] = 1
            elif action == "tắt":
                led_children.off()
                status_data["Children Led"] = 0
        elif device == "đèn nhà xe":
            if action == "bật":
                led_garage.on()
                status_data["Garage Led"] = 1
            elif action == "tắt":
                led_garage.off()
                status_data["Garage Led"] = 0
        elif device == "cảm biến":
            if action == "xem":
                humidity, temperature = dht.read_dht11()
                status_data["Humidity"] = humidity
                status_data["Temperature"] = temperature
                speak_text(f"Nhiệt độ hiện tại là {temperature} độ C và độ ẩm là {humidity} %")
        # Gửi trạng thái cập nhật đến ESP32 - Commented out as part of web communication
        # try:
        #     response = requests.post("http://192.168.1.4:10000/message", json=status_data)
        #     if response.status_code == 200:
        #         print("Đã gửi trạng thái đến ESP32")
        #     else:
        #         print(f"Lỗi khi gửi trạng thái: {response.status_code}")
        # except Exception as e:
        #     print(f"Lỗi khi gửi trạng thái đến ESP32: {e}")
    except Exception as e:
        print(f"Lỗi khi điều khiển thiết bị: {e}")

# Điểm cuối Flask để nhận lệnh từ ESP32 - Commented out Flask endpoint
# @app.route('/command', methods=['POST'])
# def command():
#     try:
#         data = request.get_json()
#         device = data.get('device')
#         action = data.get('action')
#         if device and action:
#             control_device(device, action)
#             return jsonify({"status": "success", "current_state": status_data}), 200
#         else:
#             return jsonify({"status": "error", "message": "Lệnh không hợp lệ"}), 400
#     except Exception as e:
#         print(f"Lỗi trong endpoint Flask: {e}")
#         return jsonify({"status": "error", "message": "Lỗi xử lý yêu cầu"}), 500

# Chạy Flask trong một luồng riêng - Commented out Flask thread
# def run_flask():
#     app.run(host='0.0.0.0', port=5000, threaded=True)

# flask_thread = threading.Thread(target=run_flask)
# flask_thread.daemon = True
# flask_thread.start()

# Hàm ghi âm và nhận diện giọng nói
def record_audio():
    audio = pyaudio.PyAudio()  # Khởi tạo PyAudio trong hàm
    try:
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                             rate=RATE, input=True,
                             frames_per_buffer=CHUNK)

        print("Recording...")

        frames = []
        start_time = time.time()
        # Ghi âm cho đến khi cảm biến nhả ra (nếu không có hành động kéo dài audio)
        # hoặc ghi âm một khoảng thời gian tối thiểu để đảm bảo đủ dữ liệu
        while touch_sensor.is_touched() or (time.time() - start_time < 2): # Ghi ít nhất 2 giây hoặc cho đến khi nhả sensor
            data = stream.read(CHUNK)
            frames.append(data)
            elapsed_time = time.time() - start_time
            print(f"Recording time: {elapsed_time:.2f} seconds", end="\r")

        stream.stop_stream()
        stream.close()
        print("\nRecording finished.") # Newline after recording time updates
    except Exception as e:
        print(f"Error during audio recording: {e}")
        return
    finally:
        audio.terminate()  # Đảm bảo đóng PyAudio

    try:
        with wave.open(WAVE_OUTPUT_RAW_FILENAME, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        print(f"Raw audio saved to {WAVE_OUTPUT_RAW_FILENAME}")
    except Exception as e:
        print(f"Lỗi khi ghi file âm thanh: {e}")
        return

    try:
        # Re-using the utility functions as per your original structure
        # convert_sample_rate is defined twice in original, using the latter one
        convert_sample_rate(WAVE_OUTPUT_RAW_FILENAME, WAVE_OUTPUT_RESAMPLED_FILENAME, RESAMPLED_RATE)
        sound = AudioSegment.from_file(WAVE_OUTPUT_RESAMPLED_FILENAME, format="wav")
        duplicated_sound = sound * N_TIMES_DUPLICATE
        duplicated_sound.export(WAVE_OUTPUT_RESAMPLED_FILENAME, format="wav")
        print(f"Resampled and duplicated audio saved to {WAVE_OUTPUT_RESAMPLED_FILENAME}")
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return

    try:
        members = query_members(CONN)
        permissions = query_permissions(CONN)
        check_permission = defaultdict(lambda: defaultdict(lambda: False))
        for permission in permissions:
            check_permission[permission.member_name][permission.appliance_name] = True

        audio_file_embedding = inference.get_embedding(WAVE_OUTPUT_RESAMPLED_FILENAME, SPEAKER_RECOGNITION_MODEL)
        speaker_cosine_similarity = {}

        for member_name, embedding in user_embeddings.items():
            if embedding is not None:
                similarity = inference.compute_cosine_similarity(embedding, audio_file_embedding)
                speaker_cosine_similarity[member_name] = similarity
                print(f"\033[94m {member_name}: {similarity}\033[0m")

        if not speaker_cosine_similarity:
            print("Không có embedding hợp lệ để so sánh.")
            return

        predicted_speaker = max(speaker_cosine_similarity, key=speaker_cosine_similarity.get)
        print(f"\033[92mNgười được dự đoán: {predicted_speaker}\033[0m")
    except Exception as e:
        print(f"Error in speaker recognition: {e}")
        return

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(WAVE_OUTPUT_RAW_FILENAME) as source:
            audio_data = recognizer.record(source)
            try:
                content = recognizer.recognize_google(audio_data, language="vi-VN").lower()
                print("Văn bản được nhận dạng: ", content)
            except sr.UnknownValueError:
                print("Không thể nhận dạng văn bản từ âm thanh.")
                return
            except sr.RequestError as e:
                print("Lỗi trong quá trình gửi yêu cầu: ", e)
                return
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        return

    action, device = extract_action_and_device(content)
    print(f"Action: {action} Device: {device}")

    if action is None or device is None:
        speak_text("Thiết bị hoặc hành động không nhận diện được")
        print(f"\033[92mThiết bị hoặc hành động không nhận diện được\033[0m")
    elif check_permission[predicted_speaker][device]:
        speak_text(f"Xin chào {predicted_speaker}. Bạn có quyền {action} {device}")
        print(f"\033[92m{predicted_speaker} có quyền {action} {device}\033[0m")
        control_device(device, action)
    else:
        speak_text(f"Xin chào {predicted_speaker}. Bạn không có quyền {action} {device}")
        print(f"\033[91m{predicted_speaker} không có quyền {action} {device}\033[0m")

# Vòng lặp chính
try:
    print("Ready...")
    # Thêm biến trạng thái để tránh ghi âm liên tục khi sensor vẫn chạm
    is_recording_active = False 
    
    while True:
        # Đọc dữ liệu DHT11 và cập nhật trạng thái (nếu cần gửi đi, nhưng hiện tại đã comment)
        try:
            humidity, temperature = dht.read_dht11()
            if humidity is not None and temperature is not None:
                status_data["Humidity"] = humidity
                status_data["Temperature"] = temperature
        except Exception as e:
            print(f"Lỗi khi đọc cảm biến DHT11: {e}")

        # Gửi trạng thái cập nhật đến ESP32 - Commented out
        # try:
        #     response = requests.post("http://192.168.1.4:10000/message", json=status_data)
        #     if response.status_code == 200:
        #         print("Đã gửi trạng thái đến ESP32")
        #     else:
        #         print(f"Lỗi khi gửi trạng thái: {response.status_code}")
        # except Exception as e:
        #     print(f"Lỗi khi gửi trạng thái đến ESP32: {e}")

        # Logic điều khiển bằng cảm biến chạm
        if touch_sensor.is_touched() and not is_recording_active:
            print("Cảm biến chạm được nhấn!")
            is_recording_active = True # Đặt cờ là đang ghi âm
            record_audio()
        elif not touch_sensor.is_touched() and is_recording_active:
            # Reset cờ khi cảm biến không còn chạm (hoặc sau khi ghi âm đã hoàn thành)
            is_recording_active = False
            print("Cảm biến chạm đã nhả.")
            
        time.sleep(0.1)  # Ngăn sử dụng CPU quá mức
except KeyboardInterrupt:
    print("Dừng chương trình...")
except Exception as e:
    print(f"Lỗi không mong muốn trong vòng lặp chính: {e}")
finally:
    print("Dọn dẹp tài nguyên...")
    # Đảm bảo các thiết bị GPIO được đóng đúng cách
    servo_parent.servo.close()
    led_living.close()
    led_kitchen.close()
    led_children.close()
    led_parent.close()
    led_garage.close()
    motor.pwm.close()
    motor.in1.close()
    motor.in2.close()
    stepper.step_pins.close()
    stepper.step_sequence.close()
    stepper.steps_per_revolution.close()
    stepper.step_sequence.close()
    # Nếu có sensor khác cần đóng, thêm vào đây
    touch_sensor.sensor.close()