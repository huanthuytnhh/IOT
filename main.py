import os
from dotenv import load_dotenv
from collections import defaultdict, Counter
import pyaudio
import wave
from pydub import AudioSegment
import speech_recognition as sr
import numpy as np
import time
from flask import Flask, request, jsonify
import threading
import requests
import ast
from scipy.spatial.distance import cosine
from devices.servo import ServoController
from devices.motor import MotorController
from devices.stepper import StepperController
from devices.led import Led
from devices.dht11 import DHTSensor
from devices.touch import TouchSensor
import speaker_recognition.neural_net as neural_net
import speaker_recognition.inference as inference
from db.db_helper import query_members_files, query_permissions, connect_db
from utils import extract_action_and_device, speak_text # Removed unused imports
import logging
import json
import tempfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

SPEAKER_RECOGNITION_MODEL_PATH = os.getenv("SPEAKER_RECOGNITION_MODEL_PATH")
DB_PATH = os.getenv("DB_PATH")
N_TIMES_DUPLICATE = int(os.getenv("N_TIMES_DUPLICATE", 5)) # Default if not set
FORMAT = eval(os.getenv("FORMAT", "pyaudio.paInt16"))
CHANNELS = int(os.getenv("CHANNELS", 1))
RATE = int(os.getenv("RATE", 44100))
CHUNK = int(os.getenv("CHUNK", 1024))
RAW_RECORDING_PATH = os.getenv("RAW_RECORDING_PATH", "/tmp")
RESAMPLED_RATE = int(os.getenv("RESAMPLED_RATE", 16000))
WAVE_OUTPUT_RAW_FILENAME = os.path.join(RAW_RECORDING_PATH, "output_raw.wav")
WAVE_OUTPUT_RESAMPLED_FILENAME = os.path.join(RAW_RECORDING_PATH, "output_resampled.wav")
ESP32_IP_ADDRESS = os.getenv("ESP32_IP_ADDRESS", "192.168.1.199") # Default ESP32 IP

SPEAKER_RECOGNITION_MODEL = neural_net.get_speaker_encoder(SPEAKER_RECOGNITION_MODEL_PATH)
CONN = connect_db(DB_PATH)

try:
    motor = MotorController( enable_pin=14,in1_pin=15,in2_pin=18 )
    stepper = StepperController(21, 20, 16, 12)
    servo_parent = ServoController(7)

    led_living = Led(4)
    led_kitchen = Led(17)
    led_children = Led(10)
    led_parent = Led(11)
    led_garage = Led(5)
    dht = DHTSensor(13, 19, 26)
    touch_sensor = TouchSensor(22)
    GPIO_INITIALIZED = True
    logger.info("GPIO devices initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing GPIO devices: {e}. Running in simulation mode for GPIO.", exc_info=True)
    GPIO_INITIALIZED = False
    # Create mock objects if GPIO fails
    class MockGPIOObject:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): pass
        def __getattr__(self, name):
            if name == 'close': return lambda: logger.info(f"Mock close called for {self.__class__.__name__}")
            return MockGPIOObject()
        def on(self): logger.info(f"Mock {self.__class__.__name__} ON")
        def off(self): logger.info(f"Mock {self.__class__.__name__} OFF")
        def rotate(self, direction, steps): logger.info(f"Mock Stepper rotate {direction} {steps} steps")
        def open_door_close_door(self, *args, **kwargs): logger.info(f"Mock Servo action called")
        def read_dht11(self): logger.info("Mock DHT read"); return 70, 25.0
        def is_touched(self): logger.info("Mock Touch read"); return False # Simulate not touched by default

    motor = MockGPIOObject()
    stepper = MockGPIOObject()
    servo_parent = MockGPIOObject()
    led_living = MockGPIOObject()
    led_kitchen = MockGPIOObject()
    led_children = MockGPIOObject()
    led_parent = MockGPIOObject()
    led_garage = MockGPIOObject()
    dht = MockGPIOObject()
    touch_sensor = MockGPIOObject()


status_data = {
    "Garage Led": 0,
    "Garage Door": 0,
    "Living Led": 0,
    "Kitchen Led": 0,
    "Parent Led": 0,
    "Children Led": 0,
    "Temperature": 0.0,
    "Humidity": 0,
}

def convert_sample_rate_internal(input_filename, output_filename, target_sample_rate):
    try:
        sound = AudioSegment.from_file(input_filename)
        sound = sound.set_frame_rate(target_sample_rate)
        sound.export(output_filename, format="wav")
    except Exception as e:
        logger.error(f"Lỗi khi chuyển đổi sample rate cho '{input_filename}': {e}", exc_info=True)
        raise

def extend_audio_internal(audio_segment, times=N_TIMES_DUPLICATE):
    if times > 1:
        logger.info(f"Nhân bản âm thanh {times} lần để kéo dài thời lượng...")
    return audio_segment * times

def get_embedding_from_audiosegment(audio, encoder):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmp_path = tmpfile.name
    try:
        audio.export(tmp_path, format="wav")
        embedding = inference.get_embedding(tmp_path, encoder)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return embedding

def control_device_from_voice(device_voice, action_voice):
    global status_data
    logger.info(f"Voice command: Device='{device_voice}', Action='{action_voice}'")
    try:
        original_status = status_data.copy()
        if device_voice == "cửa phòng khách":
            if GPIO_INITIALIZED: motor.open_door_close_door(time_to_wait=3, open_duration=2, close_duration=2, speed=0.35)
            logger.info(f"Đã {action_voice} cửa phòng khách (motor)")
        elif device_voice == "cửa nhà xe":
            if action_voice == "mở":
                if GPIO_INITIALIZED: stepper.rotate("forward", 5)
                status_data["Garage Door"] = 1
            elif action_voice == "đóng":
                if GPIO_INITIALIZED: stepper.rotate("backward", 5)
                status_data["Garage Door"] = 0
            logger.info(f"Đã {action_voice} cửa nhà xe (stepper). New status: {status_data['Garage Door']}")
        elif device_voice == "cửa phòng ngủ ba mẹ":
            # Assuming action_voice is implicitly open/close cycle
            if GPIO_INITIALIZED: servo_parent.open_door_close_door(angle=180, time_to_wait=6)
            logger.info(f"Đã thực hiện chu trình cửa phòng ngủ ba mẹ (servo)")
        elif device_voice == "đèn phòng khách":
            if action_voice == "bật":
                if GPIO_INITIALIZED: led_living.on()
                status_data["Living Led"] = 1
            elif action_voice == "tắt":
                if GPIO_INITIALIZED: led_living.off()
                status_data["Living Led"] = 0
            logger.info(f"Đã {action_voice} đèn phòng khách. New status: {status_data['Living Led']}")
        elif device_voice == "đèn phòng bếp":
            if action_voice == "bật":
                if GPIO_INITIALIZED: led_kitchen.on()
                status_data["Kitchen Led"] = 1
            elif action_voice == "tắt":
                if GPIO_INITIALIZED: led_kitchen.off()
                status_data["Kitchen Led"] = 0
            logger.info(f"Đã {action_voice} đèn phòng bếp. New status: {status_data['Kitchen Led']}")
        elif device_voice == "đèn phòng ngủ ba mẹ":
            if action_voice == "bật":
                if GPIO_INITIALIZED: led_parent.on()
                status_data["Parent Led"] = 1
            elif action_voice == "tắt":
                if GPIO_INITIALIZED: led_parent.off()
                status_data["Parent Led"] = 0
            logger.info(f"Đã {action_voice} đèn phòng ngủ ba mẹ. New status: {status_data['Parent Led']}")
        elif device_voice == "đèn phòng ngủ con cái":
            if action_voice == "bật":
                if GPIO_INITIALIZED: led_children.on()
                status_data["Children Led"] = 1
            elif action_voice == "tắt":
                if GPIO_INITIALIZED: led_children.off()
                status_data["Children Led"] = 0
            logger.info(f"Đã {action_voice} đèn phòng ngủ con cái. New status: {status_data['Children Led']}")
        elif device_voice == "đèn nhà xe":
            if action_voice == "bật":
                if GPIO_INITIALIZED: led_garage.on()
                status_data["Garage Led"] = 1
            elif action_voice == "tắt":
                if GPIO_INITIALIZED: led_garage.off()
                status_data["Garage Led"] = 0
            logger.info(f"Đã {action_voice} đèn nhà xe. New status: {status_data['Garage Led']}")
        elif device_voice == "cảm biến":
            if action_voice == "xem":
                humidity, temperature = dht.read_dht11() if GPIO_INITIALIZED else (70, 25.5)
                if humidity is not None and temperature is not None:
                    status_data["Humidity"] = int(humidity)
                    status_data["Temperature"] = float(temperature)
                    speak_text(f"Nhiệt độ hiện tại là {temperature} độ C và độ ẩm là {humidity} %")
                    logger.info(f"Đã xem cảm biến: Temp={temperature}, Humid={humidity}")
        else:
            logger.warning(f"Thiết bị '{device_voice}' không được hỗ trợ cho điều khiển bằng giọng nói.")
            return

        if status_data != original_status:
            send_status_to_esp()
    except Exception as e:
        logger.error(f"Lỗi khi RPi điều khiển thiết bị '{device_voice}' bằng giọng nói: {e}", exc_info=True)

def control_device_from_esp(device_name_from_esp, state_from_esp):
    global status_data
    logger.info(f"ESP command: Device='{device_name_from_esp}', State='{state_from_esp}'")
    success = False
    message = "Device or action not recognized by ESP handler"
    original_status = status_data.copy()

    action_pi = "bật" if state_from_esp == 1 else ("tắt" if state_from_esp == 0 else None)
    if device_name_from_esp == "garageDoor": # Door has open/close
        action_pi = "mở" if state_from_esp == 1 else ("đóng" if state_from_esp == 0 else None)

    if action_pi is None:
        message = f"Invalid state '{state_from_esp}' from ESP for device '{device_name_from_esp}'"
        logger.warning(message)
        return success, message
    
    try:
        if device_name_from_esp == "garageLed":
            if action_pi == "bật":
                if GPIO_INITIALIZED: led_garage.on()
                status_data["Garage Led"] = 1
            elif action_pi == "tắt":
                if GPIO_INITIALIZED: led_garage.off()
                status_data["Garage Led"] = 0
            success = True
        elif device_name_from_esp == "garageDoor":
            if action_pi == "mở":
                if GPIO_INITIALIZED: stepper.rotate("forward", 5)
                status_data["Garage Door"] = 1
            elif action_pi == "đóng":
                if GPIO_INITIALIZED: stepper.rotate("backward", 5)
                status_data["Garage Door"] = 0
            success = True
        elif device_name_from_esp == "livingLed":
            if action_pi == "bật":
                if GPIO_INITIALIZED: led_living.on()
                status_data["Living Led"] = 1
            elif action_pi == "tắt":
                if GPIO_INITIALIZED: led_living.off()
                status_data["Living Led"] = 0
            success = True
        elif device_name_from_esp == "kitchenLed":
            if action_pi == "bật":
                if GPIO_INITIALIZED: led_kitchen.on()
                status_data["Kitchen Led"] = 1
            elif action_pi == "tắt":
                if GPIO_INITIALIZED: led_kitchen.off()
                status_data["Kitchen Led"] = 0
            success = True
        elif device_name_from_esp == "parentLed":
            if action_pi == "bật":
                if GPIO_INITIALIZED: led_parent.on()
                status_data["Parent Led"] = 1
            elif action_pi == "tắt":
                if GPIO_INITIALIZED: led_parent.off()
                status_data["Parent Led"] = 0
            success = True
        elif device_name_from_esp == "childrenLed":
            if action_pi == "bật":
                if GPIO_INITIALIZED: led_children.on()
                status_data["Children Led"] = 1
            elif action_pi == "tắt":
                if GPIO_INITIALIZED: led_children.off()
                status_data["Children Led"] = 0
            success = True
        else:
            message = f"Device '{device_name_from_esp}' not handled by ESP control logic."
            logger.warning(message)

        if success:
            message = f"Successfully executed from ESP: {action_pi} {device_name_from_esp}"
            logger.info(message)
            if status_data != original_status: # Only send if status actually changed
                 send_status_to_esp()
        else:
            logger.warning(f"ESP command for '{device_name_from_esp}' with action '{action_pi}' did not result in success flag.")


    except Exception as e:
        logger.error(f"Lỗi khi RPi điều khiển thiết bị '{device_name_from_esp}' từ ESP: {e}", exc_info=True)
        message = f"Error controlling {device_name_from_esp} from ESP"
        success = False
    
    return success, message

app = Flask(__name__)

@app.route('/esp-control', methods=['POST'])
def esp_control_command_route():
    try:
        data = request.get_json()
        if not data:
            logger.error("ESP-Control: No JSON data received")
            return jsonify({"status": "error", "message": "No JSON data received"}), 400

        logger.info(f"ESP-Control: Received data from ESP32: {data}")
        device_from_esp = data.get('device')
        state_from_esp = data.get('state')

        if device_from_esp is not None and state_from_esp is not None:
            success, message = control_device_from_esp(device_from_esp, state_from_esp)
            if success:
                return jsonify({"status": "success", "message": message, "current_pi_state": status_data}), 200
            else:
                return jsonify({"status": "error", "message": message, "current_pi_state": status_data}), 400
        else:
            logger.error("ESP-Control: Lệnh từ ESP32 không hợp lệ (thiếu device hoặc state)")
            return jsonify({"status": "error", "message": "Lệnh từ ESP32 không hợp lệ"}), 400
    except Exception as e:
        logger.error(f"Lỗi trong endpoint /esp-control của Flask: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Lỗi xử lý yêu cầu trên RPi: {str(e)}"}), 500

def run_flask_server():
    try:
        logger.info("Starting Flask server for ESP32 commands on port 5000")
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except Exception as e:
        logger.error(f"Failed to start Flask server: {e}", exc_info=True)

flask_thread = threading.Thread(target=run_flask_server)
flask_thread.daemon = True

def send_status_to_esp():
    global status_data
    try:
        humidity, temperature = dht.read_dht11() if GPIO_INITIALIZED else (status_data.get("Humidity", 70), status_data.get("Temperature", 25.0))
        if humidity is not None and temperature is not None:
            status_data["Humidity"] = int(humidity)
            status_data["Temperature"] = float(temperature)
    except Exception as e:
        logger.warning(f"Lỗi khi đọc cảm biến DHT11 để gửi trạng thái: {e}")

    logger.debug(f"Attempting to send status to ESP32 ({ESP32_IP_ADDRESS}): {status_data}")
    try:
        response = requests.post(f"http://{ESP32_IP_ADDRESS}:10000/message", json=status_data, timeout=3)
        if response.status_code == 200:
            logger.info(f"Đã gửi trạng thái đến ESP32 ({ESP32_IP_ADDRESS}) thành công.")
        else:
            logger.error(f"Lỗi khi gửi trạng thái đến ESP32 ({ESP32_IP_ADDRESS}): {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Lỗi kết nối khi gửi trạng thái đến ESP32 ({ESP32_IP_ADDRESS}): {e}")
    except Exception as e:
        logger.error(f"Lỗi không xác định khi gửi trạng thái đến ESP32 ({ESP32_IP_ADDRESS}): {e}", exc_info=True)

def record_and_process_audio():
    audio_interface = pyaudio.PyAudio()
    stream = None
    try:
        stream = audio_interface.open(format=FORMAT, channels=CHANNELS,
                                      rate=RATE, input=True,
                                      frames_per_buffer=CHUNK)
        logger.info("Recording...")
        frames = []
        start_time = time.time()
        # Keep recording as long as sensor is touched, or for at least 2 seconds if sensor not working/mocked
        while (GPIO_INITIALIZED and touch_sensor.is_touched()) or \
              (not GPIO_INITIALIZED and (time.time() - start_time < 3)) or \
              (time.time() - start_time < 2) : # Min 2s recording
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            elapsed_time = time.time() - start_time
            print(f"Recording time: {elapsed_time:.2f} seconds", end="\r")
            if not GPIO_INITIALIZED and elapsed_time >= 3: # Stop mock recording after 3s
                 break
        print("\n")
        logger.info("Recording finished.")
    except Exception as e:
        logger.error(f"Error during audio recording: {e}", exc_info=True)
        return
    finally:
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except Exception as e_stream:
                logger.error(f"Error closing audio stream: {e_stream}")
        audio_interface.terminate()

    if not frames:
        logger.warning("No audio frames recorded.")
        return

    try:
        os.makedirs(RAW_RECORDING_PATH, exist_ok=True)
        with wave.open(WAVE_OUTPUT_RAW_FILENAME, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio_interface.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        logger.info(f"Raw audio saved to {WAVE_OUTPUT_RAW_FILENAME}")
    except Exception as e:
        logger.error(f"Error saving raw audio file: {e}", exc_info=True)
        return

    try:
        convert_sample_rate_internal(WAVE_OUTPUT_RAW_FILENAME, WAVE_OUTPUT_RESAMPLED_FILENAME, RESAMPLED_RATE)
        sound = AudioSegment.from_file(WAVE_OUTPUT_RESAMPLED_FILENAME, format="wav")
        duplicated_sound = extend_audio_internal(sound, times=N_TIMES_DUPLICATE) # Use N_TIMES_DUPLICATE
        duplicated_sound.export(WAVE_OUTPUT_RESAMPLED_FILENAME, format="wav")
        logger.info(f"Resampled and duplicated audio saved to {WAVE_OUTPUT_RESAMPLED_FILENAME}")
    except Exception as e:
        logger.error(f"Error processing audio file for speaker recognition: {e}", exc_info=True)
        return

    try:
        member_files = query_members_files(CONN)
        if not member_files:
            logger.error("No member files found in database for speaker recognition.")
            return

        db_embeddings = defaultdict(list)
        expected_length = 128
        for member_file in member_files:
            features_str = member_file['features']
            member_name = member_file['member_name']
            if not features_str:
                logger.warning(f"Features are None/empty for {member_name} (file: {member_file['file_path']})")
                continue
            try:
                # Assuming features are stored as a JSON string of a list or list of lists
                parsed_features_list = json.loads(features_str)
                if not isinstance(parsed_features_list, list):
                    logger.warning(f"Parsed features for {member_name} is not a list: {type(parsed_features_list)}")
                    continue

                # Handle cases where features might be a list of embeddings or a single embedding list
                if parsed_features_list and isinstance(parsed_features_list[0], list): # List of embeddings
                    # Average them or take the first one, for simplicity taking average
                    temp_embeddings = [np.array(emb_item).flatten() for emb_item in parsed_features_list if len(np.array(emb_item).flatten()) == expected_length]
                    if temp_embeddings:
                        embedding = np.mean(temp_embeddings, axis=0)
                    else:
                        logger.warning(f"No valid embeddings of length {expected_length} found for {member_name} after parsing list of lists.")
                        continue
                else: # Single embedding list
                    embedding = np.array(parsed_features_list).flatten()


                if len(embedding) == expected_length:
                    db_embeddings[member_name].append(embedding)
                else:
                    logger.warning(f"Embedding length mismatch for {member_name}: got {len(embedding)}, expected {expected_length}. File: {member_file['file_path']}")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse features JSON for {member_name}: {features_str[:100]}...")
            except Exception as e_emb:
                logger.error(f"Error processing embedding for {member_name}: {e_emb}", exc_info=True)


        if not db_embeddings:
            logger.error("No valid database embeddings loaded for speaker recognition.")
            return

        input_audio_embedding = inference.get_embedding(WAVE_OUTPUT_RESAMPLED_FILENAME, SPEAKER_RECOGNITION_MODEL)
        if input_audio_embedding is None:
            logger.error("Failed to generate embedding for input audio.")
            return
        input_audio_embedding = np.array(input_audio_embedding).flatten()

        if len(input_audio_embedding) != expected_length:
            logger.warning(f"Input audio embedding length {len(input_audio_embedding)} does not match expected {expected_length}. Adjusting.")
            if len(input_audio_embedding) > expected_length:
                input_audio_embedding = input_audio_embedding[:expected_length]
            else:
                input_audio_embedding = np.pad(input_audio_embedding, (0, expected_length - len(input_audio_embedding)), 'constant')


        similarities = {}
        for speaker, embeddings_list in db_embeddings.items():
            speaker_similarities = [1 - cosine(input_audio_embedding, db_emb) for db_emb in embeddings_list]
            if speaker_similarities:
                similarities[speaker] = np.mean(speaker_similarities)
            else:
                similarities[speaker] = -1 # Should not happen if db_embeddings is populated correctly

        if not similarities:
            logger.error("Could not compute similarities for any speaker.")
            return

        predicted_speaker = max(similarities, key=similarities.get)
        max_similarity = similarities[predicted_speaker]
        
        logger.info(f"Speaker recognition similarities: {similarities}")
        logger.info(f"Predicted Speaker: {predicted_speaker} (Similarity: {max_similarity:.4f})")
        
        # Threshold for recognition (adjust as needed)
        recognition_threshold = 0.65 # Example threshold
        if max_similarity < recognition_threshold:
            logger.warning(f"Speaker {predicted_speaker} recognized with low confidence ({max_similarity:.4f}). Treating as unknown.")
            speak_text("Không nhận dạng được người nói hoặc độ tin cậy thấp.")
            return


        permissions = query_permissions(CONN)
        user_has_permission = defaultdict(lambda: defaultdict(bool))
        for p_row in permissions:
            user_has_permission[p_row.member_name][p_row.appliance_name] = True
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(WAVE_OUTPUT_RAW_FILENAME) as source: # Use raw for STT
            audio_data_for_stt = recognizer.record(source)
        
        try:
            recognized_text_content = recognizer.recognize_google(audio_data_for_stt, language="vi-VN").lower()
            logger.info(f"Google STT recognized: '{recognized_text_content}'")
        except sr.UnknownValueError:
            logger.warning("Google Speech Recognition could not understand audio")
            speak_text("Tôi không hiểu bạn nói gì.")
            return
        except sr.RequestError as e:
            logger.error(f"Could not request results from Google Speech Recognition service; {e}")
            speak_text("Lỗi dịch vụ nhận dạng giọng nói.")
            return

        action_voice, device_voice = extract_action_and_device(recognized_text_content)
        logger.info(f"Extracted Action: {action_voice}, Device: {device_voice}")

        if action_voice is None or device_voice is None:
            speak_text("Thiết bị hoặc hành động không nhận diện được từ câu lệnh.")
            # logger.warning(f"Thiết bị hoặc hành động không nhận diện được từ: '{recognized_text_content}'")
        elif user_has_permission[predicted_speaker].get(device_voice, False):
            speak_text(f"Xin chào {predicted_speaker}. Bạn có quyền {action_voice} {device_voice}")
            # logger.info(f"{predicted_speaker} có quyền {action_voice} {device_voice}")
            control_device_from_voice(device_voice, action_voice)
        else:
            speak_text(f"Xin chào {predicted_speaker}. Bạn không có quyền {action_voice} {device_voice}")
            logger.warning(f"{predicted_speaker} không có quyền {action_voice} {device_voice}")

    except Exception as e:
        logger.error(f"Lỗi trong quá trình nhận dạng người nói hoặc giọng nói: {e}", exc_info=True)
        speak_text("Đã xảy ra lỗi trong quá trình xử lý.")

if __name__ == "__main__":
    try:
        logger.info("Initializing Raspberry Pi Smart Home System...")
        if not flask_thread.is_alive():
             flask_thread.start()
             logger.info("Flask server thread started.")
        else:
            logger.info("Flask server thread already running.")

        is_currently_recording = False
        last_status_update_to_esp = time.time()
        status_update_interval = 10 

        # Initial status send
        send_status_to_esp()

        speak_text(" Hệ thống nhà thông minh đã sẵn sàng. Bạn có thể bắt đầu điều khiển bằng giọng nói hoặc cảm biến chạm.")
        # logger.info("Smart Home System is Ready...")
        while True:
            current_loop_time = time.time()

            if current_loop_time - last_status_update_to_esp > status_update_interval:
                send_status_to_esp()
                last_status_update_to_esp = current_loop_time
            
            sensor_touched_now = touch_sensor.is_touched() if GPIO_INITIALIZED else False

            if sensor_touched_now and not is_currently_recording:
                logger.info("Cảm biến chạm được nhấn! Bắt đầu ghi âm.")
                is_currently_recording = True
                record_and_process_audio() # This is a blocking call
                is_currently_recording = False # Reset after processing is done
                logger.info("Hoàn tất xử lý ghi âm. Cảm biến chạm đã sẵn sàng.")
            elif not sensor_touched_now and is_currently_recording:
                # This case might not be hit if record_and_process_audio is blocking
                # and sensor is released during that time.
                # is_currently_recording = False
                # logger.info("Cảm biến chạm đã nhả trong khi đang xử lý (hoặc đã xong).")
                pass

            time.sleep(0.05) # Shorter sleep for responsiveness
    except KeyboardInterrupt:
        logger.info("Dừng chương trình do KeyboardInterrupt...")
    except Exception as e:
        logger.error(f"Lỗi không mong muốn nghiêm trọng trong vòng lặp chính: {e}", exc_info=True)
    finally:
        logger.info("Dọn dẹp tài nguyên trên Raspberry Pi...")
        if GPIO_INITIALIZED:
            try:
                if 'servo_parent' in locals() and hasattr(servo_parent, 'close'): servo_parent.close()
                if 'led_living' in locals() and hasattr(led_living, 'close'): led_living.close()
                if 'led_kitchen' in locals() and hasattr(led_kitchen, 'close'): led_kitchen.close()
                if 'led_children' in locals() and hasattr(led_children, 'close'): led_children.close()
                if 'led_parent' in locals() and hasattr(led_parent, 'close'): led_parent.close()
                if 'led_garage' in locals() and hasattr(led_garage, 'close'): led_garage.close()
                if 'motor' in locals() and hasattr(motor, 'close'): motor.close()
                if 'stepper' in locals() and hasattr(stepper, 'close'): stepper.close()
                if 'touch_sensor' in locals() and hasattr(touch_sensor, 'close'): touch_sensor.close()
                # dht sensor usually doesn't need explicit close with common libraries
            except Exception as e_cleanup:
                logger.error(f"Lỗi khi dọn dẹp GPIO: {e_cleanup}")

        if CONN:
            try:
                CONN.close()
                logger.info("Database connection closed.")
            except Exception as e_db_close:
                logger.error(f"Lỗi khi đóng kết nối DB: {e_db_close}")
        
        # Attempt to shutdown Flask server if it was started by this script
        # This is a bit tricky with daemon threads, usually OS handles it on exit
        # For a cleaner shutdown, one might use a shared event to signal the Flask thread.
        logger.info("Raspberry Pi system shutdown sequence complete.")