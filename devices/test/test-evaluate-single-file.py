import os
import torch
# import neural_net
import time
# import inference
import myconfig
import csv
from pydub import AudioSegment
import tempfile
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import speaker_recognition.inference as inference
import speaker_recognition.neural_net as neural_net
N_TIMES_DUPLICATE = 5

def extend_audio(audio, times=5):
    return audio * times

def get_embedding_from_audiosegment(audio, encoder):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        audio.export(tmpfile.name, format="wav")
        embedding = inference.get_embedding(tmpfile.name, encoder)
    os.remove(tmpfile.name)
    return embedding

# # Load pre-trained encoder
encoder_path = r"/home/pi/Desktop/09_06/IOT/saved_model/best-model-train-clean-360-hours-50000-epochs-specaug-80-mfcc-100-seq-len-full-inference-8-batch-3-stacks/saved_model_20240606215142.pt"
encoder = neural_net.get_speaker_encoder(encoder_path)

start_time = time.time()

# # Load and extend base audio files
tri_audio = AudioSegment.from_wav(r"/home/pi/Desktop/09_06/IOT/audio_samples/Tri-Merge_Audio.wav")
phat_audio = AudioSegment.from_wav(r"/home/pi/Desktop/09_06/IOT/audio_samples/Phat-Merge_Audio.wav")
thanh_audio = AudioSegment.from_wav(r"/home/pi/Desktop/09_06/IOT/audio_samples/Thanh-Merge_Audio.wav")
quan_audio = AudioSegment.from_wav(r"/home/pi/Desktop/09_06/IOT/audio_samples/Quan-Merge_Audio.wav")
# sum_audio = AudioSegment.from_wav(r"/home/pi/Desktop/09_06/IOT/audio_samples/Sum-Merge_Audio.wav")
quang_audio = AudioSegment.from_wav(r"/home/pi/Desktop/09_06/IOT/audio_samples/Quang-Merge_Audio.wav")

extended_tri_audio = extend_audio(tri_audio, times=N_TIMES_DUPLICATE)
extended_thanh_audio = extend_audio(thanh_audio, times=N_TIMES_DUPLICATE)
extended_quang_audio = extend_audio(quang_audio, times=N_TIMES_DUPLICATE)
extended_quan_audio = extend_audio(quan_audio, times=N_TIMES_DUPLICATE)
# extended_sum_audio = extend_audio(sum_audio, times=N_TIMES_DUPLICATE)
extended_phat_audio = extend_audio(phat_audio, times=N_TIMES_DUPLICATE)

tri_base_embedding = get_embedding_from_audiosegment(extended_tri_audio, encoder)
thanh_base_embedding = get_embedding_from_audiosegment(extended_thanh_audio, encoder)
quan_base_embedding = get_embedding_from_audiosegment(extended_quan_audio, encoder)
# sum_base_embedding = get_embedding_from_audiosegment(extended_sum_audio, encoder)
quang_base_embedding = get_embedding_from_audiosegment(extended_quang_audio, encoder)
phat_base_embedding = get_embedding_from_audiosegment(extended_phat_audio, encoder)

data_source = r"/home/pi/Desktop/09_06/IOT/devices/test/audio-test-latest"

total_prediction = 0
accurate_prediction = 0
users = ["Tri", "Thanh", "Quang", "Phat", "Quan"]
user_index = {user: i for i, user in enumerate(users)}
confusion_matrix = np.zeros((len(users), len(users)), dtype=int)

for user in os.listdir(data_source):
    user_folder_path = os.path.join(data_source, user)
    if not os.path.isdir(user_folder_path): # Bỏ qua nếu không phải là thư mục
        continue
    for audio_file in os.listdir(user_folder_path):
        audio_file_path = os.path.join(user_folder_path, audio_file)
        if not audio_file.lower().endswith(".wav"): # Chỉ xử lý file .wav
            continue
        try:
            audio = AudioSegment.from_wav(audio_file_path)
        except Exception as e:
            print(f"Lỗi khi đọc file {audio_file_path}: {e}")
            continue

        extended_audio = extend_audio(audio, times=N_TIMES_DUPLICATE)
        audio_file_embedding = get_embedding_from_audiosegment(extended_audio, encoder)

        tri_distance = inference.compute_distance(tri_base_embedding, audio_file_embedding)
        thanh_distance = inference.compute_distance(thanh_base_embedding, audio_file_embedding)
        quang_distance = inference.compute_distance(quang_base_embedding, audio_file_embedding)
        quan_distance = inference.compute_distance(quan_base_embedding, audio_file_embedding)
        # sum_distance = inference.compute_distance(sum_base_embedding, audio_file_embedding)
        phat_distance = inference.compute_distance(phat_base_embedding, audio_file_embedding)

        # data_distance = [tri_distance, thanh_distance, quang_distance, phat_distance,quan_distance, sum_distance]
        data_distance = [tri_distance, thanh_distance, quang_distance, phat_distance,quan_distance]
        prediction = users[data_distance.index(min(data_distance))]

        if user == prediction:
            # THAY ĐỔI Ở ĐÂY: Thêm audio_file vào print
            print(f"\033[94mSpeaker: {user} Predict: {prediction} File: {audio_file} [TRUE]\033[0m")
            accurate_prediction += 1
        else:
            # THAY ĐỔI Ở ĐÂY: Thêm audio_file vào print
            print(f"\033[91mSpeaker: {user} Predict: {prediction} File: {audio_file} [FALSE]\033[0m")

        confusion_matrix[user_index[user], user_index[prediction]] += 1
        total_prediction += 1

end_time = time.time()
print(f"Tổng thời gian xử lý: {end_time - start_time:.2f} giây")

base_embeddings = {
    "Tri": tri_base_embedding,
    "Thanh": thanh_base_embedding,
    "Quang": quang_base_embedding,
    "Phat": phat_base_embedding,
    "Quan": quan_base_embedding
   
}
users_list = list(base_embeddings.keys())
print("\nKhoảng cách giữa các embedding cơ sở:")
for i in range(len(users_list)):
    for j in range(i + 1, len(users_list)):
        u1 = users_list[i]
        u2 = users_list[j]
        dist = inference.compute_distance(base_embeddings[u1], base_embeddings[u2])
        print(f"Distance {u1} vs {u2}: {dist:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix)

print(f"\nModel: {encoder_path}")
if 'myconfig' in globals() and hasattr(myconfig, 'SEQ_LEN'):
    print(f"Seq length: {myconfig.SEQ_LEN}")
else:
    print("Không tìm thấy myconfig.SEQ_LEN")
print(f"Accurate prediction: {accurate_prediction}")
print(f"Total prediction: {total_prediction}")
if total_prediction > 0:
    print(f"Accuracy: {accurate_prediction / total_prediction * 100:.2f}%")
else:
    print("Không có dự đoán nào được thực hiện.")

# Plot Confusion Matrix
plt.figure(figsize=(10, 8)) # Tăng kích thước để dễ nhìn hơn
sns.set_theme(font_scale=1.0) # Điều chỉnh font scale nếu cần
sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=users, yticklabels=users, annot_kws={"size": 10}) # Điều chỉnh kích thước số trong ô
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Confusion Matrix", fontsize=14)
plt.xticks(rotation=45, ha="right") # Xoay nhãn trục x để dễ đọc hơn
plt.yticks(rotation=0)
plt.tight_layout() # Tự động điều chỉnh để vừa vặn
plt.show()

# """
# import os
# import torch
# import neural_net
# import time
# import inference
# import myconfig
# import csv
# from pydub import AudioSegment

# def extend_audio(audio, times=5):
#     return audio * times

# # Load pre-trained encoder
# encoder_path = r"D:\Code\BachKhoa\PBL 5\PBL05_smart_home_with_voice_print_and_antifraud_ai\saved_model_20240606215142.pt"
# encoder = neural_net.get_speaker_encoder(encoder_path)

# start_time = time.time()

# # Load and extend base audio files
# tri_audio = AudioSegment.from_wav(r"D:\Code\BachKhoa\PBL 5\PBL05_smart_home_with_voice_print_and_antifraud_ai\Trí-200-5.wav\Tri_base_200s.wav")
# dat_audio = AudioSegment.from_wav(r"D:\Code\BachKhoa\PBL 5\PBL05_smart_home_with_voice_print_and_antifraud_ai\Đạt-200-5.wav\Dat_base_200s.wav")
# tuan_audio = AudioSegment.from_wav(r"D:\Code\BachKhoa\PBL 5\PBL05_smart_home_with_voice_print_and_antifraud_ai\Tuấn-200-5.wav\Tuan_base_200s.wav")
# phat_audio = AudioSegment.from_wav(r"D:\Code\BachKhoa\PBL 5\PBL05_smart_home_with_voice_print_and_antifraud_ai\Phát-200-5.wav\Phat_base_200s.wav")

# tri_base_embedding = inference.get_embedding(tri_audio, encoder)
# dat_base_embedding = inference.get_embedding(dat_audio, encoder)
# tuan_base_embedding = inference.get_embedding(tuan_audio, encoder)
# phat_base_embedding = inference.get_embedding(phat_audio, encoder)

# data_source = r"D:\Code\BachKhoa\PBL 5\PBL05_smart_home_with_voice_print_and_antifraud_ai\test-dir"

# total_prediction = 0 
# accurate_prediction = 0

# for user in os.listdir(data_source):
#     user_folder_path = os.path.join(data_source, user)
#     for audio_file in os.listdir(user_folder_path):
#         audio_file_path = os.path.join(user_folder_path, audio_file)
#         audio = AudioSegment.from_wav(audio_file_path)
#         extended_audio = extend_audio(audio, times=5)
#         audio_file_embedding = inference.get_embedding(extended_audio, encoder)
        
#         tri_distance = inference.compute_distance(tri_base_embedding, audio_file_embedding)
#         dat_distance = inference.compute_distance(dat_base_embedding, audio_file_embedding)
#         tuan_distance = inference.compute_distance(tuan_base_embedding, audio_file_embedding)
#         phat_distance = inference.compute_distance(phat_base_embedding, audio_file_embedding)
        
#         data_distance = [tri_distance, dat_distance, tuan_distance, phat_distance]

#         users = ["Trí", "Đạt", "Tuấn", "Phát"]
#         prediction = users[data_distance.index(min(data_distance))]
        
#         if user == prediction:
#             print(f"\033[94mSpeaker: {user} Predict: {prediction} [TRUE]\033[0m")
#             accurate_prediction += 1
#         else:
#             print(f"\033[91mSpeaker: {user} Predict: {prediction} [FALSE]\033[0m")
        
#         total_prediction += 1

# end_time = time.time()
# print(end_time - start_time)

# print(f"Model: {encoder_path}")
# print(f"Seq length: {myconfig.SEQ_LEN}")
# print(f"Accurate prediction: {accurate_prediction}")
# print(f"Total prediction: {total_prediction}")
# print(f"Accuracy: {accurate_prediction / total_prediction * 200}%")
# """