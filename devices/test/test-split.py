from pydub import AudioSegment
import os

def cut_wav_to_3_minutes(input_file):
    try:
        # Load the WAV file
        audio = AudioSegment.from_wav(input_file)
        
        # Define 3 minutes in milliseconds (3 * 60 * 1000)
        three_minutes_ms = 3 * 60 * 1000
        
        # Cut the audio to the first 3 minutes
        cut_audio = audio[:three_minutes_ms]
        
        # Create a temporary file to store the cut audio
        temp_file = input_file + ".temp.wav"
        cut_audio.export(temp_file, format="wav")
        
        # Delete the original file
        os.remove(input_file)
        
        # Rename the temporary file to the original filename
        os.rename(temp_file, input_file)
        
        print(f"Successfully cut {input_file} to 3 minutes and overwritten the original file.")
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

# Example usage
input_wav = "/home/pi/Desktop/09_06/IOT/audio_samples/Sum-Merge_Audio.wav"
cut_wav_to_3_minutes(input_wav)