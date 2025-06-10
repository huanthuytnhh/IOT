import os
# from dotenv import load_dotenv
import time
from gpiozero import DigitalOutputDevice
import adafruit_dht
import board
import requests
import subprocess

# Load environment variables
# load_dotenv()

# Configuration from .env
DHT_HIGH_PIN = 13
DHT_DATA_PIN = 19
DHT_LOW_PIN = 26
# ESP32_MAC_ADDRESS = os.getenv("ESP32_MAC_ADDRESS", "a0:a3:b3:ab:2e:10")
# ESP32_PORT = os.getenv("ESP32_PORT", "10000")

# Initialize GPIO pins using gpiozero
high_output = DigitalOutputDevice(DHT_HIGH_PIN)  # VCC pin
low_output = DigitalOutputDevice(DHT_LOW_PIN)    # GND pin
high_output.on()   # Set VCC to 3.3V
low_output.off()   # Set GND to 0V

# Initialize DHT11 sensor
try:
    pin_mapping = getattr(board, f"D{DHT_DATA_PIN}")
    dht_sensor = adafruit_dht.DHT11(pin_mapping)
except AttributeError:
    raise ValueError(f"Invalid data pin: GPIO{DHT_DATA_PIN}. Ensure it maps to a valid board pin (e.g., D19 for GPIO19).")

# Data dictionary for API
# data = {
#     "Garage Led": 0,
#     "Garage Door": 0,
#     "Living Led": 0,
#     "Kitchen Led": 0,
#     "Parent Led": 0,
#     "Children Led": 0,
#     "Temperature": 0,
#     "Humidity": 0,
# }

# def find_ip_by_mac(target_mac):
#     try:
#         cmd = ['arp', '-a']
#         returned_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
#         decoded_output = returned_output.decode('utf-8')
#         lines = decoded_output.split('\n')
#         for line in lines:
#             if target_mac.lower() in line.lower():
#                 ip = line.split()[1][1:-1]  # Extract IP address
#                 return ip
#         return None
#     except subprocess.CalledProcessError as e:
#         print(f"Error running arp command: {e}")
#         return None

# def send_api(humidity, temperature):
#     ip_address = find_ip_by_mac(ESP32_MAC_ADDRESS)
#     if not ip_address:
#         print("Cannot find ESP32 with MAC address:", ESP32_MAC_ADDRESS)
#         return
#     url = f'http://{ip_address}:{ESP32_PORT}/message'
#     data["Humidity"] = humidity
#     data["Temperature"] = temperature
#     try:
#         response = requests.post(url, data=data, timeout=10)
#         if response.status_code == 200:
#             print("API request successful!")
#         else:
#             print(f"API request failed with status code: {response.status_code}")
#     except requests.RequestException as e:
#         print(f"Error sending API request: {e}")

def read_dht11():
    max_attempts = 5
    for _ in range(max_attempts):
        try:
            temperature = dht_sensor.temperature
            humidity = dht_sensor.humidity
            if humidity is None:
                humidity = 0
            if temperature is None:
                temperature = 0
            print(f"Temperature: {temperature:.1f}°C  Humidity: {humidity:.1f}%")
            # send_api(humidity, temperature)
            return
        except RuntimeError as e:
            print(f"Error reading DHT11: {e}. Retrying...")
            time.sleep(1)
    print("Failed to retrieve data from DHT11 after multiple attempts.")

try:
    while True:
        read_dht11()
        time.sleep(2)  # Read data every 2 seconds
except KeyboardInterrupt:
    print("Program terminated")
finally:
    high_output.close()
    low_output.close()
    dht_sensor.exit()