import os
from dotenv import load_dotenv
import requests
import subprocess
import time

class API:
    def __init__(self, mac_address, port="10000"):
        self.mac_address = mac_address.lower()  # Normalize MAC address
        self.port = port
        self.ip_address = self.find_ip()

    def find_ip(self):
        """Find IP address for the given MAC address using arp -a."""
        try:
            cmd = ['arp', '-a']
            returned_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            decoded_output = returned_output.decode('utf-8')
            lines = decoded_output.split('\n')
            for line in lines:
                if self.mac_address in line.lower():
                    ip = line.split()[1][1:-1]  # Extract IP address
                    print(f"Found IP: {ip} for MAC: {self.mac_address}")
                    return ip
            print(f"MAC address {self.mac_address} not found in ARP table.")
            return None
        except subprocess.CalledProcessError as e:
            print(f"Error running arp command: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error in find_ip: {e}")
            return None

    def send_data(self, data):
        """Send data to the ESP32 server as JSON."""
        if not self.ip_address:
            print("Cannot send data: IP address not found. Retrying to find IP...")
            self.ip_address = self.find_ip()
            if not self.ip_address:
                print("Failed to find IP address. Data not sent.")
                return False
        try:
            url = f'http://{self.ip_address}:{self.port}/message'
            # Send data as JSON instead of form-encoded
            response = requests.post(url, json=data, timeout=10)
            if response.status_code == 200:
                print("API request successful!")
                return True
            else:
                print(f"API request failed with status code: {response.status_code}")
                return False
        except requests.RequestException as e:
            print(f"Error sending API request: {e}")
            return False

# Test the API class
if __name__ == "__main__":
    try:
        # Load environment variables
        load_dotenv()
        mac_address = os.getenv("MAC_ADDRESS", "a0:b7:65:04:4e:e4")
        port = os.getenv("ESP32_PORT", "10000")

        # Initialize API
        api = API(mac_address, port)

        # Test data
        data = {
            "Garage Led": 0,
            "Garage Door": 0,
            "Living Led": 0,
            "Kitchen Led": 0,
            "Parent Led": 0,
            "Children Led": 0,
            "Temperature": 0,
            "Humidity": 0,
        }

        # Send data multiple times with retry
        max_attempts = 3
        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1}/{max_attempts}")
            if api.send_data(data):
                break
            time.sleep(2)  # Wait before retrying
        else:
            print("Failed to send data after all attempts.")

    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"Unexpected error: {e}")