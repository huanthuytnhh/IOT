from gpiozero import DigitalOutputDevice
import adafruit_dht
import board
import time

class DHTSensor:
    def __init__(self, high_pin, data_pin, low_pin):
        self.high_pin = high_pin
        self.data_pin = data_pin
        self.low_pin = low_pin

        # Use gpiozero to control high_pin (VCC) and low_pin (GND)
        self.high_output = DigitalOutputDevice(high_pin)
        self.low_output = DigitalOutputDevice(low_pin)

        # Activate high_pin (VCC) and low_pin (GND)
        self.high_output.on()  # Set to 3.3V
        self.low_output.off()  # Set to GND

        # Map GPIO pin number to board pin (e.g., GPIO19 -> D19)
        try:
            pin_mapping = getattr(board, f"D{data_pin}")
            self.sensor = adafruit_dht.DHT11(pin_mapping)
        except AttributeError:
            raise ValueError(f"Invalid data pin: GPIO{data_pin}. Ensure it maps to a valid board pin (e.g., D19 for GPIO19).")

    def read_dht11(self):
        max_attempts = 5
        for _ in range(max_attempts):
            try:
                temperature = self.sensor.temperature
                humidity = self.sensor.humidity
                if humidity is None:
                    humidity = 0
                if temperature is None:
                    temperature = 0
                return humidity, temperature
            except RuntimeError as e:
                print(f"Error reading DHT11: {e}. Retrying...")
                time.sleep(1)  # Wait before retrying
        print("Failed to retrieve data from DHT11 after multiple attempts.")
        return 0, 0

    def cleanup(self):
        self.high_output.close()
        self.low_output.close()
        self.sensor.exit()  # Clean up DHT sensor

# Test the DHTSensor class
# try:
#     # Initialize DHTSensor with GPIO pins (VCC=13, DATA=19, GND=26)
#     dht_sensor = DHTSensor(high_pin=13, data_pin=19, low_pin=26)
    
#     while True:
#         humidity, temperature = dht_sensor.read_dht11()
#         print(f"Temperature: {temperature:.1f}°C  Humidity: {humidity:.1f}%")
#         time.sleep(2)  # Read data every 2 seconds

# except KeyboardInterrupt:
#     print("Program terminated")
# finally:
#     dht_sensor.cleanup()