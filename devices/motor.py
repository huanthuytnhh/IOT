from gpiozero import PWMOutputDevice, DigitalOutputDevice
from time import sleep

class MotorController:
    def __init__(self, enable_pin, in1_pin, in2_pin):
        # Initialize GPIO pins using gpiozero
        self.pwm = PWMOutputDevice(enable_pin, frequency=50)  # PWM for speed control
        self.in1 = DigitalOutputDevice(in1_pin)  # Direction pin 1
        self.in2 = DigitalOutputDevice(in2_pin)  # Direction pin 2

    def stop_motor(self):
        """Stop the motor by disabling PWM and direction pins."""
        self.pwm.value = 0
        self.in1.off()
        self.in2.off()

    def open_door(self, speed=0.35):
        """Start motor to open the door (e.g., forward direction)."""
        print("Opening door...")
        self.in1.on()   # Set direction for opening
        self.in2.off()
        self.pwm.value = speed  # Set speed
        print(f"Motor running at {speed*100}% speed for opening.")

    def close_door(self, speed=0.35):
        """Start motor to close the door (e.g., reverse direction)."""
        print("Closing door...")
        self.in1.off()  # Set direction for closing
        self.in2.on()
        self.pwm.value = speed  # Set speed
        print(f"Motor running at {speed*100}% speed for closing.")

    def open_close_sequence(self, time_to_wait, open_duration=2, close_duration=2, speed=0.35):
        """Open and close door for fixed durations without limit switches."""
        try:
            # Open the door
            self.open_door(speed)
            sleep(open_duration)  # Run motor for open_duration seconds
            self.stop_motor()
            print("Door opened and motor stopped.")
            
            # Wait before closing
            print(f"Waiting for {time_to_wait} seconds...")
            sleep(time_to_wait)
            
            # Close the door
            self.close_door(speed)
            sleep(close_duration)  # Run motor for close_duration seconds
            self.stop_motor()
            print("Door closed and motor stopped.")
            
        except Exception as e:
            print(f"Error during operation: {e}")
            self.stop_motor()

# try:
#     # Initialize controller with specified GPIO pins (BCM numbering)
#     # Ensure these pins match your physical wiring to the H-bridge
#     door_controller = MotorController(
#         enable_pin=14,  # ENA pin on your H-bridge (PWM for speed)
#         in1_pin=15,     # IN1 pin on your H-bridge (Direction 1)
#         in2_pin=18      # IN2 pin on your H-bridge (Direction 2)
#     )
#     print("Motor controller initialized.")
#     # Example usage: open, wait, then close with default speed
#     door_controller.open_close_sequence(time_to_wait=3, open_duration=2, close_duration=2, speed=0.35) # Increased speed to 70%

# except KeyboardInterrupt:
#     print("Program interrupted by user.")
# except Exception as e:
#     print(f"An unexpected error occurred during initialization or operation: {e}")
# finally:
#     if 'door_controller' in locals() and door_controller:
#         door_controller.stop_motor()
#         print("Motor stopped and GPIO cleaned up.")