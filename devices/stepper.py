from gpiozero import DigitalOutputDevice
import time

class StepperController:
    def __init__(self, pin1, pin2, pin3, pin4, steps_per_revolution=1024, delay=0.001):
        self.step_pins = [
            DigitalOutputDevice(pin1),
            DigitalOutputDevice(pin2),
            DigitalOutputDevice(pin3),
            DigitalOutputDevice(pin4)
        ]
        self.steps_per_revolution = steps_per_revolution
        self.delay = delay
        self.step_sequence = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1]
        ]
        # Initialize all pins to off
        for pin in self.step_pins:
            pin.off()

    def step(self, sequence):
        for pin_index in range(4):
            if sequence[pin_index] == 1:
                self.step_pins[pin_index].on()
            else:
                self.step_pins[pin_index].off()
        time.sleep(self.delay)

    def rotate(self, direction, revolutions):
        try:
            total_steps = int(self.steps_per_revolution * revolutions)
            if direction == "forward":
                for i in range(total_steps):
                    self.step(self.step_sequence[i % 8])
            elif direction == "backward":
                for i in range(total_steps - 1, -1, -1):
                    self.step(self.step_sequence[i % 8])
            else:
                raise ValueError("Direction must be 'forward' or 'backward'")
        except Exception as e:
            print(f"Error during stepper rotation: {e}")
            raise

    def cleanup(self):
        for pin in self.step_pins:
            pin.off()
            pin.close()

# Test the StepperController class
# if __name__ == "__main__":
#     try:
#         # Initialize StepperController with GPIO pins
#         stepper = StepperController(pin1=21, pin2=20, pin3=16, pin4=12)
        
#         while True:
#             user_input = input("Enter 'open' to rotate forward, 'close' to rotate backward, or 'exit' to quit: ")
#             if user_input == "open":
#                 print("Rotating forward...")
#                 stepper.rotate("forward", 5)
#             elif user_input == "close":
#                 print("Rotating backward...")
#                 stepper.rotate("backward", 5)
#             elif user_input == "exit":
#                 print("Exiting program...")
#                 break
#             else:
#                 print("Invalid input! Use 'open', 'close', or 'exit'.")
    
#     except KeyboardInterrupt:
#         print("Program terminated by user")
#     except Exception as e:
#         print(f"Unexpected error: {e}")
#     finally:
#         stepper.cleanup()