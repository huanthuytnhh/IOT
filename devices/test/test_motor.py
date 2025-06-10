from gpiozero import PWMOutputDevice, DigitalOutputDevice
import time

# Define GPIO pins
ENA = 14  # PWM pin for speed control
IN1 = 15  # Direction pin 1
IN2 = 18  # Direction pin 2

# Initialize devices
pwm = PWMOutputDevice(ENA, frequency=50)  # 50 Hz PWM
in1 = DigitalOutputDevice(IN1)
in2 = DigitalOutputDevice(IN2)

def open_door():
    print("Opening door...")
    in1.on()   # IN1 HIGH (forward)
    in2.off()  # IN2 LOW
    pwm.value = 0.35  # Increased to 50% duty cycle for faster speed
    time.sleep(5)    # Run for 5 seconds
    pwm.value = 0    # Stop motor
    print("Door opened.")

def close_door():
    print("Closing door...")
    in1.off()  # IN1 LOW
    in2.on()   # IN2 HIGH (reverse)
    pwm.value = 0.35  # Increased to 50% duty cycle for faster speed
    time.sleep(5)    # Run for 5 seconds
    pwm.value = 0    # Stop motor
    print("Door closed.")

try:
    # Example sequence: open, wait, close, wait
    open_door()
    time.sleep(2)  # Pause for 2 seconds
    close_door()
    time.sleep(2)  # Pause for 2 seconds

finally:
    # Ensure motor is stopped and pins are reset
    pwm.value = 0
    in1.off()
    in2.off()