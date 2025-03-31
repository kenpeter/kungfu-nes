# test_keyboard.py
import keyboard
import time

print("Press any key to test (Ctrl+C to exit)...")
while True:
    if keyboard.is_pressed('left'):
        print("Left Arrow pressed!")
    if keyboard.is_pressed('z'):
        print("Z pressed!")
    if keyboard.is_pressed('q'):
        print("Q pressed! Exiting...")
        break
    time.sleep(0.1)  # Avoid CPU overload