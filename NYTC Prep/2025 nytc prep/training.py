# Connect to the ugot
from ugot import ugot 
got = ugot.UGOT()
got.initialize('0.0.0.0')
import time

# Forward/backward movement
# Direction: 0 is forward, 1 is backward
got.mecanum_move_speed_times(direction=0, speed=30, times=50, unit=1)

# Left/right movement
got.mecanum_turn_speed_times(turn=2, speed=45, times=90, unit=2)

# Omnidirectional movement
got.mecanum_translate_speed_times(angle=90, speed=30, times=50, unit=1)

def square_v1():
    for _ in range(4):
        got.mecanum_move_speed_times(direction=0, speed=30, times=50, unit=1)
        got.mecanum_turn_speed_times(turn=2, speed=45, times=90, unit=2)

def square_v2():
    got.mecanum_translate_speed_times(angle=0, speed=30, times=50, unit=1)
    got.mecanum_translate_speed_times(angle=90, speed=30, times=50, unit=1)
    got.mecanum_translate_speed_times(angle=180, speed=30, times=50, unit=1)
    got.mecanum_translate_speed_times(angle=-90, speed=30, times=50, unit=1)

# Robot arm control
got.mechanical_clamp_release()
time.sleep(1)

got.mechanical_joint_control(angle1=0, angle2=0, angle3=-90, duration=500)
time.sleep(1)

got.mechanical_clamp_close()
time.sleep(0.5)

got.mecanum_move_speed_times(direction=0, speed=30, times=50, unit=1)