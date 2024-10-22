import pygame
from pyrplidar import PyRPlidar
import time


def filter_scan(scan):
    if scan.quality == 0 or scan.distance < 1e-3:
        return True

    return False


def simple_scan():
    lidar = PyRPlidar()
    lidar.connect(port="/dev/cu.SLAB_USBtoUART", baudrate=115200, timeout=3)
    # Linux   : "/dev/ttyUSB0"
    # MacOS   : "/dev/cu.SLAB_USBtoUART"
    # Windows : "COM5"
    angle_min = 155
    angle_max = 205
    dist_min = 250

    lidar.set_motor_pwm(500)
    time.sleep(1)

    scan_generator = lidar.force_scan()

    anomaly_angle = -1
    duration_time = -1
    cnt = 0
    angle_thres = 30
    play = False
    max_allowed_cnt = 5

    for count, scan in enumerate(scan_generator()):

        # if filter_scan(scan) is True:
        #     continue

        if angle_min < scan.angle < angle_max:
            print(cnt)

            # print(count, scan)
            if 0 < scan.distance < dist_min:
                print(scan)
                if anomaly_angle == -1:
                    anomaly_angle = scan.angle
                    duration_time = time.time()

                if anomaly_angle != -1:
                    if max(anomaly_angle-angle_thres, 0) <= scan.angle <= min(anomaly_angle+angle_thres, 360):
                        cnt += 1
                        anomaly_angle = scan.angle
                        duration_time = time.time()

                if cnt >= max_allowed_cnt:
                    if play is False:
                        pygame.mixer.music.play()
                        play = True
                    print("alert!!!!!!" + " Distance is " + str(scan.distance) + ", Angle is: " + str(scan.angle) + ", cnt is :" + str(cnt))
        # if count == 200: break
        if duration_time != -1 and time.time() - duration_time > 2:
            anomaly_angle = -1
            duration_time = -1
            cnt = 0
            if play is True:
                pygame.mixer.music.pause()
                play = False

    lidar.stop()
    lidar.set_motor_pwm(0)

    lidar.disconnect()


if __name__ == "__main__":
    pygame.mixer.init()
    pygame.mixer.music.load("alert.mp3")
    simple_scan()
