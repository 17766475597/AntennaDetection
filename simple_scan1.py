# import pygame
from pyrplidar import PyRPlidar, PyRPlidarConnectionError
import time
import socket
import json

def filter_scan(scan):
    if scan.quality == 0 or scan.distance < 1e-3:
        return True

    return False


def UDP_client(message):

    # 创建 UDP 套接字
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 服务器地址
    server_address = ('192.168.1.124', 9955)

    # 发送消息
    data_json = json.dumps(message)
    client_socket.sendto(data_json.encode(), server_address)

    # # 接收响应
    # response, server = client_socket.recvfrom(1024)
    # print(f"收到来自服务器的响应: {response.decode()}")

    # 关闭套接字
    client_socket.close()


def simple_scan():
    
    try:
        lidar = PyRPlidar()
        lidar.connect("COM3", baudrate=115200, timeout=3)
        # Linux   : "/dev/ttyUSB0"
        # MacOS   : "/dev/cu.SLAB_USBtoUART"
        # Windows : "COM5"
        angle_min = 155
        angle_max = 205
        dist_min = 200
        lidar.set_motor_pwm(500)
        time.sleep(1)

        scan_generator = lidar.force_scan()
        anomaly_angle = -1
        duration_time = -1
        cnt = 0
        angle_thres = 30
        # play = False
        max_allowed_cnt = 5
        last_time = time.time()

        for count, scan in enumerate(scan_generator()):
            # if filter_scan(scan) is True:
            #     continue

            data = {}
            alert = False
            health = None
            # health = lidar.get_health()
            # print("health :", health)

            data['health'] = health
            # 字典型数据 health: {'status': 0, 'error_message': ''}
            # 0：正常（设备健康）
            # 1：警告（有轻微问题）
            # 2：错误（设备出现严重问题）
            # 3：未知错误

            if angle_min < scan.angle < angle_max:
                # print(cnt)

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
                        alert = True
                        # if play is False:
                        #    pygame.mixer.music.play()
                        #    play = True
                        print("alert!!!!!!" + " Distance is " + str(scan.distance) + ", Angle is: " + str(scan.angle) + ", cnt is :" + str(cnt))
                        data['alert'] = alert
                        data['message'] = {'Distance': scan.distance, 'Angle': scan.angle}
                        UDP_client(data)
            # if count == 200: break
            if duration_time != -1 and time.time() - duration_time > 2:
                anomaly_angle = -1
                duration_time = -1
                cnt = 0
                # if play is True:
                #        pygame.mixer.music.pause()
                #         play = False
            if time.time() - last_time > 3:
                last_time = time.time()
                data['alert'] = alert
                data['message']  = 0
                UDP_client(data)      
    except PyRPlidarConnectionError as e:
        for i in range(1, 21):
            data = {}
            data['health'] = 1
            print(i)
            UDP_client(data)
            time.sleep(2)





if __name__ == "__main__":
    # pygame.mixer.init()
    # pygame.mixer.music.load("alert.mp3")
    simple_scan()
