#!/usr/bin/env python

import rospy
from clover import srv
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
from clover import long_callback
import cv2
from ultralytics import YOLO
from sensor_msgs.msg import Image

rospy.init_node('my_node')
bridge = CvBridge()

# Создаем сервис-прокси
get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
set_color = rospy.ServiceProxy("led/set_effect", srv.SetLEDEffect)
land = rospy.ServiceProxy('land', Trigger)

# Загружаем модель YOLO
model = YOLO('best_trafic.pt')

# Глобальная переменная для хранения последнего кадра
last_frame = None

def neural_network_check(frame):
    results = model.predict(frame, stream=True, conf=0.75)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            cv2.rectangle(frame, r[:2], r[2:], (0, 0, 255), 3)

            object_name = result.names[box.cls[0].astype(int)]
            print(f"Обнаруженный объект: {object_name}")
    return frame

@long_callback
def image_callback(data):
    global last_frame
    last_frame = bridge.imgmsg_to_cv2(data, 'bgr8')

def main():
    global last_frame

    set_color(r=255, g=0, b=0)  # Устанавливаем цвет LED
    navigate(x=0, y=0, z=5, frame_id='body', auto_arm=True)  # Взлетаем на 5 метров
    rospy.sleep(6)

    distance_traveled = 0
    step_distance = 5

    while distance_traveled < 20:
        navigate(x=step_distance, y=0, z=5, frame_id='body')
        rospy.sleep(5)

        if last_frame is not None:  # Проверяем, есть ли последний кадр
            processed_frame = neural_network_check(last_frame)

        distance_traveled += step_distance

    land()

if __name__ == '__main__':
    main()