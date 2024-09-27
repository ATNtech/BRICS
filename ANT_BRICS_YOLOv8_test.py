import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Загрузка двух моделей
model1 = YOLO("best_traf_sing.pt")
model2 = YOLO("best_traficlight.pt")  # Замените на путь к вашей второй модели

def neural_network_check(frame):
    """Проверяет кадр на наличие объектов с использованием двух моделей YOLO и возвращает результат в виде изображения"""
    img = frame.copy()  # Копируем изображение для рисования
    detected_objects = []  # Список для хранения имен обнаруженных объектов

    # Проверка с первой моделью
    results1 = model1.predict(img, stream=True, conf=0.51)
    for result in results1:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            cv2.rectangle(img, r[:2], r[2:], (0, 0, 255), 3)

            class_id = int(box.cls[0])
            object_name = result.names[class_id]
            detected_objects.append(object_name)
            print(f"Обнаруженный объект (модель 1): {object_name}")

    # Проверка со второй моделью
    results2 = model2.predict(img, stream=True, conf=0.51)
    for result in results2:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            cv2.rectangle(img, r[:2], r[2:], (255, 0, 0), 3)  # Используем другой цвет для второй модели

            class_id = int(box.cls[0])
            object_name = result.names[class_id]
            detected_objects.append(object_name)
            print(f"Обнаруженный объект (модель 2): {object_name}")

    # Выводим все обнаруженные объекты
    if detected_objects:
        unique_objects = set(detected_objects)
        print("Обнаруженные объекты на фото:", unique_objects)
    else:
        print("Объекты не обнаружены.")

    return img

# Загрузка изображения
image_path = 'C:\\Users\\compv\PycharmProjects\\Backend_VisualScout\\sing0.jpg'
frame = cv2.imread(image_path)

# Обработка изображения
processed_image = neural_network_check(frame)

# Показ изображения с обнаруженными объектами
plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Отключаем оси
plt.show()