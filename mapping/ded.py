import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Инициализация детектора ORB
orb = cv.ORB_create()  # type: ignore

# Открываем видеофайл
video_path = 'C:/Users/Студент Т/Documents/GitHub/hackathon_raduga/ded.mp4' # C:/Users/User.B305C14/PycharmProjects/hackathon_raduga/mapping/ded1.mp4
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть видеофайл")
    exit()

# Для хранения предыдущего кадра и ключевых точек
prev_frame = None
prev_kp = None
prev_des = None

# Для визуализации пути
path_image = None
path_points = []  # Initialize path_points here

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Конвертируем в оттенки серого
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Находим ключевые точки и дескрипторы
    kp, des = orb.detectAndCompute(gray, None)

    # Если это не первый кадр, находим соответствия
    if prev_frame is not None and prev_kp is not None:
        # Создаем matcher
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(prev_des, des)

        # Сортируем matches по расстоянию
        matches = sorted(matches, key=lambda x: x.distance)

        # Отбираем лучшие matches
        good_matches = matches[:20]

        # Получаем координаты точек
        prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Вычисляем среднее смещение
        if len(prev_pts) > 0 and len(curr_pts) > 0:
            mean_shift = np.mean(curr_pts - prev_pts, axis=0)

            # Инициализируем изображение для пути, если нужно
            if path_image is None:
                path_image = np.zeros_like(frame)
                path_start = (frame.shape[1] // 2, frame.shape[0] // 2)
                path_points = [path_start]

            # Добавляем новую точку пути
            last_point = path_points[-1]
            new_point = (int(last_point[0] + mean_shift[0][0]),
                         int(last_point[1] + mean_shift[0][1]))
            path_points.append(new_point)

            # Рисуем путь
            for i in range(1, len(path_points)):
                cv.line(path_image, path_points[i - 1], path_points[i], (0, 255, 0), 2)

    # Обновляем предыдущий кадр и точки
    prev_frame = gray.copy()
    prev_kp = kp
    prev_des = des

    # Рисуем ключевые точки на текущем кадре
    frame_with_kp = cv.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=0)

    # Накладываем путь на кадр
    if path_image is not None:
        combined = cv.addWeighted(frame_with_kp, 0.7, path_image, 0.3, 0)
    else:
        combined = frame_with_kp

    # Показываем результат
    cv.imshow('Drone Path Tracking', combined)

    # Выход по нажатию 'q'
    if cv.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
