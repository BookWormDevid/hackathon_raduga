import numpy as np
import cv2 as cv
from collections import deque


class DroneTracker:
    def __init__(self):
        # Инициализация детектора
        self.orb = cv.ORB_create(nfeatures=5000, scaleFactor=1.5, nlevels=16)
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

        # Параметры карты (увеличивае размер)
        self.map_size = (4000, 4000)
        self.map_center = np.array([self.map_size[0] // 2, self.map_size[1] // 2])
        self.world_map = np.ones((self.map_size[1], self.map_size[0], 3), dtype=np.uint8) * 255

        # Траектория и состояние
        self.trajectory = deque(maxlen=1000)
        self.position = np.zeros(2, dtype=np.float64)
        self.orientation = 0
        self.scale_factor = 0.5
        self.min_matches = 30

        # Фильтры для сглаживания
        self.position_filter = deque(maxlen=5)

        # Коррекция для перевернутлй камеры
        self.camera_correction = 1  # -1 для камеры вверх ногами

        # Визуализация (уменьшенне окна)
        cv.namedWindow('Trajectory Map', cv.WINDOW_NORMAL)
        cv.resizeWindow('Trajectory Map', 400, 400)  # Уменьшили размер окна карты
        cv.namedWindow('Drone View', cv.WINDOW_NORMAL)
        cv.resizeWindow('Drone View', 1600, 1200)  # Размер окна с видео

    def process_frame(self, frame):
        # Коррекция перевернутой камеры
        frame = cv.flip(frame, 0)  # 0 - flip vertical
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Детекция особенностей
        kp, des = self.orb.detectAndCompute(gray, None)

        if not hasattr(self, 'prev_kp'):
            self.prev_kp = kp
            self.prev_des = des
            self.prev_frame = gray
            return frame

        # Сопоставление особенностей
        matches = self.matcher.knnMatch(self.prev_des, des, k=2)

        # Фильтрация по соотношению расстояний
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) < self.min_matches:
            return frame

        # Получение точек
        prev_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
        curr_pts = np.float32([kp[m.trainIdx].pt for m in good_matches])

        # Вычисление Essential Matrix
        E, mask = cv.findEssentialMat(
            curr_pts, prev_pts,
            focal=1.0, pp=(0, 0),
            method=cv.RANSAC, prob=0.999, threshold=1.0
        )

        # Восстановление позиции
        _, R, t, mask = cv.recoverPose(E, curr_pts, prev_pts)

        # Извлечение угла поворота
        angle = np.linalg.norm(cv.Rodrigues(R)[0])

        # Коррекция движения для перевернутой камеры
        t[1::2] *= self.camera_correction  # Инвертируем ось Y

        # Фильтрация малых движений
        if angle < 0.1 and np.linalg.norm(t) < 0.1:
            t.fill(0)

        # Обновление позиции
        movement = R.dot(t).flatten() * self.scale_factor
        self.position += movement[:2]

        # Фильтрация позиции
        self.position_filter.append(self.position.copy())
        filtered_position = np.mean(self.position_filter, axis=0)

        # Сохранение позиции на карте (с коррекцией Y)
        map_pos = self.map_center + np.array([
            filtered_position[0],
            -filtered_position[1]  # Инвертируем ось Y для карты
        ]).astype(int)

        self.trajectory.append(map_pos)

        # Обновление предыдущих данных
        self.prev_kp = kp
        self.prev_des = des
        self.prev_frame = gray

        # Визуализация
        vis = self.visualize_trajectory(frame)
        return vis

    def visualize_trajectory(self, frame):
        # Создаем копию карты
        map_img = self.world_map.copy()

        # Рисуем траекторию (зеленая)
        if len(self.trajectory) > 1:
            for i in range(1, len(self.trajectory)):
                cv.line(map_img, tuple(self.trajectory[i - 1]), tuple(self.trajectory[i]),
                        (0, 255, 0), 2)

        # Рисуем текущую позицию (красная)
        if len(self.trajectory) > 0:
            cv.circle(map_img, tuple(self.trajectory[-1]), 8, (0, 0, 255), -1)

        # Масштабируем для отображения (уменьшенная область просмотра)
        h, w = 400, 400  # Соответствует размеру окна
        x, y = self.trajectory[-1] if len(self.trajectory) > 0 else self.map_center
        x1, y1 = max(0, x - w // 2), max(0, y - h // 2)
        x2, y2 = min(map_img.shape[1], x1 + w), min(map_img.shape[0], y1 + h)

        map_view = map_img[y1:y2, x1:x2]

        # Переворачиваем карту для правильного отображения
        map_view = cv.flip(map_view, 0)
        cv.imshow('Trajectory Map', map_view)

        # Рисуем ключевые точкки на кадре
        frame_with_kp = cv.drawKeypoints(frame, self.prev_kp, None, color=(0, 255, 0))
        cv.imshow('Drone View', frame_with_kp)

        return frame_with_kp


# Основной цикл
if __name__ == "__main__":
    video_path = 'C:/Users/User.B305C14/PycharmProjects/hackathon_raduga/mapping/ded1.mp4'
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print("Ошибка открытия видео")
        exit()

    tracker = DroneTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tracker.process_frame(frame)
        # ДЛЯ ВЫХОДА ИЗ ПРИЛОЖЕНИЯ ПРОСТО НАЖАТЬ Q (на англише)!!!!
        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()