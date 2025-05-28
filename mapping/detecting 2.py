import cv2
import numpy as np


class SimpleVisualOdometry:
    def __init__(self, camera_matrix=None, dist_coeffs=None):
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
        self.camera_position = np.zeros(3)
        self.rotation = np.eye(3)

        # Параметры камеры (можно заменить на реальные значения калибровки)
        self.camera_matrix = camera_matrix if camera_matrix is not None else np.array([
            [1000, 0, 640],
            [0, 1000, 360],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(4)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.feature_detector.detectAndCompute(gray, None)

        if self.prev_frame is not None and self.prev_kp is not None and des is not None and self.prev_des is not None:
            matches = self.matcher.knnMatch(des, self.prev_des, k=2)
            good = [m for m, n in matches if m.distance < 0.7 * n.distance]

            if len(good) > 10:
                src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([self.prev_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                # Используем параметры камеры для более точных вычислений
                E, mask = cv2.findEssentialMat(src_pts, dst_pts,
                                               cameraMatrix=self.camera_matrix,
                                               method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None:
                    _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, cameraMatrix=self.camera_matrix)
                    self.camera_position += self.rotation.dot(t.flatten())
                    self.rotation = R @ self.rotation

        self.prev_frame = gray
        self.prev_kp = kp
        self.prev_des = des
        return self.camera_position.copy(), self.rotation.copy()


class MultiObjectTracker:
    def __init__(self):
        self.trackers = []
        self.tracker_types = ['CSRT', 'KCF', 'MOSSE']
        self.current_tracker_type = 0

    def init_tracker(self, frame, bbox):
        # Создаем новый трекер
        tracker = cv2.TrackerCSRT_create() if self.current_tracker_type == 0 else \
            cv2.TrackerKCF_create() if self.current_tracker_type == 1 else \
                cv2.TrackerMOSSE_create()

        tracker.init(frame, bbox)
        self.trackers.append({
            'tracker': tracker,
            'bbox': bbox,
            'color': (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)),
            'id': len(self.trackers) + 1
        })
        return True

    def update(self, frame):
        updated_bboxes = []
        for tracker_info in self.trackers[:]:
            success, bbox = tracker_info['tracker'].update(frame)
            if success:
                tracker_info['bbox'] = bbox
                updated_bboxes.append(tracker_info)
            else:
                # Удаляем трекер если потеряли объект
                self.trackers.remove(tracker_info)
        return updated_bboxes


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: камера не найдена!")
        return

    # Инициализация визуальной одометрии
    vo = SimpleVisualOdometry()

    # Инициализация трекера
    tracker = MultiObjectTracker()

    # Создаем окно с параметрами
    cv2.namedWindow("Object Tracking & Odometry")
    cv2.createTrackbar('Tracker', 'Object Tracking & Odometry', 0, 2, lambda x: None)

    selecting_object = False
    bbox = (0, 0, 0, 0)
    objects = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Обновляем тип трекера
        tracker.current_tracker_type = cv2.getTrackbarPos('Tracker', 'Object Tracking & Odometry')

        # Обработка визуальной одометрии
        cam_pos, cam_rot = vo.process_frame(frame)

        # Обновление трекеров
        tracked_objects = tracker.update(frame)

        # Отрисовка треков
        for obj in tracked_objects:
            x, y, w, h = [int(v) for v in obj['bbox']]
            cv2.rectangle(frame, (x, y), (x + w, y + h), obj['color'], 2)

            # Вычисление положения объекта относительно камеры
            obj_center = np.array([x + w / 2, y + h / 2, 1])
            obj_pos = cam_rot.T @ (obj_center - cam_pos[:3])

            cv2.putText(frame, f"ID:{obj['id']} Pos:{obj_pos[:2].round(2)}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj['color'], 1)

        # Отрисовка положения камеры
        cv2.putText(frame, f"Camera Pos: {cam_pos[:2].round(2)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Инструкция для пользователя
        cv2.putText(frame, "Нажмите 's' для выбора объекта", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Нажмите 'q' для выхода", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Выбор объекта
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            bbox = cv2.selectROI("Object Tracking & Odometry", frame, False, False)
            if bbox != (0, 0, 0, 0):
                tracker.init_tracker(frame, bbox)
            cv2.destroyWindow("Object Tracking & Odometry")
        elif key == ord('q'):
            break

        cv2.imshow("Object Tracking & Odometry", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()