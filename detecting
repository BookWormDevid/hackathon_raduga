import cv2
import numpy as np


class SimpleVisualOdometry:
    def __init__(self):
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
        self.camera_position = np.zeros(3)
        self.rotation = np.eye(3)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.feature_detector.detectAndCompute(gray, None)

        if self.prev_frame is not None and self.prev_kp is not None and des is not None and self.prev_des is not None:
            matches = self.matcher.knnMatch(des, self.prev_des, k=2)
            good = [m for m, n in matches if m.distance < 0.7 * n.distance]

            if len(good) > 10:
                src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([self.prev_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0, 0),
                                               method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None:
                    _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts)
                    self.camera_position += self.rotation.dot(t.flatten())
                    self.rotation = R @ self.rotation

        self.prev_frame = gray
        self.prev_kp = kp
        self.prev_des = des
        return self.camera_position, self.rotation


class MultiObjectTracker:
    def init(self):
        self.trackers = []

    def init_tracker(self, frame, bbox):
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)
        self.trackers = [tracker]

    def update(self, frame):
        updated_bboxes = []
        for tracker in self.trackers:
            success, bbox = tracker.update(frame)
            if success:
                updated_bboxes.append(bbox)
        return updated_bboxes


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: камера не найдена!")
        return

    vo = SimpleVisualOdometry()
    tracker = MultiObjectTracker()

    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения кадра")
            return

    # Выбор объекта
    bbox = cv2.selectROI("Выберите объект для отслеживания", frame, False, False)
    if bbox == (0, 0, 0, 0):
        print("Объект не выбран")
        return

    tracker.init_tracker(frame, bbox)
    cv2.destroyAllWindows()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cam_pos, cam_rot = vo.process_frame(frame)
        objects = tracker.update(frame)

        for bbox in objects:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            obj_center = np.array([x + w / 2, y + h / 2, 1])
            obj_pos = cam_rot.T @ (obj_center - cam_pos[:3])

            cv2.putText(frame, f"Pos: {obj_pos[:2]}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(frame, f"Camera: {cam_pos[:2]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Object Tracking & Odometry", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__": 
    main()
