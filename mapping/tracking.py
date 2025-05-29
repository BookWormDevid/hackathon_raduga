import cv2
import sys
import os

# Создаем папку для сохранения фото, если её нет
if not os.path.exists('photos'):
    os.makedirs('photos')

if __name__ == '__main__':
    # Инициализация трекера CSRT
    tracker = cv2.TrackerCSRT_create()

    # Инициализация веб-камеры
    video = cv2.VideoCapture(0)

    # Проверка подключения камеры
    if not video.isOpened():
        print("Не удалось открыть веб-камеру")
        sys.exit()

    # Чтение первого кадра
    ok, frame = video.read()
    if not ok:
        print("Не удалось получить кадр с веб-камеры")
        sys.exit()

    # Выбор объекта для отслеживания
    bbox = cv2.selectROI("Выберите объект для отслеживания", frame, False)
    cv2.destroyWindow("Выберите объект для отслеживания")

    # Сохраняем кадр с выделенным объектом
    frame_with_selection = frame.copy()
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame_with_selection, p1, p2, (255, 0, 0), 2, 1)
    cv2.imwrite('photos/selected_object.jpg', frame_with_selection)
    print("Кадр с выделенным объектом сохранен в photos/selected_object.jpg")

    # Инициализация трекера
    ok = tracker.init(frame, bbox)

    while True:
        ok, frame = video.read()
        if not ok:
            break

        # Таймер для FPS
        timer = cv2.getTickCount()

        # Обновление трекера
        ok, bbox = tracker.update(frame)

        # Расчет FPS
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Отрисовка bounding box
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            cv2.putText(frame, "Ошибка отслеживания", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Отображение информации
        cv2.putText(frame, "CSRT Tracker", (100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Покадровая запись (опционально)
        # cv2.imwrite(f'photos/frame_{int(timer)}.jpg', frame)

        # Показ результата
        cv2.imshow("Tracking", frame)

        # Выход по ESC
        if cv2.waitKey(1) == 27:
            break

    # Освобождение ресурсов
    video.release()
    cv2.destroyAllWindows()
