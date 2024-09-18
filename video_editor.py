# import cv2

# # Список для хранения координат размеченных объектов
# annotations = []

# # Функция обратного вызова для захвата кликов мыши и записи координат
# def click_event(event, x, y, flags, params):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         annotations.append((x, y))
#         # Отображаем точку на видео
#         cv2.circle(params, (x, y), 5, (0, 255, 0), -1)
#         cv2.imshow('Video', params)

# # Открываем видео
# cap = cv2.VideoCapture('videos/colon.mp4')

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Отображаем текущий кадр
#     cv2.imshow('Video', frame)

#     # Связываем событие клика с функцией click_event
#     cv2.setMouseCallback('Video', click_event, frame)

#     # Ждем 30 мс или выхода по нажатию клавиши 'q'
#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         break

# # Освобождаем ресурсы
# cap.release()
# cv2.destroyAllWindows()

# # Выводим координаты размеченных объектов
# print(annotations)

import cv2

# Инициализация трекера
tracker = cv2.TrackerCSRT_create()

# Открываем видео
cap = cv2.VideoCapture('videos/colon.mp4')

ret, frame = cap.read()

# Пользователь размечает объект на первом кадре
bbox = cv2.selectROI('Select Object', frame, False)
cv2.destroyWindow('Select Object')

# Инициализируем трекер с первого кадра
tracker.init(frame, bbox)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Обновляем трекер на каждом кадре
    success, bbox = tracker.update(frame)
    
    if success:
        # Рисуем рамку вокруг объекта
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    # Показываем видео
    cv2.imshow('Tracking', frame)

    # Останавливаем по нажатию 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
