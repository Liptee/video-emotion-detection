import mediapipe as mp
import cv2
import csv
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

geusture = "ok"

with mp_holistic.Holistic() as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(img)

        # #RIGHT HAND
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=1)
            )

        try:
            right_hand = results.right_hand_landmarks.landmark
            right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())

            right_hand_row.insert(0, geusture)

            with open('data/geusture.csv', mode = 'a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(right_hand_row)

        except:
            pass
        
        cv2.imshow("WebCam", frame)

        # try:
        #     num_coords = len(results.right_hand_landmarks.landmark)
        # except:
        #     num_coords = 0

        # landmarks = ['class']
        # for val in range(1, num_coords + 1):
        #     landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

        # with open('data/geusture.csv', mode='w', newline='') as f:
        #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #     csv_writer.writerow(landmarks)     

        if cv2.waitKey(10) and 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()