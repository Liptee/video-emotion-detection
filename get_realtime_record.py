import mediapipe as mp
import cv2
import csv
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

class_name = "Happiness"
with mp_holistic.Holistic() as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(img)

        #FACE
        mp_drawing.draw_landmarks(
            frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
            )

        # #RIGHT HAND
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=1)
            )

        # #LEFT HAND
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=1)  
            )

        #BODY
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=3, circle_radius=1),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=3, circle_radius=1)
            )

        try:
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            row = pose_row + face_row
            row.insert(0, class_name)

            with open('data/face_body.csv', mode = 'a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)


        except:
            pass

        cv2.imshow("WebCam", frame)

        try:       
            num_coords = len(results.pose_landmarks.landmark) + len(results.face_landmarks.landmark)
        except:
            num_coords = 0
        
        landmarks = ['class']
        for val in range(1, num_coords+1):
            landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]        
        
        with open('data/face_body.csv', mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks)  

        if cv2.waitKey(10) and 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()