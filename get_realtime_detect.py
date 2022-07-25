import mediapipe as mp
import cv2
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

exit_model = 'models\model_geusture_gb.pkl'
with open(exit_model, 'rb') as f:
    exit_model = pickle.load(f)

file_model = 'models/model_rf.pkl'
with open(file_model, 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)
def main():
    drawing = True
    with mp_holistic.Holistic() as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(img)

            try:             
                reght_hand = results.right_hand_landmarks.landmark
                right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in reght_hand]).flatten())

                RH_X = pd.DataFrame([right_hand_row])

                if exit_model.predict(RH_X)[0] == 'ok':
                    return 0

                elif exit_model.predict(RH_X)[0] == 'five':
                    drawing = False

                elif exit_model.predict(RH_X)[0] == 'piece':
                    drawing = True
            except:
                pass

            if drawing:

            #FACE
                mp_drawing.draw_landmarks(
                    frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                    )

                # # #RIGHT HAND
                # mp_drawing.draw_landmarks(
                #     frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                #     mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=1),
                #     mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=1)
                #     )

                # # #LEFT HAND
                # mp_drawing.draw_landmarks(
                #     frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                #     mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=1),
                #     mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=1)  
                #     )

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
                X = pd.DataFrame([row])
                #X = scaler(X)
                predict = model.predict(X)[0]
                prob = model.predict_proba(X)[0]
                
    # DISPLAY STATUS BOX
                cv2.rectangle(
                    frame, 
                    (0,0), 
                    (250, 60), 
                    (245,117,16), 
                    -1)

    # DISPLAY CLASS
                cv2.putText(
                    frame, 
                    'CLASS', 
                    (95,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0,0,0), 
                    1, 
                    cv2.LINE_AA)

                cv2.putText(
                    frame, 
                    predict.split(' ')[0], 
                    (90, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA)

    #DISPLAY PROB
                cv2.putText(
                    frame, 
                    'PROB', 
                    (15, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0,0,0), 
                    1, 
                    cv2.LINE_AA)

                cv2.putText(
                    frame, 
                    str(round(prob[np.argmax(prob)],2)), 
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA)

            except:
                pass

            cv2.imshow("WebCam", frame)

            try:       
                num_coords = len(results.pose_landmarks.landmark) + len(results.face_landmarks.landmark)
            except:
                num_coords = 0
            
            if cv2.waitKey(10) and 0xFF == ord('q'):
                break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()