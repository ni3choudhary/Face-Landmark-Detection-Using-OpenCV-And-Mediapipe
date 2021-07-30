import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1) # for webcam

mpdraw = mp.solutions.drawing_utils
mpmeshes = mp.solutions.face_mesh
facemesh = mpmeshes.FaceMesh(max_num_faces=2) 
drawspec = mpdraw.DrawingSpec(thickness=1, circle_radius=2)

previous_time = 0
current_time = 0

while True:
    
    _ , frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = facemesh.process(img)
    
    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            mpdraw.draw_landmarks(frame, facelms, mpmeshes.FACE_CONNECTIONS, drawspec, drawspec)
                
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    
    cv2.putText(frame, "fps: "+str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
    cv2.imshow('Face Landmarks Detection', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()