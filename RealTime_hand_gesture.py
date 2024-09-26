import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

model = load_model(r'C:\Users\mwael\OneDrive\Desktop\after_cource\projects\hand_gesture\hand_recognation_model_no_optimizer.h5')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1) 
mp_draw = mp.solutions.drawing_utils

gesture_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                     'K', 'L', 'M', 'N', 'O', 'P', 'Q', 
                    'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

cap = cv2.VideoCapture(0) 

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

 
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:  
        for hand_landmarks in results.multi_hand_landmarks:
            
            h, w, c = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            for id, lm in enumerate(hand_landmarks.landmark):
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min: x_min = x
                if y < y_min: y_min = y
                if x > x_max: x_max = x
                if y > y_max: y_max = y

        
            hand_img = frame[y_min:y_max, x_min:x_max]

             
            hand_img_resized = cv2.resize(hand_img, (224, 224))
            hand_img_resized = hand_img_resized / 255.0  # تطبيع
            hand_img_resized = np.expand_dims(hand_img_resized, axis=0)

            
            prediction = model.predict(hand_img_resized)
            class_id = np.argmax(prediction)  
            label = gesture_labels[class_id]  

            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

     
    cv2.imshow('Hand Gesture Recognition', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()



# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np

# eyes_model=load_model(r"C:\Users\mwael\OneDrive\Desktop\after_cource\projects\hand_gesture\hand_recognation_model_no_optimizer.h5")

# image_path=r'C:\Users\mwael\OneDrive\Desktop\after_cource\projects\hand_gesture\hand_gesture_data\Train\Q\69_Q.jpg'

# img=image.load_img(image_path,target_size=(224,224))

# img_array = image.img_to_array(img)

# img_array = np.expand_dims(img_array, axis=0)
# img_array /= 255.0

# prediction = eyes_model.predict(img_array)

# predicted_class = np.argmax(prediction, axis=1)
# real_pred=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
#                      'K', 'L', 'M', 'N', 'O', 'P', 'Q', 
#                     'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
# print(f"The image belongs to class: {prediction}")

# print(f"The image belongs to class: {real_pred[predicted_class[0]]}")






















