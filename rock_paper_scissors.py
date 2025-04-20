import cv2
import mediapipe as mp
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from pygame import mixer
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play
File = 'squid.wav'
mixer.init()
mixer.music.load(File)
mixer.music.play(loops=-1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
model = YOLO('six.pt')  



class Player:
    def __init__(self, number):
        self.number = number  
        self.bonus = 0        
        self.choice = None    
        self.detectAction = False  
        self.choice2 = None
        self.detectAction2 = False
        

    def add_bonus(self, points):
        self.bonus += points

    def set_choice(self, choice):
        self.choice = choice
        self.detectAction = True  
    
    def set_choice2(self, choice):
        self.choice2 = choice
        self.detectAction = True

    def reset_choice(self):
        self.choice = None  
        self.detectAction = False

    def __str__(self):
        return f"Player {self.number}, Bonus: {self.bonus}, Choice: {self.choice}"


def determine_winner(player1, player2):
    if player1.choice == player2.choice:
        return None  
    elif (player1.choice == 'rock' and player2.choice == 'scissors') or \
         (player1.choice == 'scissors' and player2.choice == 'paper') or \
         (player1.choice == 'paper' and player2.choice == 'rock'):
        return player1  
    else:
        return player2  
    


player1 = Player(1)
player2 = Player(2)

def wait_for_rock(player1, player2, hands, cap):
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot access the camera.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        h, w, _ = frame.shape
        center_x = w // 2  

        player1.detectAction = False
        player2.detectAction = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                y_max = max([landmark.y for landmark in hand_landmarks.landmark])

                x_min, y_min, x_max, y_max = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)

                padding_factor = 0.2  
                width_padding = int((x_max - x_min) * padding_factor)
                height_padding = int((y_max - y_min) * padding_factor)

                x_min = max(0, x_min - width_padding)
                y_min = max(0, y_min - height_padding)
                x_max = min(w, x_max + width_padding)
                y_max = min(h, y_max + height_padding)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                hand_image = frame[y_min:y_max, x_min:x_max]
                hand_image_resized = cv2.resize(hand_image, (300, 300))
                hand_image_rgb = cv2.cvtColor(hand_image_resized, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(hand_image_rgb)

                results = model(pil_image, verbose=False)
                predictions = results[0].boxes  

                confidence_threshold = 0.5  
                if len(predictions) > 0:
                    for i in range(len(predictions)):
                        confidence = predictions.conf[i].item()
                        if confidence > confidence_threshold:
                            predicted_class = predictions.cls[i].item()
                            if predicted_class == 1:  
                                label = "rock"
                            else:
                                label = "not rock"

                            if x_min < center_x:  
                                player1.set_choice(label)
                            else:  
                                player2.set_choice(label)

        if player1.choice == "rock" and player2.choice == "rock":
            print("Both players are ready! Starting countdown...")
            break

        cv2.putText(frame, "Both players must show ROCK", (50, 50), font, 1, (0, 0, 255), 1)
        cv2.imshow('Hand Detection and Prediction', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break



def check_hand_movement(hand_landmarks, prev_landmarks,frame, threshold=0.02):
    if prev_landmarks is None:
        return True  
    total_movement = 0
    for i in range(len(hand_landmarks.landmark)):
        x_diff = abs(hand_landmarks.landmark[i].x - prev_landmarks.landmark[i].x)
        y_diff = abs(hand_landmarks.landmark[i].y - prev_landmarks.landmark[i].y)
        total_movement += (x_diff + y_diff)

    return total_movement> 0.5 

def count_down_with_movement_check(cap, hands, player1, player2):
    prev_landmarks_p1 = None
    prev_landmarks_p2 = None
    p1_moving = True
    p2_moving = True

    text = "1"
    for i in range(5, 0, -1):
        ret, frame = cap.read()
        if not ret:
            print("Cannot access the camera.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        frame_pil = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(frame_pil)
        font = ImageFont.truetype("arial.ttf", 100)
        text = str(i)

        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        position = ((frame.shape[1] - text_width) // 2, (frame.shape[0] - text_height) // 2)
        draw.text(position, text, font=font, fill=(255, 255, 255))

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                h, w, _ = frame.shape
                center_x = w // 2
                x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                y_max = max([landmark.y for landmark in hand_landmarks.landmark])

                x_min, y_min, x_max, y_max = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)

                padding_factor = 0.2  
                width_padding = int((x_max - x_min) * padding_factor)
                height_padding = int((y_max - y_min) * padding_factor)

                x_min = max(0, x_min - width_padding)
                y_min = max(0, y_min - height_padding)
                x_max = min(w, x_max + width_padding)
                y_max = min(h, y_max + height_padding)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                if hand_landmarks.landmark[0].x * w < center_x:  
                    if check_hand_movement(hand_landmarks, prev_landmarks_p1,frame)==0:
                        p1_moving = False
                    prev_landmarks_p1 = hand_landmarks
                else:  
                    if check_hand_movement(hand_landmarks, prev_landmarks_p2,frame)==0:
                        p2_moving = False
                    prev_landmarks_p2 = hand_landmarks
        cv2.waitKey(1000)
        cv2.imshow('Hand Detection and Prediction', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.putText(frame, f"1 cheat{player1.bonus}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)   
    if not p1_moving:
        player1.add_bonus(-1)
        print(f"Player 1 did not move during the countdown. 1 point deducted.{player1.bonus}")
    if not p2_moving:
        player2.add_bonus(-1)
        print(f"Player 2 did not move during the countdown. 1 point deducted.{player2.bonus}")

    #cv2.destroyWindow('')






def detect_hands():
    cap = cv2.VideoCapture(1)  
    ret, frame = cap.read()  
    if not ret:
        print("Cannot access the camera.")
        return

    
    wins_required = int(input("Enter the number of wins required to declare a winner: "))

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while(player1.bonus < wins_required and player2.bonus < wins_required):
            print(player1.bonus,player2.bonus)
            wait_for_rock(player1, player2, hands, cap)

            if player1.choice == "rock" and player2.choice == "rock":
                cv2.waitKey(1500)
                count_down_with_movement_check(cap,hands,player1,player2)
                cv2.waitKey(1000) 
                player1.reset_choice()
                player2.reset_choice()
            round_is_done = False
            counter = 0
            while cap.isOpened() and round_is_done==False:
                cv2.waitKey(500)
                ret, frame = cap.read()
                if not ret:
                    print("Camera not found")
                    break
               
               
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                h, w, _ = frame.shape
                center_x = w // 2  

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
    
                        x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                        y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                        x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                        y_max = max([landmark.y for landmark in hand_landmarks.landmark])

                        x_min, y_min, x_max, y_max = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)

                        padding_factor = 0.2  
                        width_padding = int((x_max - x_min) * padding_factor)
                        height_padding = int((y_max - y_min) * padding_factor)

                        x_min = max(0, x_min - width_padding)
                        y_min = max(0, y_min - height_padding)
                        x_max = min(w, x_max + width_padding)
                        y_max = min(h, y_max + height_padding)

                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        hand_image = frame[y_min:y_max, x_min:x_max]
                        hand_image_resized = cv2.resize(hand_image, (300, 300))
                        hand_image_rgb = cv2.cvtColor(hand_image_resized, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(hand_image_rgb)

                        results = model(pil_image, verbose=False)
                        predictions = results[0].boxes

                        confidence_threshold = 0.5  
                        if len(predictions) > 0:
                            for i in range(len(predictions)):
                                confidence = predictions.conf[i].item()
                                if confidence > confidence_threshold:
                                    predicted_class = predictions.cls[i].item()
                                    if predicted_class == 0:
                                        label = "paper"
                                    elif predicted_class == 1:
                                        label = "rock"
                                    elif predicted_class == 2:
                                        label = "scissors"
                                    else:
                                        label = "Unknown"
                                    if counter%2!=0:
                                        if x_min < center_x:  
                                            player1.set_choice(label)
                                        else:  
                                            player2.set_choice(label)
                                    else:
                                        if x_min < center_x:  
                                            player1.set_choice2(label)
                                        else:  
                                            player2.set_choice2(label)
                                    cv2.putText(frame, f"{label} ({confidence:.2f})", 
                                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                      
                        if player1.detectAction and player2.detectAction:
                            counter+=1
                            if(counter==2):
                               round_is_done = True
                            else:
                               cv2.waitKey(1500)
                            winner = determine_winner(player1, player2)
                            print(f"player 1 Action is :{player1.choice} ")
                            print(f"player 2 Action is :{player2.choice} ")
                            player1.detectAction = False 
                            player2.detectAction = False
                            if(counter==2):
                                if(player1.choice==player1.choice2 and player2.choice==player2.choice2):
                                    if winner:
                                        winner.add_bonus(1)  
                                        print(f"{winner} wins this round!")
                                        cv2.putText(frame, f"Player {winner.number} Wins!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                                        cv2.putText(frame, f"p1: {player1.bonus}- p2: {player2.bonus}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                                        cv2.imshow('Hand Detection and Prediction', frame)
                                        cv2.waitKey(1000)
                                    else:
                                        print("It's a tie!")

                                    cv2.imshow('Hand Detection and Prediction', frame)
                                    break  
                                else:
                                    if(player1.choice!=player1.choice2):
                                        player1.add_bonus(-1)
                                    else:
                                        player2.add_bonus(-1)

                cv2.imshow('Hand Detection and Prediction', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return

        

    cap.release()
    cv2.destroyAllWindows()




def detect_winner_face(winner_number):
    cap = cv2.VideoCapture(1)  
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    crown_img = cv2.imread('cr.png', cv2.IMREAD_UNCHANGED)  
    
    print(f"Detecting Player {winner_number}'s face...")
    detected_faces = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot access the camera.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        h, w, _ = frame.shape
        center_x = w // 2  

        for (x, y, w, h) in faces:
            if (winner_number == 1 and x < center_x) or (winner_number == 2 and x > center_x):
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                
                crown_resized = cv2.resize(crown_img, (w, int(crown_img.shape[0] * w / crown_img.shape[1])))
                cy = y - crown_resized.shape[0]  

                if cy > 0:
                    for i in range(crown_resized.shape[0]):
                        for j in range(crown_resized.shape[1]):
                            if crown_resized.shape[2] == 4:  
                                if crown_resized[i, j][3] != 0:  
                                    frame[cy + i, x + j] = crown_resized[i, j][:3]  
                            else:
                                frame[cy + i, x + j] = crown_resized[i, j][:3]  
                
               
                zoom_factor = 1.5  
                h_new, w_new = int(h * zoom_factor), int(w * zoom_factor)
                y2, x2 = min(y + h_new, frame.shape[0]), min(x + w_new, frame.shape[1])

                zoomed_face = frame[y:y+h, x:x+w]
                zoomed_face = cv2.resize(zoomed_face, (x2 - x, y2 - y))

                
                frame[y:y2, x:x2] = zoomed_face

                detected_faces.append(frame[y:y+h, x:x+w])
        
        cv2.putText(frame, f"Player {winner_number} Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Winner Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return detected_faces



if __name__ == "__main__":
    detect_hands()
   
    if player1.bonus > player2.bonus:
        print(f"Player 1 wins with bonus: {player1.bonus}")
        print(f"Player 2 lost with bonus: {player2.bonus}")
        detect_winner_face(1)
    else:
        print(f"Player 2 wins with bonus: {player2.bonus}")
        print(f"Player 1 lost with bonus: {player1.bonus}")
        detect_winner_face(2)
