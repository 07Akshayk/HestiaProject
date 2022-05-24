from cgitb import grey
import pygame
import webbrowser
from pygame.locals import *
import numpy as np
import imutils
from imutils.video import VideoStream
from directkeys import PressKey, A, D, Space, ReleaseKey
import mediapipe as mp
import cv2
import numpy as np
import time
from playsound import playsound
import pyautogui
from time import time
from math import hypot
import mediapipe as mp
import matplotlib.pyplot as plt
import os
from pygame import mixer


pygame.init()


width = 600
height = 600

screen = pygame.display.set_mode((width, height))
mixer.init()
mixer.music.load('music.mp3')
mixer.music.play()

# Set title to the window
#pygame.display.set_caption("Hello World")


#background_image = pygame.image.load("C:\\Users\\user\\Desktop\\Python\\my game\\_pycache_\\pic.jpg").convert()

bg = pygame.image.load('pic.jpg').convert()
bg = pygame.transform.scale(bg,(width,height))



pygame.display.set_caption('GAME HUB')

font = pygame.font.SysFont('Constantia', 30)

# define colours
#bg = pygame.image.load("C:\\Users\\user\\Desktop\\Python\\my game\\_pycache_\\pic.jpg").convert()
red = (255, 0, 0)
black = (0, 0, 0)
white = (255, 255, 255)

# define global variable
clicked = False
counter = 0

class button():

    # colours for button and text
    button_col = (0,0,0)
    hover_col = (43,40,42)
    click_col = (37, 24, 17)
    text_col = (223,190,217)
    width = 220
    height = 80

    def _init_(self, x, y, text):
        self.x = x
        self.y = y
        self.text = text

    def draw_button(self):

        global clicked
        action = False

        # get mouse position
        pos = pygame.mouse.get_pos()

        # create pygame Rect object for the button
        button_rect = Rect(self.x, self.y, self.width, self.height)

        # check mouseover and clicked conditions
        if button_rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1:
                clicked = True
                pygame.draw.rect(screen, self.click_col, button_rect)
            elif pygame.mouse.get_pressed()[0] == 0 and clicked == True:
                clicked = False
                action = True
            else:
                pygame.draw.rect(screen, self.hover_col, button_rect)
        else:
            pygame.draw.rect(screen, self.button_col, button_rect)

        # add shading to button
        pygame.draw.line(screen, white, (self.x, self.y),
                         (self.x + self.width, self.y), 2)
        pygame.draw.line(screen, white, (self.x, self.y),
                         (self.x, self.y + self.height), 2)
        pygame.draw.line(screen, black, (self.x, self.y + self.height),
                         (self.x + self.width, self.y + self.height), 2)
        pygame.draw.line(screen, black, (self.x + self.width, self.y),
                         (self.x + self.width, self.y + self.height), 2)

        # add text to button
        text_img = font.render(self.text, True, self.text_col)
        text_len = text_img.get_width()
        screen.blit(text_img, (self.x + int(self.width / 2) -
                    int(text_len / 2), self.y + 25))
        return action


    again = button(55, 200, 'Squid Game')   
    quit = button(325, 200, 'Subway Surfers') #325, 200, 'Subway Surfers'
    down = button(185, 340, 'Asphalt 9') #185, 340, 'Asphalt 9'
#up = button(325, 350, 'Up')


run = True
while run:

    screen.fill((0 , 0, 0))
    screen.blit(bg,(0,0))

    if again.draw_button():
        mixer.music.stop()
        cap = cv2.VideoCapture(0)
        cPos = 0
        startT = 0
        endT = 0
        userSum = 0
        dur = 0
        isAlive = 1
        isInit = False
        cStart, cEnd = 0, 0
        isCinit = False
        tempSum = 0
        winner = 0
        inFrame = 0
        inFramecheck = False
        thresh = 180

        def calc_sum(landmarkList):

            tsum = 0
            for i in range(11, 33):
                tsum += (landmarkList[i].x * 480)

            return tsum

        def calc_dist(landmarkList):
            return (landmarkList[28].y*640 - landmarkList[24].y*640)

        def isVisible(landmarkList):
            if (landmarkList[28].visibility > 0.7) and (landmarkList[24].visibility > 0.7):
                return True
            return False

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        drawing = mp.solutions.drawing_utils

        im1 = cv2.imread('greenlight.png')
        im2 = cv2.imread('redlight.png')

        currWindow = im1

        while True:

            _, frm = cap.read()
            rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            frm = cv2.blur(frm, (5, 5))
            drawing.draw_landmarks(frm, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if not(inFramecheck):
                try:
                    if isVisible(res.pose_landmarks.landmark):
                        inFrame = 1
                        inFramecheck = True
                    else:
                        inFrame = 0
                except:
                    print("You are not visible at all")

            if inFrame == 1:
                if not(isInit):
                    playsound('greenLight.mp3')
                    currWindow = im1
                    startT = time.time()
                    endT = startT
                    dur = np.random.randint(1, 5)
                    isInit = True

                    if (endT - startT) <= dur:
                        try:
                            m = calc_dist(res.pose_landmarks.landmark)
                            if m < thresh:
                                cPos += 1

                            print("current progress is : ", cPos)
                        except:
                            print("Not visible")

                        endT = time.time()

                    else:

                        if cPos >= 100:
                            print("WINNER")
                            winner = 1

                        else:
                            if not(isCinit):
                                isCinit = True
                                cStart = time.time()
                                cEnd = cStart
                                currWindow = im2
                                playsound('redLight.mp3')
                                userSum = calc_sum(res.pose_landmarks.landmark)

                            if (cEnd - cStart) <= 3:
                                tempSum = calc_sum(res.pose_landmarks.landmark)
                                cEnd = time.time()
                                if abs(tempSum - userSum) > 150:
                                    print("DEAD ", abs(tempSum - userSum))
                                    isAlive = 0

                            else:
                                isInit = False
                                isCinit = False

                        cv2.circle(currWindow, ((55 + 6*cPos), 280),
                                15, (0, 0, 255), -1)

                        mainWin = np.concatenate(
                            (cv2.resize(frm, (800, 400)), currWindow), axis=0)
                        cv2.imshow("Main Window", mainWin)
                        #cv2.imshow("window", frm)
                        #cv2.imshow("light", currWindow)

            else:
                cv2.putText(frm, "Please Make sure you are fully in frame",
                            (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 4)
                cv2.imshow("window", frm)

            if cv2.waitKey(1) == 27 or isAlive == 0 or winner == 1:
                cv2.destroyAllWindows()
                cap.release()
                break

        frm = cv2.blur(frm, (5, 5))

        if isAlive == 0:
            cv2.putText(frm, "You are Dead", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.imshow("Main Window", frm)

        if winner == 1:
            cv2.putText(frm, "You are Winner", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            cv2.imshow("Main Window", frm)

    cv2.waitKey(0)

    if quit.draw_button():
        mixer.music.stop()
        webbrowser.open(r"https://poki.com/en/g/subway-surfers")

      

        mp_pose = mp.solutions.pose

        pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

        pose_video = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7,
                                min_tracking_confidence=0.7)

        mp_drawing = mp.solutions.drawing_utils 

        def detectPose(image, pose, draw=False, display=False):
        
            output_image = image.copy()
            
            
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            
            results = pose.process(imageRGB)
            
            
            if results.pose_landmarks and draw:
    
       
                mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                        connections=mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),
                                                                                    thickness=3, circle_radius=3),
                                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237),
                                                                                    thickness=2, circle_radius=2))

    
            if display:
            
       
                plt.figure(figsize=[22,22])
                plt.subplot(121);
                plt.imshow(image[:,:,::-1]);
                plt.title("Original Image");
                plt.axis('off');
                plt.subplot(122);
                plt.imshow(output_image[:,:,::-1]);
                plt.title("Output Image");
                plt.axis('off');
        
  
            else:

       
                return output_image, results

        def checkHandsJoined(image, results, draw=False, display=False):
    
            height, width, _ = image.shape
            
        
            output_image = image.copy()
            
        
            left_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height)

        
            right_wrist_landmark = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)
            
            
            euclidean_distance = int(hypot(left_wrist_landmark[0] - right_wrist_landmark[0],
                                        left_wrist_landmark[1] - right_wrist_landmark[1]))
            
            
            if euclidean_distance < 130:
        
        
                hand_status = 'Hands Joined'
                color = (0, 255, 0)
                
            
            else:
                
            
                hand_status = 'Hands Not Joined'
                
            
                color = (0, 0, 255)
        
    
            if draw:

                
                cv2.putText(output_image, hand_status, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
                
                
                cv2.putText(output_image, f'Distance: {euclidean_distance}', (10, 70),
                            cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        
   
            if display:

                
                plt.figure(figsize=[10,10])
                plt.imshow(output_image[:,:,::-1]);
                plt.title("Output Image");
                plt.axis('off');
            
    
            else:
            
                
                return output_image, hand_status

        def checkLeftRight(image, results, draw=False, display=False):
            
            horizontal_position = None
            
            
            height, width, _ = image.shape
            
            
            output_image = image.copy()
            
            
            left_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)

            
            right_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width)
            
            if (right_x <= width//2 and left_x <= width//2):
                
            
                horizontal_position = 'Left'

            elif (right_x >= width//2 and left_x >= width//2):
                
                horizontal_position = 'Right'
            
            elif (right_x >= width//2 and left_x <= width//2):
                
            
                horizontal_position = 'Center'
                
        
            if draw:

                
                cv2.putText(output_image, horizontal_position, (5, height - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
                
            
                cv2.line(output_image, (width//2, 0), (width//2, height), (255, 255, 255), 2)
                
        
            if display:

            
                plt.figure(figsize=[10,10])
                plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
            
        
            else:
            
                return output_image, horizontal_position

        def checkJumpCrouch(image, results, MID_Y=250, draw=False, display=False):
        
            height, width, _ = image.shape
            
            
            output_image = image.copy()
            
            left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)

            right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)

            actual_mid_y = abs(right_y + left_y) // 2
            
            lower_bound = MID_Y-15
            upper_bound = MID_Y+100
            
            if (actual_mid_y < lower_bound):
                
                posture = 'Jumping'
            
            
            elif (actual_mid_y > upper_bound):
            
                posture = 'Crouching'
            
            else:
                
                
                posture = 'Standing'
                
            
            if draw:

                cv2.putText(output_image, posture, (5, height - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
                
                cv2.line(output_image, (0, MID_Y),(width, MID_Y),(255, 255, 255), 2)
                
            if display:


                plt.figure(figsize=[10,10])
                plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
            
            else:
            
                return output_image, posture

        camera_video = cv2.VideoCapture(0)
        camera_video.set(3,1280)
        camera_video.set(4,960)

        cv2.namedWindow('Subway Surfers with Pose Detection', cv2.WINDOW_NORMAL)
        

        time1 = 0


        game_started = False   

        x_pos_index = 1

        y_pos_index = 1

        MID_Y = None


        counter = 0

        num_of_frames = 10

        while camera_video.isOpened():
            
            
            ok, frame = camera_video.read()
            
            if not ok:
                continue
            
            frame = cv2.flip(frame, 1)
            
            frame_height, frame_width, _ = frame.shape
            
            frame, results = detectPose(frame, pose_video, draw=game_started)
            
            if results.pose_landmarks:
            
                if game_started:
                    
                    
                    frame, horizontal_position = checkLeftRight(frame, results, draw=True)
                    
                    if (horizontal_position=='Left' and x_pos_index!=0) or (horizontal_position=='Center' and x_pos_index==2):
                        
                        
                        pyautogui.press('left')
                        
                        x_pos_index -= 1               

                
                    elif (horizontal_position=='Right' and x_pos_index!=2) or (horizontal_position=='Center' and x_pos_index==0):
                        
                    
                        pyautogui.press('right')
                        
                        x_pos_index += 1
                    
            
                else:

                    cv2.putText(frame, 'JOIN BOTH HANDS TO START THE GAME.', (5, frame_height - 10), cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 255, 0), 3)
            
                if checkHandsJoined(frame, results)[1] == 'Hands Joined':

                    
                    counter += 1

                    
                    if counter == num_of_frames:

                    
                        if not(game_started):

                        
                            game_started = True

                            
                            left_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height)

                            
                            right_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height)

                            MID_Y = abs(right_y + left_y) // 2

                            pyautogui.click(x=1300, y=800, button='left')
                        
                        else:

                            pyautogui.press('space')
                        
                
                        counter = 0
            
                else:

                    counter = 0
                    
                if MID_Y:
                    
                    frame, posture = checkJumpCrouch(frame, results, MID_Y, draw=True)
                    
                    
                    if posture == 'Jumping' and y_pos_index == 1:

                        
                        pyautogui.press('up')
                        
                    
                        y_pos_index += 1 

                    
                    elif posture == 'Crouching' and y_pos_index == 1:

                        
                        pyautogui.press('down')
                        
                    
                        y_pos_index -= 1
                    
                    
                    elif posture == 'Standing' and y_pos_index   != 1:
                        
                    
                        y_pos_index = 1
                    
            else:

                counter = 0
        
            time2 = time()
            
            if (time2 - time1) > 0:
            
                frames_per_second = 1.0 / (time2 - time1)
                
                cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
            
            time1 = time2
            
            cv2.imshow('Subway Surfers with Pose Detection', frame)
            
            
            k = cv2.waitKey(1) & 0xFF    
            
            
            if(k == 27):
                break

                    
        camera_video.release()
        cv2.destroyAllWindows()

    if down.draw_button():                                                                                                                                                                                                                                                       
        mixer.music.stop()
        

        cam = VideoStream(src=0).start()
        currentKey = list()

        while True:

            key = False

            img = cam.read()
            img = np.flip(img, axis=1)
            img = imutils.resize(img, width=640)
            img = imutils.resize(img, height=480)

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            value = (11, 11)
            blurred = cv2.GaussianBlur(hsv, value, 0)
            colourLower = np.array([49, 47, 118])
            colourUpper = np.array([180, 255, 255])

            height = img.shape[0]
            width = img.shape[1]

            mask = cv2.inRange(blurred, colourLower, colourUpper)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            upContour = mask[0:height//2, 0:width]
            downContour = mask[3*height//4:height, 2*width//5:3*width//5]

            cnts_up = cv2.findContours(
                upContour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts_up = imutils.grab_contours(cnts_up)

            cnts_down = cv2.findContours(
                downContour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts_down = imutils.grab_contours(cnts_down)

            if len(cnts_up) > 0:

                c = max(cnts_up, key=cv2.contourArea)
                M = cv2.moments(c)
                cX = int(M["m10"]/(M["m00"]+0.000001))

                if cX < (width//2 - 35):
                    PressKey(A)
                    key = True
                    currentKey.append(A)
                elif cX > (width//2 + 35):
                    PressKey(D)
                    key = True
                    currentKey.append(D)

            if len(cnts_down) > 0:
                PressKey(Space)
                key = True
                currentKey.append(Space)

            img = cv2.rectangle(
                img, (0, 0), (width//2 - 35, height//2), (0, 255, 0), 1)
            cv2.putText(img, 'LEFT', (110, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (139, 0, 0))

            img = cv2.rectangle(img, (width//2 + 35, 0),
                                (width-2, height//2), (0, 255, 0), 1)
            cv2.putText(img, 'RIGHT', (440, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (139, 0, 0))

            img = cv2.rectangle(img, (2*(width//5), 3*(height//4)),
                                (3*width//5, height), (0, 255, 0), 1)
            cv2.putText(img, 'NITRO', (2*(width//5) + 20, height-10),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (139, 0, 0))

            cv2.imshow("Steering", img)

            if not key and len(currentKey) != 0:
                for current in currentKey:
                    ReleaseKey(current)
                    currentKey = list()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    pygame.display.update()


pygame.quit()