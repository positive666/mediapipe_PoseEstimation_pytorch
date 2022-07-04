import cv2
import mediapipe as mp
import json
from scipy.signal import savgol_filter
import time
# as requested in comment

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def RetriveXYZ(landmarkslist, index):
    origin = landmarkslist[0]
    landmark = landmarkslist[index]
    x,y,z = (landmark.x,landmark.y,landmark.z)
    x = x-origin.x
    y = origin.y-y
    roundto = 9
    x,y,z = (round(x, roundto), round(y, roundto), round(z, roundto))
    return x,y,z

def nfilter(smoothness, array_input):
	init = array_input[0]
	array_output = [0.0] * len(array_input)
	for i in range(len(array_input)):
		array_output[i] = smoothness * array_input[i] + (1-smoothness)*(init)
		init = array_output[i]
	return array_output
    
def smoothen_tracking(timeline, smoothing = 0.4):
    timeline_length = len(timeline)
    formatted_timeline = {}
    for key, item in timeline[0].items():
        formatted_timeline[key] = [[],[],[]]

        for index in range(timeline_length):
            for axis_index in range(3):

                values = timeline[index][key][axis_index]
                formatted_timeline[key][axis_index].append(values)
            

    filtered_timeline = {}
    for key, item in formatted_timeline.items():
        filtered_timeline[key] = [[],[],[]]

        for axis_index in range(3):
            filtered_values = nfilter(smoothing, formatted_timeline[key][axis_index])
            filtered_timeline[key][axis_index] = filtered_values

    new_timeline = []
    for frame_index in range(timeline_length):
        pose = {}
        for key, item in filtered_timeline.items():
            pose[key] = [filtered_timeline[key][0][frame_index], filtered_timeline[key][1][frame_index], filtered_timeline[key][2][frame_index]]
        new_timeline.append(pose)
    return new_timeline

# For webcam input:
cap = cv2.VideoCapture('tesla.mp4')

timeline = []

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
raw_video = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))
processed_video = cv2.VideoWriter('output_processed.mp4',fourcc, 20.0, (1920,1080))

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.8, max_num_hands=1, static_image_mode = False) as hands:
    while cap.isOpened():
        t0=time.time()
        success, image = cap.read()
        image = cv2.flip(image, 1)
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        raw_video.write(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                shape = image.shape 
                image = cv2.rectangle(image, (0, 0), (shape[1], shape[0]), (255,255,255), -1)

                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                HandData = {
                        'Wrist': RetriveXYZ(hand_landmarks.landmark, 0),
                        'Pinky': RetriveXYZ(hand_landmarks.landmark, 20),
                        'Ring': RetriveXYZ(hand_landmarks.landmark, 16),
                        'Middle': RetriveXYZ(hand_landmarks.landmark, 12),
                        'Index': RetriveXYZ(hand_landmarks.landmark, 8),
                        'Thumb': RetriveXYZ(hand_landmarks.landmark, 4),
                        'PinkyBase': RetriveXYZ(hand_landmarks.landmark, 17),
                        'IndexBase': RetriveXYZ(hand_landmarks.landmark, 5),
                        'MiddleBase': RetriveXYZ(hand_landmarks.landmark, 9),
                        'ThumbBase': RetriveXYZ(hand_landmarks.landmark, 2),
                        'RingBase': RetriveXYZ(hand_landmarks.landmark, 13),
                    }

                for index, landmark in enumerate(hand_landmarks.landmark):
                    x = landmark.x
                    y = landmark.y
                    z = landmark.y

                    relative_x = int(x * shape[1])
                    relative_y = int(y * shape[0])

                    cv2.putText(image,str((z)), (relative_x,relative_y), 0, 0.5, 255)

                break

            print(HandData)
            timeline.append(HandData)
            print("per fps:",1/(time.time()-t0))
            processed_video.write(image)
            #cv2.write()
        cv2.imshow("Blender Hand Tracking", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
                break


timeline = smoothen_tracking(timeline)        

with open('HandData.txt', 'w') as file:
    file.write(json.dumps(timeline)) 


cap.release()
