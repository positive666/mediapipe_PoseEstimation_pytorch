#   human pose estimation project by  positive6666, 2022/6
#   github
#
from pickle import NONE
import numpy as np
import torch
import cv2
import sys
import time
import copy
from blazebase import resize_pad, denormalize_detections,BlazeLandmark,PoseEstimator
from blazeface import BlazeFace
from blazepalm import BlazePalm
from blazepose import BlazePose
from blazeface_landmark import BlazeFaceLandmark
from blazehand_landmark import BlazeHandLandmark
from visualization import *
from torchvision import transforms
from eyes_landmarks import *
from kalman import *
import socket
from argparse import ArgumentParser

#import cupy as cp
from blazeiris import IrisLM
from blazepose_landmark import BlazePoseLandmark
#from cupy._core.dlpack import toDlpack
#from cupy._core.dlpack import fromDlpack
#from torch.utils.dlpack import to_dlpack
#from torch.utils.dlpack import from_dlpack

#  init TCP connection with unity
# return the socket connected
# global variable

port = 5066         # have to be same as unity
def init_TCP():
    port = args.port

    # 'localhost' = your IP
    address = ('10.11.0.181', port)

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(address)
        # print(socket.gethostbyname(socket.gethostname()) + "::" + str(port))
        print("Connected to address:", socket.gethostbyname(socket.gethostname()) + ":" + str(port))
        return s
    except OSError as e:
        print("Error while connecting :: %s" % e)
        
        # quit the script if connection fails (e.g. Unity server side quits suddenly)
        sys.exit()

def send_info_to_unity(s, args):
    msg = '%.4f ' * len(args) % args

    try:
        s.send(bytes(msg, "utf-8"))
    except socket.error as e:
        print("error while sending :: " + str(e))
        # quit the script if connection fails (e.g. Unity server side quits suddenly)
        sys.exit()

def print_debug_msg(args):
    msg = '%.4f ' * len(args) % args
    print(msg)
    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # # print(socket.gethostbyname(socket.gethostname()))
    # s.connect(address)
    # return s


    
def pad_image(im, desired_size=64):
    
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
    return new_im

def pose_detect(frame,pose_struct,pose_stable_lists,draw=True):
   
    frame = np.ascontiguousarray(frame[:,::-1,::-1])
    img1, img2, scale, pad = resize_pad(frame)

    normalized_pose_detections = pose_struct['pose_detector'].predict_on_image(img2)
    pose_detections = denormalize_detections(normalized_pose_detections, scale, pad)

    xc, yc, scale, theta = pose_struct['pose_detector'].detection2roi(pose_detections.cpu())
    img, affine, box = pose_struct['pose_regressor'].extract_roi(frame, xc, yc, theta, scale)
    flags, normalized_landmarks, mask = pose_struct['pose_regressor'](img.to(pose_struct['gpu']))
    landmarks = pose_struct['pose_regressor'].denormalize_landmarks(normalized_landmarks.cpu(), affine)
    if draw:
        draw_detections(frame, pose_detections)
        draw_roi(frame, box)
        
        for i in range(len(flags)):
            landmark, flag = landmarks[i], flags[i]
            if flag>.5:
                draw_landmarks(frame, landmark[:,:2], POSE_CONNECTIONS, size=2)
    return frame
    
def palm_detect(frame,pose_struct,pose_stable_lists,draw=True):
    frame = np.ascontiguousarray(frame[:,:,::-1])
    img1, img2, scale, pad = resize_pad(frame)
    normalized_palm_detections = pose_struct['palm_detector'].predict_on_image(img1)
  
    palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)
      
    xc, yc, scale, theta = pose_struct['palm_detector'].detection2roi(palm_detections.cpu())
    img, affine, box = pose_struct['hand_regressor'].extract_roi(frame, xc, yc, theta, scale)
    flags, handed2, normalized_landmarks=pose_struct['hand_regressor'](img.to(pose_struct['gpu']))
    landmarks =pose_struct['hand_regressor'].denormalize_landmarks(normalized_landmarks.cpu(), affine)
    if draw:
        for i in range(len(flags)):
                    landmark, flag = landmarks[i], flags[i]
                    if flag>.5:
                        draw_landmarks(frame, landmark[:,:2], HAND_CONNECTIONS, size=2)
        #draw_detections(frame, palm_detections)
        #draw_roi(frame, box)    
        #cv2.imshow('palm',frame[:,:,::-1])
        #key = cv2.waitKey(10)
    
    return frame
          
def face_detect(frame,pose_struct,pose_stable_lists,draw=True):

    head_pose_reuslt=[]
 
    img1, img2, scale, pad = resize_pad(frame)
    normalized_face_detections = pose_struct['face_detector'].predict_on_image(img2)
                
           
    face_detections = denormalize_detections(normalized_face_detections, scale, pad)
     
    xc, yc, scale, theta = pose_struct['face_detector'].detection2roi(face_detections.cpu())
  
    img, affine, box = pose_struct['face_regressor'].extract_roi(frame, xc, yc, theta, scale)
    flags, normalized_landmarks = pose_struct['face_regressor'](img.to(pose_struct['gpu']))
    landmarks = pose_struct['face_regressor'].denormalize_landmarks(normalized_landmarks.cpu(), affine)
    if draw:
            draw_roi(frame, box)
            
    for i in range(len(flags)):
                landmark, flag = landmarks[i], flags[i]
            
                if flag>.5:
                    
                    for i in range(len(pose_struct['image_points'])):
                        
                        pose_struct['image_points'][i, 0] = landmark[:,:2][i][0]
                        pose_struct['image_points'][i, 1] = landmark[:,:2][i][1]
                       
                    if args.kalaman_driver:  
                        for j in range(len(pose_struct['iris_image_points'])):
                            pose_struct['iris_image_points'][j, 0] = landmark[:,:2][j][0]
                            pose_struct['iris_image_points'][j, 1] = landmark[:,:2][j][1]
                    
                        # pose: [[rvec], [tvec]]
                        pose = pose_struct['pose_estimator'].solve_pose_by_all_points(pose_struct['image_points'])
                
                    
                        x_ratio_left,  y_ratio_left  = FacialFeatures.detect_iris(pose_struct['image_points'], pose_struct['iris_image_points'], Eyes.LEFT)
                        x_ratio_right, y_ratio_right = FacialFeatures.detect_iris(pose_struct['image_points'], pose_struct['iris_image_points'], Eyes.RIGHT)
                        print(f'INFO__eyes xro:{x_ratio_right},yrota{y_ratio_right}')
                        
                        ear_left = FacialFeatures.eye_aspect_ratio(pose_struct['image_points'], Eyes.LEFT)
                        ear_right = FacialFeatures.eye_aspect_ratio(pose_struct['image_points'], Eyes.RIGHT)
                        #draw_iris(eye_roi, image_points, Eyes.LEFT）
                        
                        pose_eye = [ear_left, ear_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right]
                        #get mouth
                        mar = FacialFeatures.mouth_aspect_ratio(pose_struct['image_points'])
                        mouth_distance = FacialFeatures.mouth_distance(pose_struct['image_points'])

                        print("mouth marign:",mar)
                        print("mouth_distance",mouth_distance)
                        
                        # Stabilize the pose.
                        steady_pose = []
                        pose_np = np.array(pose).flatten()

                        for value, ps_stb in zip(pose_np,pose_stable_lists['pose_stabilizers']):
                            ps_stb.update([value])
                            steady_pose.append(ps_stb.state[0])

                        steady_pose = np.reshape(steady_pose, (-1, 3))

                        # stabilize the eyes value
                        steady_pose_eye = []
                        for value, ps_stb in zip(pose_eye, (pose_stable_lists['eyes_stabilizers'])):
                            ps_stb.update([value])
                            steady_pose_eye.append(ps_stb.state[0])

                        pose_stable_lists['mouth_dist_stabilizers'].update([mouth_distance])
                        steady_mouth_dist = pose_stable_lists['mouth_dist_stabilizers'].state[0]  

                        # calculate the roll/ pitch/ yaw
                        # roll: +ve when the axis pointing upward
                        # pitch: +ve when we look upward
                        # yaw: +ve when we look left
                        roll = np.clip(np.degrees(steady_pose[0][1]), -90, 90)
                        pitch = np.clip(-(180 + np.degrees(steady_pose[0][0])), -90, 90)
                        yaw =  np.clip(np.degrees(steady_pose[0][2]), -90, 90)   
                        head_pose_reuslt=(roll, pitch, yaw,
                             ear_left, ear_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right,
                              mar, mouth_distance)
                    
                    # IRIs
                    if args.detect_iris:
                        print("虹膜识别---extract iris")
                        left_eye, right_eye =pose_struct['iris_regressor'].calc_around_eye_bbox(pose_struct['image_points'])
                        # detect iris
                        left_iris, right_iris = detectx_iris(frame,pose_struct['iris_landmark'], left_eye,
                                                    right_eye)
                        #print('left_iris:',left_iris)
                        left_center, left_radius = calc_min_enc_losingCircle(left_iris)
                        right_center, right_radius = calc_min_enc_losingCircle(right_iris)
                        draw_debug_image(
                        frame,
                    left_iris,
                    right_iris,
                    left_center,
                    left_radius,
                    right_center,
                    right_radius,
                )
                    if draw:
                        draw_landmarks(frame, landmark[:,:2], FACE_CONNECTIONS, size=1)
                       # draw_landmarks(frame, landmark[:,:2], EYE_left,(0,255,0), size=1)
                       # draw_landmarks(frame, landmark[:,:2], EYE_right, (0,255,0),size=1)
                       # pose_struct['pose_estimator'].draw_annotation_box(img, pose[0], pose[1], color=(255, 128, 128))
                         # pose_estimator.draw_axis(img, pose[0], pose[1])
                        pose_struct['pose_estimator'].draw_axes(frame, steady_pose[0], steady_pose[1])

                    print(f"欧拉角——：roll:{roll},pitch:{pitch},yaw{yaw}")    
                    #rotation_keypoints = [(point[0], point[1]) for point in left_eye_landmarks]
                else:
                    # reset our pose estimator
                    pose_struct['pose_estimator'] = PoseEstimator((1920, 1080))   
  
    return head_pose_reuslt,frame
  
def run(args):
    #set torch cuda env 
    gpu = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
   # if view:
    WINDOW='app'
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    if  args.source :
        print("info:",args.source)
        capture = cv2.VideoCapture(args.source)
        mirror_img = False
    else:
        capture = cv2.VideoCapture(0)
        mirror_img = True

    if capture.isOpened():
        hasFrame, frame = capture.read()
        frame_ct = 0
    else:
        hasFrame = False
  
    #init 
    back_detector=False
    iris_regressor=BlazeLandmark()
    image_points = np.zeros((468, 2))
    pose_estimator = PoseEstimator((int(capture.get(3)), int(capture.get(4))))
    # extra 10 points due to new attention model (in iris detection)
    iris_image_points = np.zeros((10, 2))
    
    #init models list 
    pose_struct={'image_points':image_points,'iris_image_points':iris_image_points,'pose_estimator':pose_estimator,'face_detector':None,
                 'face_regressor':None,'palm_detector': None,"pose_detector":None,"pose_regressor":None,
                 'hand_regressor ':None,'iris_landmark':None,'iris_regressor':iris_regressor,'save_writer':None,
                 'gpu':gpu}
    if args.save_file:
         pose_struct["save_writer"] = cv2.VideoWriter('output_pose.mp4', cv2.VideoWriter_fourcc(*"MJPG"), 
                        25, (int(capture.get(3)), int(capture.get(4))))
    #body pose models
    if args.detect_pose:
        pose_detector = BlazePose().to(gpu)
        pose_detector.load_weights("model_weights\\blazepose.pth")
        pose_detector.load_anchors("model_weights\\anchors_pose.npy")
        pose_regressor = BlazePoseLandmark().to(gpu)
        pose_regressor.load_weights("model_weights\\blazepose_landmark.pth")
        pose_struct['pose_detector']=pose_detector
        pose_struct['pose_regressor']=pose_regressor
        print('load pose done ')
        
    #head detect
    if args.detect_face:
        face_detector = BlazeFace(back_model=back_detector).to(gpu)
        if back_detector:
            face_detector.load_weights("model_weights\\blazefaceback.pth")
            face_detector.load_anchors("model_weights\\anchors_face_back.npy")
        else:
            face_detector.load_weights("model_weights\\blazeface.pth")
            face_detector.load_anchors("model_weights\\anchors_face.npy")
        pose_struct['face_detector']= face_detector
        #load facemesh models
        
        face_regressor = BlazeFaceLandmark().to(gpu)
        face_regressor.load_weights("model_weights\\blazeface_landmark.pth")
        pose_struct['face_regressor']= face_regressor
        if args.detect_iris:
                 #load iris models
                iris_landmark=IrisLM().to(gpu)
                wts = torch.load('model_weights\\irislandmarks.pth')
                iris_landmark.load_state_dict(wts)
                iris_landmark=iris_landmark.eval()
                pose_struct['iris_landmark']= iris_landmark
                print("load iris")
            
    if args.detect_palm:
         #load hand models
        palm_detector = BlazePalm().to(gpu)
        palm_detector.load_weights("model_weights\\blazepalm.pth")
        palm_detector.load_anchors("model_weights\\anchors_palm.npy")
        palm_detector.min_score_thresh = .75
        pose_struct['palm_detector']= palm_detector
        
        hand_regressor = BlazeHandLandmark().to(gpu)
        hand_regressor.load_weights("model_weights\\blazehand_landmark.pth")
        pose_struct['hand_regressor']= hand_regressor
        
   
    else:
        print('choose custom tasks!')
   
    #init kalaman mooth
    pose_stable_lists={'pose_stabilizers': None,'eyes_stabilizers': None,'mouth_dist_stabilizer': None}
    if args.kalaman_driver:
        
        # Introduce scalar stabilizers for pose.
        pose_stable_lists['pose_stabilizers'] = [Stabilizer(
                state_num=2,
                measure_num=1,
                cov_process=0.1,
                cov_measure=0.1) for _ in range(6)]

            # for eyes
        pose_stable_lists['eyes_stabilizers'] = [Stabilizer(
                state_num=2,
                measure_num=1,
                cov_process=0.1,
                cov_measure=0.1) for _ in range(6)]

            # for mouth_dist
        pose_stable_lists['mouth_dist_stabilizers'] = Stabilizer(
                state_num=2,
                measure_num=1,
                cov_process=0.1,
                cov_measure=0.1
            )
        print(f'kalaman filter loads done')
    
    # Initialize TCP connection
    if args.connect:
        socket = init_TCP()
    
    while hasFrame:
            t0=time.time()
            #frame_ct +=1
            #fr=frame.copy()
            #if mirror_img:
                #frame = np.ascontiguousarray(frame[:,::-1,::-1])
            #else:
            
            #frame = np.ascontiguousarray(frame[:,:,::-1])
            if args.detect_face:
                print("run face")
                pose_result,frame=face_detect(frame,pose_struct,pose_stable_lists)
            if  args.detect_palm:
                frame=palm_detect(frame,pose_struct,pose_stable_lists)           
            if  args.detect_pose:
                frame=pose_detect(frame,pose_struct,pose_stable_lists)
            if args.connect:
            # for sending to live2d model (Hiyori)
                send_info_to_unity(socket,pose_result)

            # print the sent values in the terminal
            if args.debug:
                print_debug_msg(pose_result)
            cv2.imshow(WINDOW, frame)
            t1=time.time()
            print(f"conuse time:{t1-t0}")
            hasFrame, frame = capture.read()
            
            key = cv2.waitKey(10)
            if key == 27:
               break
            if not hasFrame:
                print("Ignoring empty camera frame.")
                return {'PASS_THROUGH'}


    capture.release()
    cv2.destroyAllWindows()
 

            
if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--connect", action="store_true",
                        help="connect to unity character",
                        default=False)

    parser.add_argument("--port", type=int, 
                        help="specify the port of the connection to unity. Have to be the same as in Unity", 
                        default=5066)

    parser.add_argument("--source", type=str,
                        help="specify the camera number if you have multiple cameras",
                        default=NONE)
    parser.add_argument("--save_file", action="store_true",
                        help="save filename ",
                        default=False)
    parser.add_argument("--debug", action="store_true",
                        help="showing raw values of detection in the terminal",
                        default=False)
    
    parser.add_argument("--detect_iris", action="store_true",
                        help="showing raw values of detection in the terminal",
                        default=False)
    
    parser.add_argument("--detect_pose", action="store_true",
                        help="showing raw values of detection in the terminal",
                        default=False)


    parser.add_argument("--detect_face", action="store_true",
                        help="showing raw values of detection in the terminal",
                        default=False)
    
    parser.add_argument("--detect_palm", action="store_true",
                        help="showing raw values of detection in the terminal",
                        default=False)
    
    parser.add_argument("--device", type=int,
                        help="showing raw values of detection in the terminal",
                        default="0")
    
    parser.add_argument("--kalaman_driver", action="store_true",
                        help="showing raw values of detection in the terminal",
                        default=True)
    args = parser.parse_args()
    
    # demo code
    run(args)
    