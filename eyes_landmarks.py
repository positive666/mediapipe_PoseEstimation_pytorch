"""
Miscellaneous facial features detection implementation
"""

import cv2
import numpy as np
from enum import Enum

class Eyes(Enum):
    LEFT = 1
    RIGHT = 2

class FacialFeatures:

    eye_key_indicies=[
        [
        # Left eye
        # eye lower contour
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        # eye upper contour (excluding corners)
        246,
        161,
        160,
        159,
        158,
        157,
        173
        ],
        [
        # Right eye
        # eye lower contour
        263,
        249,
        390,
        373,
        374,
        380,
        381,
        382,
        362,
        # eye upper contour (excluding corners)
        466,
        388,
        387,
        386,
        385,
        384,
        398
        ]
    ]

    # custom img resize function
    def resize_img(img, scale_percent):
        width = int(img.shape[1] * scale_percent / 100.0)
        height = int(img.shape[0] * scale_percent / 100.0)

        return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

    # calculate eye apsect ratio to detect blinking
    # and/ or control closing/ opening of eye
    def eye_aspect_ratio(image_points, side):

        p1, p2, p3, p4, p5, p6 = 0, 0, 0, 0, 0, 0
        tip_of_eyebrow = 0

        # get the contour points at img pixel first
        # following the eye aspect ratio formula with little modifications
        # to match the facemesh model
        if side == Eyes.LEFT:

            eye_key_left = FacialFeatures.eye_key_indicies[0]

            p2 = np.true_divide(
                np.sum([image_points[eye_key_left[10]], image_points[eye_key_left[11]]], axis=0),
                2)
            p3 = np.true_divide(
                np.sum([image_points[eye_key_left[13]], image_points[eye_key_left[14]]], axis=0),
                2)
            p6 = np.true_divide(
                np.sum([image_points[eye_key_left[2]], image_points[eye_key_left[3]]], axis=0),
                2)
            p5 = np.true_divide(
                np.sum([image_points[eye_key_left[5]], image_points[eye_key_left[6]]], axis=0),
                2)
            p1 = image_points[eye_key_left[0]]
            p4 = image_points[eye_key_left[8]]

            # tip_of_eyebrow = image_points[63]
            tip_of_eyebrow = image_points[105]

        elif side == Eyes.RIGHT:
            eye_key_right = FacialFeatures.eye_key_indicies[1]

            p3 = np.true_divide(
                np.sum([image_points[eye_key_right[10]], image_points[eye_key_right[11]]], axis=0),
                2)
            p2 = np.true_divide(
                np.sum([image_points[eye_key_right[13]], image_points[eye_key_right[14]]], axis=0),
                2)
            p5 = np.true_divide(
                np.sum([image_points[eye_key_right[2]], image_points[eye_key_right[3]]], axis=0),
                2)
            p6 = np.true_divide(
                np.sum([image_points[eye_key_right[5]], image_points[eye_key_right[6]]], axis=0),
                2)
            p1 = image_points[eye_key_right[8]]
            p4 = image_points[eye_key_right[0]]

            tip_of_eyebrow = image_points[334]

        # https://downloads.hindawi.com/journals/cmmm/2020/1038906.pdf
        # Fig (3)
        ear = np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)
        ear /= (2 * np.linalg.norm(p1-p4) + 1e-6)
        ear = ear * (np.linalg.norm(tip_of_eyebrow-image_points[2]) / np.linalg.norm(image_points[6]-image_points[2]))
        return ear

    # calculate mouth aspect ratio to detect mouth movement
    # to control opening/ closing of mouth in avatar
    # https://miro.medium.com/max/1508/0*0rVqugQAUafxXYXE.jpg
    def mouth_aspect_ratio(image_points):
        p1 = image_points[78]
        p2 = image_points[81]
        p3 = image_points[13]
        p4 = image_points[311]
        p5 = image_points[308]
        p6 = image_points[402]
        p7 = image_points[14]
        p8 = image_points[178]

        mar = np.linalg.norm(p2-p8) + np.linalg.norm(p3-p7) + np.linalg.norm(p4-p6)
        mar /= (2 * np.linalg.norm(p1-p5) + 1e-6)
        return mar

    def mouth_distance(image_points):
        p1 = image_points[78]
        p5 = image_points[308]
        return np.linalg.norm(p1-p5)


    # detect iris through new landmark coordinates produced by mediapipe
    # replacing the old image processing method
    def detect_iris(image_points, iris_image_points, side):
        '''
            return:
                x_rate: how much the iris is toward the left. 0 means totally left and 1 is totally right.
                y_rate: how much the iris is toward the top. 0 means totally top and 1 is totally bottom.
        '''

        iris_img_point = -1
        p1, p4 = 0, 0
        eye_y_high, eye_y_low = 0, 0
        x_rate, y_rate = 0.5, 0.5

        # get the corresponding image coordinates of the landmarks
        if side == Eyes.LEFT:
            iris_img_point = 468

            eye_key_left = FacialFeatures.eye_key_indicies[0]
            p1 = image_points[eye_key_left[0]]
            p4 = image_points[eye_key_left[8]]

            eye_y_high = image_points[eye_key_left[12]]
            eye_y_low = image_points[eye_key_left[4]]

        elif side == Eyes.RIGHT:
            iris_img_point = 473

            eye_key_right = FacialFeatures.eye_key_indicies[1]
            p1 = image_points[eye_key_right[8]]
            p4 = image_points[eye_key_right[0]]

            eye_y_high = image_points[eye_key_right[12]]
            eye_y_low = image_points[eye_key_right[4]]

        p_iris = iris_image_points[iris_img_point - 468]

        # find the projection of iris_image_point on the straight line fromed by p1 and p4
        # through vector dot product
        # to get x_rate

        vec_p1_iris = [p_iris[0] - p1[0], p_iris[1] - p1[1]]
        vec_p1_p4 = [p4[0] - p1[0], p4[1] - p1[1]]
        
        x_rate = (np.dot(vec_p1_iris, vec_p1_p4) / (np.linalg.norm(p1-p4) + 1e-06)) / (np.linalg.norm(p1-p4) + 1e-06)

        # find y-rate simiilarily

        vec_eye_h_iris = [p_iris[0] - eye_y_high[0], p_iris[1] - eye_y_high[1]]
        vec_eye_h_eye_l = [eye_y_low[0] - eye_y_high[0], eye_y_low[1] - eye_y_high[1]]

        y_rate = (np.dot(vec_eye_h_eye_l, vec_eye_h_iris) / (np.linalg.norm(eye_y_high - eye_y_low) + 1e-06)) / (np.linalg.norm(eye_y_high - eye_y_low) + 1e-06)

        return x_rate, y_rate
    
def draw_debug_image(
    image,
    left_iris,
    right_iris,
    left_center,
    left_radius,
    right_center,
    right_radius,
):
    
    cv2.circle(image, left_center, left_radius, (0, 255, 0), 2)
    cv2.circle(image, right_center, right_radius, (0, 255, 0), 2)

    # 
    for point in left_iris:
        cv2.circle(image, (point[0], point[1]), 1, (0, 0, 255), 2)
    for point in right_iris:
        cv2.circle(image, (point[0], point[1]), 1, (0, 0, 255), 2)

    # show 
    cv2.putText(image, 'r:' + str(left_radius) + 'px',
               (left_center[0] + int(left_radius * 1.5),
                left_center[1] + int(left_radius * 0.5)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    cv2.putText(image, 'r:' + str(right_radius) + 'px',
               (right_center[0] + int(right_radius * 1.5),
                right_center[1] + int(right_radius * 0.5)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    #return image
def draw_iris(im, iris, eye_contour):
   
   
    #im = pad_image(im, desired_size=64)
   
    lm = iris[0]*3
    h, w, _ = im.shape

    cv2.circle(im, (int(lm[0]), int(lm[1])), 2, (0, 255, 0), -1)
    cv2.circle(im, (int(lm[3]), int(lm[4])), 1, (255, 0, 255), -1)
    cv2.circle(im, (int(lm[6]), int(lm[7])), 2, (255, 0, 255), -1)
    cv2.circle(im, (int(lm[9] ), int(lm[10])), 1, (255, 0, 255), -1)
    cv2.circle(im, (int(lm[12] ), int(lm[13])), 1, (255, 0, 255), -1)

    eye_contour
    for idx in range(71):
        cv2.circle(im, (int(eye_contour[0][idx*3]), int(eye_contour[0][idx*3 + 1])), 1, (0, 0, 255), -1)
    #v2.imshow('eye',im)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #return im 
    
def calc_iris_point(eye_bbox, eye_contour, iris, input_shape):
    iris_list = []
    for index in range(5):
        point_x = int(iris[index * 3] *
                      ((eye_bbox[2] - eye_bbox[0]) / input_shape[0]))
        point_y = int(iris[index * 3 + 1] *
                      ((eye_bbox[3] - eye_bbox[1]) / input_shape[1]))
        point_x += eye_bbox[0]
        point_y += eye_bbox[1]

        iris_list.append((point_x, point_y))

    return iris_list

def calc_min_enc_losingCircle(landmark_list):
    center, radius = cv2.minEnclosingCircle(np.array(landmark_list))
    center = (int(center[0]), int(center[1]))
    radius = int(radius)

    return center, radius

def detectx_iris(image, iris_detector, left_eye, right_eye):
    image_width, image_height = image.shape[1], image.shape[0]
    input_shape = (64,64)

    # left eye
    # roi image
    left_eye_x1 = max(left_eye[0], 0)
    left_eye_y1 = max(left_eye[1], 0)
    left_eye_x2 = min(left_eye[2], image_width)
    left_eye_y2 = min(left_eye[3], image_height)
    left_eye_image = copy.deepcopy(image[left_eye_y1:left_eye_y2,
                                         left_eye_x1:left_eye_x2])
    # detect iris
    eye_roi = cv2.resize(left_eye_image, input_shape)          
    eye_contour, iris = iris_detector.predict(eye_roi )
    
     #   local landmarks------- gobal landmarks
    left_iris = calc_iris_point(left_eye, np.squeeze(eye_contour), np.squeeze(iris), input_shape)

    #  right eye
    #  roi image
    right_eye_x1 = max(right_eye[0], 0)
    right_eye_y1 = max(right_eye[1], 0)
    right_eye_x2 = min(right_eye[2], image_width)
    right_eye_y2 = min(right_eye[3], image_height)
    right_eye_image = copy.deepcopy(image[right_eye_y1:right_eye_y2,
                                          right_eye_x1:right_eye_x2])
    #   detect iris
    eye_roi = cv2.resize(right_eye_image, input_shape)
    eye_contour, iris = iris_detector.predict(eye_roi)
    #   local landmarks------- gobal landmarks
    right_iris = calc_iris_point(right_eye, np.squeeze(eye_contour), np.squeeze(iris), input_shape)

    return left_iris, right_iris    