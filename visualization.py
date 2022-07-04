import numpy as np
import cv2
import torch
from enum import IntEnum


def draw_detections(img, detections, with_keypoints=True):
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    n_keypoints = detections.shape[1] // 2 - 2

    for i in range(detections.shape[0]):
        ymin = detections[i, 0]
        xmin = detections[i, 1]
        ymax = detections[i, 2]
        xmax = detections[i, 3]
        
        start_point = (int(xmin), int(ymin))
        end_point = (int(xmax), int(ymax))
        img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 1) 

        if with_keypoints:
            for k in range(n_keypoints):
                kp_x = int(detections[i, 4 + k*2    ])
                kp_y = int(detections[i, 4 + k*2 + 1])
                cv2.circle(img, (kp_x, kp_y), 2, (0, 0, 255), thickness=2)
    return img


def draw_roi(img, roi):
    for i in range(roi.shape[0]):
        (x1,x2,x3,x4), (y1,y2,y3,y4) = roi[i]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,0), 2)
        cv2.line(img, (int(x1), int(y1)), (int(x3), int(y3)), (0,255,0), 2)
        cv2.line(img, (int(x2), int(y2)), (int(x4), int(y4)), (0,0,0), 2)
        cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), (0,0,0), 2)


def draw_landmarks(img, points, connections=[], color=(0, 255, 0), size=2):
   # print("222:",points.shape)
    points = points[:,:2]
    
    for point in points:
        x, y= point
        x, y= int(x), int(y)
        cv2.circle(img, (x, y), size, color, thickness=size)
        #print(x,y,z)
    for connection in connections:
       
        x0, y0= points[connection[0]]
        x1, y1= points[connection[1]]
       
        x0, y0 = int(x0), int(y0)
        x1, y1 = int(x1), int(y1)
       
        cv2.line(img, (x0, y0), (x1, y1), (255,255,255), size)


# ROI scale factor for 25% margin around eye
ROI_SCALE = (2.3, 2.3)
# Landmark index of the left eye start point
LEFT_EYE_START = 33
# Landmark index of the left eye end point
LEFT_EYE_END = 133
# Landmark index of the right eye start point
RIGHT_EYE_START = 362
# Landmark index of the right eye end point
RIGHT_EYE_END = 263
# Number of face landmarks (from face landmark results)
NUM_FACE_LANDMARKS = 468

# Landmark element count (x, y, z)
NUM_DIMS = 3
NUM_EYE_LANDMARKS = 71
NUM_IRIS_LANDMARKS = 5

# eye contour default visualisation settings
# (from iris_and_depth_renderer_cpu.pbtxt)
EYE_LANDMARK_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (5, 6), (6, 7), (7, 8), (9, 10), (10, 11),
    (11, 12), (12, 13), (13, 14), (0, 9), (8, 14)
]
MAX_EYE_LANDMARK = len(EYE_LANDMARK_CONNECTIONS)

EYE_left=[ (33, 7), (7, 163), (163, 144), (144, 145), (145, 153),
    (246,161),(161,160),(160,159),(159,158),(158,157),(157,173),
    (153, 154), (154, 155), (155, 133), (33, 246), 
    (173, 133), 
    
    #(35,124), (124,46), (46,53) ,(53,52), (52,65),
    # halo x5 lower contour
    #(130,25), (25,110),(110,24),(24,23), (23,22), (22,26, (26,112) ,(112,243),
    # halo x5 upper contour excluding corners or eyebrow outer contour 
    #  (226,31), (31,228), (228,229),(229,230), (230,231), (231,232), (232,233), (233,244),
    # (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105),
    # (105, 66), (66, 107)
]
EYE_right=[
(263, 249), (249, 390), (390, 373), (373, 374), (374, 380),
    (380, 381), (381, 382), (382, 362), (263, 466), (466, 388),
    (388, 387), (387, 386), (386, 385), (385, 384), (384, 398),
    (398, 362)
]
# mapping from left eye contour index to face landmark index
LEFT_EYE_TO_FACE_LANDMARK_INDEX = [
    # eye lower contour
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    # eye upper contour excluding corners
    246, 161, 160, 159, 158, 157, 173,
    # halo x2 lower contour
    130, 25, 110, 24, 23, 22, 26, 112, 243,
    # halo x2 upper contour excluding corners
    247, 30, 29, 27, 28, 56, 190,
    # halo x3 lower contour
    226, 31, 228, 229, 230, 231, 232, 233, 244,
    # halo x3 upper contour excluding corners
    113, 225, 224, 223, 222, 221, 189,
    # halo x4 upper contour (no upper due to mesh structure)
    # or eyebrow inner contour
    35, 124, 46, 53, 52, 65,
    # halo x5 lower contour
    143, 111, 117, 118, 119, 120, 121, 128, 245,
    # halo x5 upper contour excluding corners or eyebrow outer contour
    156, 70, 63, 105, 66, 107, 55, 193,
]

# mapping from right eye contour index to face landmark index
RIGHT_EYE_TO_FACE_LANDMARK_INDEX = [
    # eye lower contour
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    # eye upper contour excluding corners
    466, 388, 387, 386, 385, 384, 398,
    # halo x2 lower contour
    359, 255, 339, 254, 253, 252, 256, 341, 463,
    # halo x2 upper contour excluding corners
    467, 260, 259, 257, 258, 286, 414,
    # halo x3 lower contour
    446, 261, 448, 449, 450, 451, 452, 453, 464,
    # halo x3 upper contour excluding corners
    342, 445, 444, 443, 442, 441, 413,
    # halo x4 upper contour (no upper due to mesh structure)
    # or eyebrow inner contour
    265, 353, 276, 283, 282, 295,
    # halo x5 lower contour
    372, 340, 346, 347, 348, 349, 350, 357, 465,
    # halo x5 upper contour excluding corners or eyebrow outer contour
    383, 300, 293, 334, 296, 336, 285, 417,
]

# 35mm camera sensor diagonal (36mm * 24mm)
SENSOR_DIAGONAL_35MM = np.math.sqrt(36 ** 2 + 24 ** 2)
# average human iris size
IRIS_SIZE_IN_MM = 11.8


class IrisIndex(IntEnum):
    """Index into iris landmarks as returned by `IrisLandmark`
    """
    CENTER = 0
    LEFT = 1
    TOP = 2
    RIGHT = 3
    BOTTOM = 4

# https://github.com/metalwhale/hand_tracking/blob/b2a650d61b4ab917a2367a05b85765b81c0564f2/run.py
#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

POSE_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,7),
    (0,4), (4,5), (5,6), (6,8),
    (9,10),
    (11,13), (13,15), (15,17), (17,19), (19,15), (15,21),
    (12,14), (14,16), (16,18), (18,20), (20,16), (16,22),
    (11,12), (12,24), (24,23), (23,11)
]
EYE_ALL=[(33, 7), (7, 163), (163, 144), (144, 145), (145, 153),
    (246,161),(161,160),(160,159),(159,158),(158,157),(157,173),
    (153, 154), (154, 155), (155, 133), (33, 246), 
    (173, 133), 
    #(35,124), (124,46), (46,53) ,(53,52), (52,65),
    # halo x5 lower contour
    #(130,25), (25,110),(110,24),(24,23), (23,22), (22,26, (26,112) ,(112,243),
    # halo x5 upper contour excluding corners or eyebrow outer contour
     (226,31), (31,228), (228,229),(229,230), (230,231), (231,232), (232,233), (233,244),
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105),
    (105, 66), (66, 107),
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380),
    (380, 381), (381, 382), (382, 362), (263, 466), (466, 388),
    (388, 387), (387, 386), (386, 385), (385, 384), (384, 398),
    (398, 362)



]
# Vertex indices can be found in
# github.com/google/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualisation.png
# Found in github.com/google/mediapipe/python/solutions/face_mesh.py

FACE_CONNECTIONS = [
    # Lips.
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
    (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0),
    (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
    (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
    (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
    # Left eye.
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380),
    (380, 381), (381, 382), (382, 362), (263, 466), (466, 388),
    (388, 387), (387, 386), (386, 385), (385, 384), (384, 398),
    (398, 362),
    # Left eyebrow.
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293),
    (293, 334), (334, 296), (296, 336),
    # Right eye.
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153),
    (153, 154), (154, 155), (155, 133), (33, 246), (246, 161),
    (161, 160), (160, 159), (159, 158), (158, 157), (157, 173),
    (173, 133),
    # Right eyebrow.
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105),
    (105, 66), (66, 107),
    # Face oval.
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251),
    (251, 389), (389, 356), (356, 454), (454, 323), (323, 361),
    (361, 288), (288, 397), (397, 365), (365, 379), (379, 378),
    (378, 400), (400, 377), (377, 152), (152, 148), (148, 176),
    (176, 149), (149, 150), (150, 136), (136, 172), (172, 58),
    (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109),
    (109, 10)
]
