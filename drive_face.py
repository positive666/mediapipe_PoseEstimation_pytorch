from argparse import ArgumentParser
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from platform import system
import bpy
import mathutils
import time
from imutils import face_utils


class FaceMeshDetector:
    def __init__(self,
                 static_image_mode=False,
                 max_num_faces=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Facemesh
        self.mp_face_mesh = mp.solutions.face_mesh
        # The object to do the stuffs
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            self.static_image_mode,
            self.max_num_faces,
            True,
            self.min_detection_confidence,
            self.min_tracking_confidence
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        # convert the img from BRG to RGB
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        img.flags.writeable = False
        self.results = self.face_mesh.process(img)

        # Draw the face mesh annotations on the image.
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        self.imgH, self.imgW, self.imgC = img.shape

        self.faces = []

        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        image = img,
                        landmark_list = face_landmarks,
                        connections = self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec = self.drawing_spec,
                        connection_drawing_spec = self.drawing_spec)

                face = []
                for id, lmk in enumerate(face_landmarks.landmark):
                    x, y = int(lmk.x * self.imgW), int(lmk.y * self.imgH)
                    face.append([x, y])

                    # show the id of each point on the image
                    # cv2.putText(img, str(id), (x-4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

                self.faces.append(face)

        return img, self.faces



"""
Estimate head pose according to the facial landmarks
"""

class PoseEstimator:

    def __init__(self, img_size=(480, 640)):
        self.size = img_size

        self.model_points_full = self.get_full_model_points()

        # Camera internals
        self.focal_length = self.size[1] #height
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = None
        self.t_vec = None

    def get_full_model_points(self, filename='/media/ubuntu/data/tool/blender-2.82-linux64/pure/model.txt'):
        """Get all 468 3D model points from file"""
        raw_value = []

        with open(filename) as file:
            for line in file:
                raw_value.append(line)

        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (-1, 3))

        return model_points

    def solve_pose_by_all_points(self, image_points):
        """
        Solve pose from all the 468 image points
        Return (rotation_vector, translation_vector) as pose.
        """

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_full, image_points, self.camera_matrix, self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_full,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector),rotation_vector

    def draw_annotation_box(self, image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""
        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 40
        front_depth = 400
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)


    def draw_axis(self, img, R, t):
        axis_length = 20
        axis = np.float32(
            [[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).reshape(-1, 3)

        axisPoints, _ = cv2.projectPoints(
            axis, R, t, self.camera_matrix, self.dist_coeefs)

        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[0].ravel()), (255, 0, 0), 3)
        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[1].ravel()), (0, 255, 0), 3)
        img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[2].ravel()), (0, 0, 255), 3)

    def draw_axes(self, img, R, t):
            img	= cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeefs, R, t, 20)

    def reset_r_vec_t_vec(self):
        self.r_vec = None
        self.t_vec = None







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
    
    def mouth_height(image_points):
        p3 = image_points[13]
        p7 = image_points[14]
        return np.linalg.norm(p3-p7)-0.5


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








def print_debug_msg(args):
    msg = '%.4f ' * len(args) % args
    print(msg)




class MediaPipeAnimOperator(bpy.types.Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "wm.mediapipe_operator"
    bl_label = "MediaPipe Animation Operator"

   
    rig_name = "RIG-Vincent"

    _timer = None
    _cap  = None
    
    width = 800  #640
    height = 600 #480

    stop :bpy.props.BoolProperty()
    


    # Facemesh
    detector = FaceMeshDetector()



    # Pose estimation related 480 640
    pose_estimator = PoseEstimator((height, width))
    image_points = np.zeros((pose_estimator.model_points_full.shape[0], 2))

    # extra 10 points due to new attention model (in iris detection)
    iris_image_points = np.zeros((10, 2))


    
       # Keeps a moving average of given length
    def smooth_value(self, name, length, value):
        if not hasattr(self, 'smooth'):
            self.smooth = {}
        if not name in self.smooth:
            self.smooth[name] = np.array([value])
        else:
            self.smooth[name] = np.insert(arr=self.smooth[name], obj=0, values=value)
            if self.smooth[name].size > length:
                self.smooth[name] = np.delete(self.smooth[name], self.smooth[name].size-1, 0)
        sum = 0
        for val in self.smooth[name]:
            sum += val
        return sum / self.smooth[name].size

    def modal(self, context, event):

        if (event.type in {'RIGHTMOUSE', 'ESC'}) or self.stop == True:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            self.init_camera()
            success, img = self._cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                return {'PASS_THROUGH'}
            
            
            img_facemesh, faces = self.detector.findFaceMesh(img)

            # flip the input image so that it matches the facemesh stuff
            img = cv2.flip(img, 1)

            # if there is any face detected
            if faces:
                print("face")
                # only get the first face
                for i in range(len(self.image_points)):
                    self.image_points[i, 0] = faces[0][i][0]
                    self.image_points[i, 1] = faces[0][i][1]
                    
                for j in range(len(self.iris_image_points)):
                    self.iris_image_points[j, 0] = faces[0][j + 468][0]
                    self.iris_image_points[j, 1] = faces[0][j + 468][1]

                # The third step: pose estimation
                # pose: [[rvec], [tvec]]
                pose,rotation_vector = self.pose_estimator.solve_pose_by_all_points(self.image_points)

                x_ratio_left, y_ratio_left = FacialFeatures.detect_iris(self.image_points, self.iris_image_points, Eyes.LEFT)
                x_ratio_right, y_ratio_right = FacialFeatures.detect_iris(self.image_points, self.iris_image_points, Eyes.RIGHT)


                ear_left = FacialFeatures.eye_aspect_ratio(self.image_points, Eyes.LEFT)
                ear_right = FacialFeatures.eye_aspect_ratio(self.image_points, Eyes.RIGHT)

                pose_eye = [ear_left, ear_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right]

                mar = FacialFeatures.mouth_aspect_ratio(self.image_points)
                mouth_distance = FacialFeatures.mouth_distance(self.image_points) #width
                mouth_height = FacialFeatures.mouth_height(self.image_points)
                

                # print("left eye: %.2f, %.2f" % (x_ratio_left, y_ratio_left))
                # print("right eye: %.2f, %.2f" % (x_ratio_right, y_ratio_right))


                
                bones = bpy.data.objects['RIG-Vincent'].pose.bones

                if not hasattr(self, 'first_angle'):
                    self.first_angle = np.copy(rotation_vector)
                    
                x=rotation_vector[0]
                y=rotation_vector[1]
                z=rotation_vector[2]   
                bones["head_fk"].rotation_euler[0] = x - self.first_angle[0]  # Up/Down
                bones["head_fk"].rotation_euler[2] = -(y - self.first_angle[1])  # Rotate
                bones["head_fk"].rotation_euler[1] = z - self.first_angle[2]  # Left/Right
                
                bones["head_fk"].keyframe_insert(data_path="rotation_euler", index=-1)
                
                

            else:
                # reset our pose estimator
                pass

                     
            cv2.imshow("Output",img_facemesh)
            cv2.waitKey(1)

        return {'PASS_THROUGH'}
    
    def init_camera(self):
        if self._cap == None:
            self._cap = cv2.VideoCapture("/home/project/code/RobustVideoMatting-master/tesla.mp4")
            #self._cap = cv2.VideoCapture("/media/ubuntu/data/sign_videos/HabenSieSchmerzen0.mp4")
            
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(0.5)
    
    def stop_playback(self, scene):
        print(format(scene.frame_current) + " / " + format(scene.frame_end))
        if scene.frame_current == scene.frame_end:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        
    def execute(self, context):
        bpy.app.handlers.frame_change_pre.append(self.stop_playback)

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.02, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        cv2.destroyAllWindows()
        self._cap.release()
        self._cap = None

def register():
    bpy.utils.register_class(MediaPipeAnimOperator)

def unregister():
    bpy.utils.unregister_class(MediaPipeAnimOperator)

if __name__ == "__main__":
    register()

    # test call
    bpy.ops.wm.mediapipe_operator()
