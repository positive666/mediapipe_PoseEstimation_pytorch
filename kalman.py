"""
Direct Copy from https://github.com/yinguobing/head-pose-estimation/blob/master/stabilizer.py
Using Kalman Filter as a point stabilizer to stabilize a 2D point.
"""
import numpy as np

import cv2

class Stabilizer:
    """Using Kalman filter as a point stabilizer."""

    def __init__(self,
                 state_num=4,
                 measure_num=2,
                 cov_process=0.0001,
                 cov_measure=0.1):
        """Initialization"""
        # Currently we only support scalar and point, so check user input first.
        assert state_num == 4 or state_num == 2, "Only scalar and point supported, Check state_num please."

        # Store the parameters.
        self.state_num = state_num
        self.measure_num = measure_num

        # The filter itself.
        self.filter = cv2.KalmanFilter(state_num, measure_num, 0)

        # Store the state.
        self.state = np.zeros((state_num, 1), dtype=np.float32)

        # Store the measurement result.
        self.measurement = np.array((measure_num, 1), np.float32)

        # Store the prediction.
        self.prediction = np.zeros((state_num, 1), np.float32)

        # Kalman parameters setup for scalar.
        if self.measure_num == 1:
            self.filter.transitionMatrix = np.array([[1, 1],
                                                     [0, 1]], np.float32)

            self.filter.measurementMatrix = np.array([[1, 1]], np.float32)

            self.filter.processNoiseCov = np.array([[1, 0],
                                                    [0, 1]], np.float32) * cov_process

            self.filter.measurementNoiseCov = np.array(
                [[1]], np.float32) * cov_measure

        # Kalman parameters setup for point.
        if self.measure_num == 2:
            self.filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                     [0, 1, 0, 1],
                                                     [0, 0, 1, 0],
                                                     [0, 0, 0, 1]], np.float32)

            self.filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                      [0, 1, 0, 0]], np.float32)

            self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], np.float32) * cov_process

            self.filter.measurementNoiseCov = np.array([[1, 0],
                                                        [0, 1]], np.float32) * cov_measure

    def update(self, measurement):
        """Update the filter"""
        # Make kalman prediction
        self.prediction = self.filter.predict()

        # Get new measurement
        if self.measure_num == 1:
            self.measurement = np.array([[np.float32(measurement[0])]])
        else:
            self.measurement = np.array([[np.float32(measurement[0])],
                                         [np.float32(measurement[1])]])

        # Correct according to measurement
        self.filter.correct(self.measurement)

        # Update state value.
        self.state = self.filter.statePost

    def set_q_r(self, cov_process=0.1, cov_measure=0.001):
        """Set new value for processNoiseCov and measurementNoiseCov."""
        if self.measure_num == 1:
            self.filter.processNoiseCov = np.array([[1, 0],
                                                    [0, 1]], np.float32) * cov_process
            self.filter.measurementNoiseCov = np.array(
                [[1]], np.float32) * cov_measure
        else:
            self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], np.float32) * cov_process
            self.filter.measurementNoiseCov = np.array([[1, 0],
                                                        [0, 1]], np.float32) * cov_measure


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



def main():
    """Test code"""
    global mp
    mp = np.array((2, 1), np.float32)  # measurement

    def onmouse(k, x, y, s, p):
        global mp
        mp = np.array([[np.float32(x)], [np.float32(y)]])

    cv2.namedWindow("kalman")
    cv2.setMouseCallback("kalman", onmouse)
    kalman = Stabilizer(4, 2)
    frame = np.zeros((480, 640, 3), np.uint8)  # drawing canvas

    while True:
        kalman.update(mp)
        point = kalman.prediction
        state = kalman.filter.statePost
        cv2.circle(frame, (state[0], state[1]), 2, (255, 0, 0), -1)
        cv2.circle(frame, (point[0], point[1]), 2, (0, 255, 0), -1)
        cv2.imshow("kalman", frame)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break


if __name__ == '__main__':
    main()