import cv2
import numpy as np
from skimage.color import rgb2gray




class CarRacingFeatureExtractor:
    def __init__(self):
        self.prev_position = None

    def extract_features(self, obs):
        gray_image = rgb2gray(obs)
        car_position = self.get_car_position(gray_image)
        track_center = [obs.shape[1] // 2, obs.shape[0] // 2]


        speed = self.calculate_speed(car_position)

        angle = self.calculate_angle(car_position, track_center)

        distance = np.linalg.norm(np.array(car_position) - np.array(track_center))

        return speed, angle, distance

    def get_car_position(self, gray_image):
        lower_red = np.array([0, 0, 120])
        upper_red = np.array([50, 50, 255])
        mask = cv2.inRange(gray_image, lower_red, upper_red)

        # Find contours in the mask and select the largest contour as the car
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                return cX, cY

        # Default position if not found
        return gray_image.shape[1] // 2, gray_image.shape[0] // 2

    def calculate_speed(self, car_position):
        if self.prev_position is None:
            self.prev_position = car_position
            return 0.0

        # Euclidean distance between current and previous position
        speed = np.linalg.norm(np.array(car_position) - np.array(self.prev_position))
        self.prev_position = car_position
        return speed

    def calculate_angle(self, car_position, track_center):
        delta_x = car_position[0] - track_center[0]
        delta_y = car_position[1] - track_center[1]
        angle = np.arctan2(delta_y, delta_x)  # Angle with respect to the centerline
        return angle