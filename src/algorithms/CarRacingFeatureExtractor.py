import cv2
import numpy as np
from skimage.color import rgb2gray


#TODO moyna pridat hodnotu kam se meni trat ale ted se mi nechce a ay ted to ma jiste vysledkz

class CarRacingFeatureExtractor:
    def __init__(self):
        self.prev_frame = None

    def extract_features(self, obs):
        car_position = self.get_car_position(obs)
        track_center = [obs.shape[1] // 2, obs.shape[0] // 2]


        speed = self.calculate_speed(obs)

        angle = self.calculate_angle(obs, track_center)

        distance = self.calculate_distance_from_center(obs, track_center)

        return speed, angle, distance

    def get_car_position(self, rgb_image):
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 0, 120])
        upper_red = np.array([50, 50, 255])
        mask = cv2.inRange(hsv_image, lower_red, upper_red)

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
        return rgb_image.shape[1] // 2, rgb_image.shape[0] // 2

    def calculate_speed(self, obs):
        if self.prev_frame is None:
            self.prev_frame = obs
            return 0.0

            # Compute absolute difference between current and previous frames
        diff = cv2.absdiff(obs, self.prev_frame)
        self.prev_frame = obs

        # Count non-zero differences as a proxy for speed
        movement_amount = np.count_nonzero(diff)
        return movement_amount / (obs.shape[0] * obs.shape[1])

    def calculate_angle(self, obs, track_center):
        gray_image = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, 50, 150)

        # Define a small region around the center for analysis
        region = edges[track_center[1] - 5:track_center[1] + 5, :]

        # Find angle of curvature based on region lines
        lines = cv2.HoughLinesP(region, 1, np.pi / 180, threshold=5, minLineLength=5, maxLineGap=10)
        if lines is not None:
            avg_angle = np.mean([np.arctan2(y2 - y1, x2 - x1) for x1, y1, x2, y2 in lines[:, 0]])
            return avg_angle
        return 0.0

    def calculate_distance_from_center(self, obs, track_center):
        # Convert to grayscale and detect edges
        gray_image = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, 50, 150)

        # Sum edges along y-axis to find track's horizontal center
        edge_sum = edges.sum(axis=0)
        track_midpoint = np.argmax(edge_sum)

        # Distance from car's position (center) to track center
        distance = abs(track_midpoint - track_center[0])
        return distance