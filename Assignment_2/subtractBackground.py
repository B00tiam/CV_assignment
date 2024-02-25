import cv2
import numpy as np
import os


def subtract_background(video_path, background_model_path):
    cap = cv2.VideoCapture(video_path)
    background_model = cv2.imread(background_model_path)
    background_model_hsv = cv2.cvtColor(background_model, cv2.COLOR_BGR2HSV)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to hsv
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Background subtraction
        foreground_mask = cv2.absdiff(frame_hsv, background_model_hsv)

        # Thresholding
        thresh_hue = cv2.inRange(foreground_mask[:, :, 0], 10, 100)
        thresh_sat = cv2.inRange(foreground_mask[:, :, 1], 30, 255)
        thresh_val = cv2.inRange(foreground_mask[:, :, 2], 40, 255)

        # Combine masks
        combined_mask = cv2.bitwise_and(thresh_hue, cv2.bitwise_and(thresh_sat, thresh_val))

        # Post-processing: Noise reduction (using morphological operations)
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        # Display the result
        cv2.imshow('Foreground Mask', cleaned_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, 'data')
camera_dirs = ['cam1', 'cam2', 'cam3', 'cam4']

for cam_dir in camera_dirs:
    video_path = os.path.join(root_dir, cam_dir, 'video.avi')
    background_model_path = os.path.join(root_dir, cam_dir, 'background_model.jpg')
    subtract_background(video_path, background_model_path)
    print(f"Processed background subtraction for {cam_dir}")
