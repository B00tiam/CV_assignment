import cv2
import numpy as np
import os


def subtract_background(video_path, background_model_path, save_dir):
    cap = cv2.VideoCapture(video_path)
    background_model = cv2.imread(background_model_path)
    background_model_hsv = cv2.cvtColor(background_model, cv2.COLOR_BGR2HSV)
    accumulator = None

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

        if accumulator is None:
            accumulator = np.zeros_like(cleaned_mask, dtype=np.float32)

            # Accumulate the changes
        accumulator += cleaned_mask

        _, final_mask = cv2.threshold(accumulator, 1, 255, cv2.THRESH_BINARY)
        final_mask = final_mask.astype(np.uint8)

        mask_save_path = os.path.join(save_dir, 'foreground_mask.jpg')
        cv2.imshow('Foreground Mask', cleaned_mask)
        cv2.imwrite(mask_save_path, final_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

base_dir = 'C:\\Users\\luiho\\PycharmProjects\\CV_assignment\\Assignment_2\\data'
camera_dirs = ['cam1', 'cam2', 'cam3', 'cam4']

for cam_dir in camera_dirs:
    video_path = os.path.join(base_dir, cam_dir, 'video.avi')
    background_model_path = os.path.join(base_dir, cam_dir, 'background_model.jpg')
    save_dir = os.path.join(base_dir, cam_dir)
    subtract_background(video_path, background_model_path, save_dir)
    print(f"Processed background subtraction for {cam_dir}")
