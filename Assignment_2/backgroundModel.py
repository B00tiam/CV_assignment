import cv2
import numpy as np
import os

# calculate average for background model creation
def calc_bg_avg(video_path):
    # initialize varibale frame_sum to store the sum of all frames
    frame_sum = None
    frame_count = 0

    # open video file
    cap = cv2.VideoCapture(video_path)

    while True:
        # read a frame from the video
        ret, frame = cap.read()
        # if frame is not read correctly, then break while loop.
        if not ret:
            break

        # convert frame to float. allows for more precise accumulation
        frame_float = frame.astype(np.float32)

        # initialize frame_sum if it's the first frame
        if frame_sum is None:
            frame_sum = frame_float
        else:
            #add current frame to sum
            frame_sum += frame_float

        frame_count += 1
    # calculate average
    background_avg = frame_sum / frame_count
    # convert average back uint8
    background_avg = np.uint8(background_avg)
    # release video file
    cap.release()

    return background_avg

root_dir = 'C:\\Users\\luiho\\PycharmProjects\\CV_assignment\\Assignment_2\\data'

camera_dirs = ['cam1', 'cam2', 'cam3', 'cam4']

# iterate over each camera directory which has 'background.avi'
for cam_dir in camera_dirs:
    video_path = os.path.join(root_dir, cam_dir, 'background.avi')
    background = calc_bg_avg(video_path)
    # save the background model as image for subtraction
    bg_img_path = os.path.join(root_dir, cam_dir, 'background_model.jpg')
    cv2.imwrite(bg_img_path, background)
    print(f"Background model saved for {cam_dir} at {bg_img_path}")