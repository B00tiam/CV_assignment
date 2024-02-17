import cv2 as cv

# Create a VideoCapture object and open the Webcam
cap = cv.VideoCapture(0)
# The parameter '0' represents the default camera device.
# If you have multiple cameras, you can try different index values (e.g., 1, 2) to select other cameras.

while True:
    # Read video frames in a loop
    ret, frame = cap.read()
    if not ret:
        break
    # Perform operations on each frame here

    # Display the frame
    cv.imshow('Webcam', frame)

    # Check for keyboard input and exit the loop when 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

# Release resources and close the window
cap.release()
cv.destroyAllWindows()