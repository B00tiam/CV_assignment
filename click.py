import cv2 as cv
import numpy as np

# function to display the coordinates of
# of the points clicked on the image
manual_corners = []
def click_event(event, x, y, flags, params, img):

	# checking for left mouse clicks
	if event == cv.EVENT_LBUTTONDOWN:
		# displaying the coordinates
		# on the Shell
		manual_corners.append((x, y))
		print(x, ' ', y)

		# displaying the coordinates
		# on the image window
		font = cv.FONT_HERSHEY_SIMPLEX
		cv.putText(img, str(x) + ',' +
					str(y), (x, y), font,
					1, (255, 0, 0), 2)
		cv.imshow('image', img)

	# checking for right mouse clicks
	if event == cv.EVENT_RBUTTONDOWN:
		# displaying the coordinates
		# on the Shell
		manual_corners.append((x, y))
		print(x, ' ', y)

		# displaying the coordinates
		# on the image window
		font = cv.FONT_HERSHEY_SIMPLEX
		b = img[y, x, 0]
		g = img[y, x, 1]
		r = img[y, x, 2]
		cv.putText(img, str(b) + ',' +
					str(g) + ',' + str(r),
					(x, y), font, 1,
					(255, 255, 0), 2)
		cv.imshow('image', img)

	cv.waitKey(10000)

	lengths = [5, 8, 5, 8]
	manual_corners.append(manual_corners[0])
	for i in range(0, len(manual_corners) - 1):
		l = lengths[i]
		a = manual_corners[i]
		b = manual_corners[i + 1]
		xy0 = np.array(a)
		xy1 = np.array(b)
		interp = np.linspace(start=xy0, stop=xy1, num=l + 1)
		print(interp)
		print('')

		for xy in interp:
			x, y = xy
			x = int(x.item());
			y = int(y.item())
			print(x, y)
			font = cv.FONT_HERSHEY_SIMPLEX
			cv.circle(img, (x, y), 6, (0, 0, 255), -1)
			cv.imshow('image', img)
			cv.waitKey(0)

	# close the window
	# cv.waitKey(10000)
	cv.destroyAllWindows()


