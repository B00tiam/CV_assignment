import cv2 as cv
import numpy as np

# function to display the coordinates of
# of the points clicked on the image

stored_coordinates = None
img = None

# function to get the param img

def manual_process(images_invalid):

	for fname in images_invalid:
		stored_coordinates = []
		img = cv.imread(fname, 1)
		# cv.setMouseCallback('img', lambda event, x, y, flags, img=img: click_event(event, x, y, flags, img))
		# Reshape the window
		cv.namedWindow('img', cv.WINDOW_NORMAL)
		cv.resizeWindow('img', img.shape[1], img.shape[0])

		cv.setMouseCallback('img', click_event, {'stored_coordinates': stored_coordinates, 'img': img})

		cv.imshow('img', img)
		cv.waitKey(0)

		cv.destroyAllWindows()


def click_event(event, x, y, flags, params):
	if event == cv.EVENT_LBUTTONDOWN:
		print(x, '', y)
		font = cv.FONT_HERSHEY_SIMPLEX
		cv.putText(params['img'], str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2)
		cv.imshow('img', params['img'])
		params['stored_coordinates'].append((x, y))

		# Draw corners only if the correct number of corners is reached
		if len(params['stored_coordinates']) >= 6 * 9:
			# Draw and display the corners and lines
			cv.drawChessboardCorners(params['img'], (9, 6), np.array(params['stored_coordinates']), True)
			for i in range(1, len(params['stored_coordinates'])):
				cv.line(params['img'], tuple(params['stored_coordinates'][i - 1]),
						tuple(params['stored_coordinates'][i]), (0, 255, 0), 2)

			cv.imshow('img', params['img'])
			cv.waitKey(500)


