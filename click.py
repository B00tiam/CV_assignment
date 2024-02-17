import cv2 as cv
import numpy as np

# function to display the coordinates of
# of the points clicked on the image

stored_coordinates = None
img = None

def interpolate_internal_corners(external_corners, rows, cols):
    external_corners = external_corners.reshape(4, 2)
    top_edge = np.linspace(external_corners[0], external_corners[1], num=cols)
    bottom_edge = np.linspace(external_corners[3], external_corners[2], num=cols)
    return np.vstack([np.linspace(top_edge[i], bottom_edge[i], num=rows) for i in range(cols)])

def determine_chessboard_orientation(external_corners):
    # Calculate distances between clicked corners
    distances = np.linalg.norm(np.diff(external_corners, axis=0), axis=1)

    # Determine orientation based on distances
    if np.allclose(distances[::2], distances[0]):
        return 6, 9  # Orientation: 6x9
    elif np.allclose(distances[1::2], distances[1]):
        return 9, 6  # Orientation: 9x6
    else:
        return None  # Unable to determine orientation

def manual_process(images_invalid):

	for fname in images_invalid:
		stored_coordinates = []
		img = cv.imread(fname, 1)
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

		# cv.setMouseCallback('img', lambda event, x, y, flags, img=img: click_event(event, x, y, flags, img))
		# Reshape the window
		cv.namedWindow('img', cv.WINDOW_NORMAL)
		cv.resizeWindow('img', img.shape[1], img.shape[0])

		cv.setMouseCallback('img', click_event, {'stored_coordinates': stored_coordinates, 'img': img, 'gray': gray, 'criteria': criteria})

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

		# Draw corners only if enough corners are reached
		if len(params['stored_coordinates']) >= 4:
			external_corners = np.array(params['stored_coordinates'], dtype=np.float32)
			print('External Corners:', external_corners)

			# Determine chessboard orientation
			rows, cols = determine_chessboard_orientation(external_corners)
			if rows and cols:
				interpolated_coordinates = interpolate_internal_corners(external_corners, rows, cols)
				print('Interpolated coordinates:', interpolated_coordinates)

				interpolated_coordinates2 = cv.cornerSubPix(params['gray'], interpolated_coordinates, (80, 80), (-1, -1), params['criteria'])
				print('Subpixel Corners:', interpolated_coordinates2)

				cv.drawChessboardCorners(params['img'], (cols, rows), interpolated_coordinates, True)
				cv.imshow('img', params['img'])
				cv.waitKey(500)
			else:
				print('Unable to determine chessboard orientation.')

	# cv.imshow('img', params['img'])
	# cv.waitKey(500)


