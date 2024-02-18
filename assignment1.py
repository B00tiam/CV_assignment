# importing the module

from camera import get_corners




# driver function
if __name__=="__main__":
	
	choose = int(input("Please choose the RUN your want to start:(1, 2 or 3) "))
	# determine start which run
	if choose == 1:
		get_corners('chessboards/run1')  # camera func
	elif choose == 2:
		get_corners('chessboards/run2')  # camera func
	else:
		get_corners('chessboards/run3')  # camera func

