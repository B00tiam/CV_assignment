# color model
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def histogram(img_path):

    proj_img = cv.imread(img_path)
