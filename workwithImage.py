# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 00:17:00 2021

@author: saura
"""

import numpy as np
import cv2
import math

img = cv2.imread('pic.jpg')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
