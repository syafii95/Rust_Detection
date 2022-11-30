# this is a project based on detection of rust iron portion
# you need to pass an argument stating the file name of the image.
# for simplicity keep the script and image in the same folder

import cv2
from sys import argv
import numpy as np
import os
import glob

count = 0




"""def rust_detect(file):
	cap = cv2.VideoCapture('rust.mp4')
	
	while (cap.isOpened()):
		ret, frame = cap.read()
		frame = cv2.resize(frame, (540,380),fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
		cv2.imshow('Frame', frame)
		img = cv2.imread(file)
		img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
        # Range for lower red
		lower_red = np.array([0,70,70])
		upper_red = np.array([20,200,150])
		mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    
        # range for upper red
		lower_red = np.array([170,70,70])
		upper_red = np.array([180,200,150])
		mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    
        # add both masks
		mask = mask0+mask1
    
		output_img = cv2.bitwise_and(img,img,mask=mask)
    
		scale_percent = 60 # percent of original size
		width = int(output_img.shape[1] * scale_percent / 200)
		height = int(output_img.shape[0] * scale_percent / 200)
		dim = (width, height)

		resized = cv2.resize(output_img, dim, interpolation = cv2.INTER_AREA)

		print("\n\n\n Number of pixels depicting rust \n >> %d"%(np.sum(mask)/255))
		#cv2.imshow('image1',resized)
		#cv2.imshow('image2',img)
		cap.release()
		cv2.waitKey(0)
		#cv2.imwrite('output_image%d.jpg'%count,resized)
		#cv2.imwrite('image%d.jpg'%count,img)
		cv2.destroyAllWindows()
		os.system("cls")"""
	


# Creating a VideoCapture object to read the video
cap = cv2.VideoCapture('rust.mp4')


# Loop until the end of the video
while (cap.isOpened()):

	# Capture frame-by-frame
	ret, frame = cap.read()
	frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0,
						interpolation = cv2.INTER_CUBIC)

	# Display the resulting frame
	cv2.imshow('Frame', frame)
	#images = cv2.imread('tank2.jpeg')
	#cv2.imshow('Images', images)

	#print(f"----------- {type(frame)} ------------ {type(images)} ----------")
	
	img_hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Range for lower red
	lower_red = np.array([0,70,70])
	upper_red = np.array([20,200,150])
	mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    
    # range for upper red
	lower_red = np.array([170,70,70])
	upper_red = np.array([180,200,150])
	mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    
    # add both masks
	mask = mask0+mask1
    
	output_img = cv2.bitwise_and(frame,frame,mask=mask)
    
	scale_percent = 60 # percent of original size
	width = int(output_img.shape[1] * scale_percent / 200)
	height = int(output_img.shape[0] * scale_percent / 200)
	dim = (width, height)

	resized = cv2.resize(output_img, dim, interpolation = cv2.INTER_AREA)
	
	cv2.imshow('image1',resized)
	cv2.waitKey(0)
	#cv2.imwrite('output_image%d.jpg'%count,resized)
	# define q as the exit button
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break

# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()


os.system("color 0a")
os.system("cls")

print(""" Welcome to the rust detection software!! 
 The software detects the rusted portion of metal
 and calculates nuber of rust piels for 
 comparitive analysis.\n\n""")
print("**********************************************")

images = glob.glob("Images/*.jpg")

for path in images:
	count+=1
	#rust_detect(path)

input("\n PRESS ENTER TO EXIT ")
