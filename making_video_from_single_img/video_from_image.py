import numpy as np
import cv2 as cv
import exifread

## Loading Image ##
file_name = 'image.jpg'
image = cv.imread(file_name)


## Setting frame size, number of frames per second, and length of video ##
frame_size = (720 , 720)
frames_number_per_sec = 30
length_of_video = 480

## Building trajectory and giving indices ##
y_trajectory = np.sin(np.linspace(-np.pi , np.pi , num = length_of_video))
x_trajectory = np.linspace(start=500 , stop = image.shape[1] , num=length_of_video , dtype = np.int16)

y_center = image.shape[0] // 2
changed_y_trajectory = []

for val in y_trajectory:
    sign_val = val / abs(val)
    changed_y_val = int(y_center * abs(val))
    changed_y_val = int(y_center + changed_y_val * sign_val)
    changed_y_trajectory.append(changed_y_val)


time_diff = [second - first for second,first in zip(x_trajectory[1:] , x_trajectory)]


## Appending images to a single list ##
imgs = []
current_start = 0

for idx , t in enumerate(changed_y_trajectory):
    img = image[t: , current_start : 500 + current_start , :]
    img = cv.resize(img , frame_size)
    
    imgs.append(img)
    if idx < len(time_diff):
        current_start += time_diff[idx] 

## Overlaying exiff data on image ##
image_file = open(file_name , 'rb')
tags = exifread.process_file(image_file)
date_information = str(tags['EXIF DateTimeOriginal'])

exiff_imgs = []
for idx , img in enumerate(imgs):
    updated_img = cv.putText(img , 
    date_information , 
    org=(50,50) , 
    fontFace=cv.FONT_HERSHEY_SIMPLEX , 
    fontScale=1,
    color = (255,0,0) , thickness=2)

## Exporting video ##
out_video = cv.VideoWriter('final_vid.avi' , cv.VideoWriter_fourcc('M' , 'J' , 'P' , 'G'), frames_number_per_sec , frame_size)

for img in imgs:
    out_video.write(img)

out_video.release()