import os
import cv2
import numpy as np
import random

def rotate(filename, rotate_angle, ratio):  #ratio: bigger/smaller ratio
    src = cv2.imread(filename)
    rows, cols = src.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2), rotate_angle, ratio)
    res = cv2.warpAffine(src,M,(rows,cols))
    
    #print test
    cv2.imshow("src", src)
    cv2.imshow("res", res)
    cv2.waitKey(0)
    
    return res;
   
def translation(filename,dis_x, dis_y): #pingyi, x: dis_x, y: dis_y
    src = cv2.imread(filename)
    H = np.float32([[1,0,dis_x],[0,1,dis_y]])
    rows,cols = src.shape[:2]
    res = cv2.warpAffine(src,H,(rows,cols))

    #print test
    cv2.imshow("src", src)
    cv2.imshow("res", res)
    cv2.waitKey(0)

    return res;

def changeRGB(filename,n): #randomly change the RGB value of n pixels
    src = cv2.imread(filename)
    rows,cols = src.shape[:2]
    for k in range(n): #Create 5000 noisy pixels
	i = random.randint(0,rows-1)
        j = random.randint(0,cols-1)
        color = (random.randrange(256),random.randrange(256),random.randrange(256))
        src[i,j] = color

    #print test
    cv2.imshow("changeRGB", src)
    cv2.waitKey(0)

    return src

def crop(filename, des_size): # des_size is the size of the target image, for example 224
    src = cv2.imread(filename)
    rows,cols = src.shape[:2]
    if (rows <= des_size) and (cols <= des_size): #if the size of original image is smaller than 224*224
	des = cv2.resize(src, (des_size, des_size))
    else: 
	if rows < cols:
		rows0 = des_size;
		cols0 = cols*des_size/rows
	else:
		rows0 = rows*des_size/cols
		cols0 = des_size
	
        des0 = cv2.resize(src, (cols0, rows0))
	
	i = random.randint(0,cols0-des_size)
        j = random.randint(0,rows0-des_size)
	des = des0[i:i+224,j:j+224]  #cut a 224*224 part of the resized image

	#print test
	#cv2.imshow("crop0", des0)
        cv2.imshow("crop1", des)
        cv2.waitKey(0)

	return des
    

def main():
																																																																																																																																																																																																																																																																																																																																																																																																												
    #test code
    #rotate("1.jpg",45,1)
    #translation("1.jpg", 50, 10)
    #changeRGB("1.jpg", 2000)
    crop("1.jpg", 224)
     

if __name__ == '__main__':
    main()
