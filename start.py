#===================================
#   this is a test file
#   author: wzj
#     time: 2019-12-12 17:15:43
#

import cv2
import glob
import numpy as np
from collections import Counter




def detect_circle(img, roi):
    """
    detect circle
    inpput:
        img: gray picture
        roi: Relative position between img and original picture (left_top_x, left_top_y, w, h)
    output:
        bbox of circle
    """
    row, column = img.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    img_f = img.copy()
    img_f = img_f.astype("float")
    # 1.calculate gradient
    gradient = np.zeros((row, column))
    for x in range(row-1):
        for y in range(column-1):
            gx = abs(img_f[x+1,y]-img_f[x,y])
            gy = abs(img_f[x,y+1]-img_f[x,y])
            gradient[x,y] = gx+gy
    sharp = img_f + 15*gradient
    sharp = sharp.astype("uint8")
    sharp = np.where(sharp<0,0,np.where(sharp>255,255,sharp)) #cv::saturate_cast
    _, gradient_binarization = cv2.threshold(gradient, 1, 255, cv2.THRESH_BINARY)
    # 2. filter gradient
    hg, wg = gradient_binarization.shape
    for i in range(hg//kernel.shape[0]):
        for j in range(wg//kernel.shape[0]):
            v1 = kernel[1,1] * gradient_binarization[3*i+1, 3*j+1] > 0
            v2 = kernel[0,1] * gradient_binarization[3*i, 3*j+1] > 0
            v3 = kernel[2,1] * gradient_binarization[3*i+2, 3*j+1] > 0
            v4 = kernel[1,0] * gradient_binarization[3*i+1, 3*j] > 0
            v5 = kernel[1,2] * gradient_binarization[3*i+1, 3*j+2] > 0
            if v1*v2*v3*v4*v5 == False:
                gradient_binarization[3*i, 3*j]     = 0
                gradient_binarization[3*i, 3*j+1]   = 0
                gradient_binarization[3*i, 3*j+2]   = 0
                gradient_binarization[3*i+1, 3*j]   = 0
                gradient_binarization[3*i+1, 3*j+1] = 0
                gradient_binarization[3*i+1, 3*j+2] = 0
                gradient_binarization[3*i+2, 3*j]   = 0
                gradient_binarization[3*i+2, 3*j+1] = 0
                gradient_binarization[3*i+2, 3*j+2] = 0
    h,w = gradient_binarization.shape
    gradient_binarization = gradient_binarization.astype("uint8")
    with open("./gradient.txt", "w") as fp:
        for i in range(h):
            for j in range(w):
                fp.writelines("%03d"%(gradient_binarization[i,j])+" ")
            fp.writelines("\n")
    # 3. build bbox with extreme coordinate
    arg_x = np.argmax(gradient_binarization, 0)
    arg_y = np.argmax(gradient_binarization, 1)
    for i,v in enumerate(arg_x):
        if v != 0:
            left = (i,v)
            break
    for i,v in enumerate(arg_x[::-1]):
        if v != 0:
            right = (len(arg_x)-1-i,v)
            break
    for i,v in enumerate(arg_y):
        if v != 0:
            top = (v,i)
            break
    for i,v in enumerate(arg_y[::-1]):
        if v != 0:
            bot = (v,len(arg_y)-1-i)
            break
    left_top = (left[0]+roi[0], top[1]+roi[1])
    right_bot = (right[0]+roi[0], bot[1]+roi[1])
    cv2.rectangle(imCrop, left_top, right_bot, (0,255,0), 2)


def detect_line(img, lowThreshold):
    """
    detect line
    inpput:
        img: gray picture
    output:
        linear equation
    """
    # init parameters
    ratio = 5
    kernel_size = 3
    threshold = 30
    minLineLength = 10
    maxLineGap = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    detect_edge = cv2.GaussianBlur(img, (3,3), 0)
    detect_edge = cv2.Canny(detect_edge,
                    lowThreshold,
                    lowThreshold*ratio,
                    apertureSize = kernel_size)
    cv2.imshow("edge", detect_edge)
    dst = cv2.morphologyEx(detect_edge, cv2.MORPH_CLOSE, kernel)
    corner = cv2.cornerHarris(dst, 2, 3, 0.04)
    index = np.where(np.array(corner)>0.01*corner.max())
    corner_pixel_coordinate = []
    for i in range(len(index[0])):
        corner_pixel_coordinate.append([index[0][i], index[1][i]])
    left_roi    = (120,565,35,35)
    right_roi   = (790,565,35,35)
    line1_left  = (190,525,25,25)
    line1_right = (730,525,25,25)
    h1,h2,w1,w2 = [],[],[],[]
    H1,H2,W1,W2 = [],[],[],[]
    for coor in corner_pixel_coordinate:
        if left_roi[0]<coor[1]<left_roi[0]+left_roi[2] and \
            left_roi[1]<coor[0]<left_roi[1]+left_roi[3]:
            w1.append(coor[1])
            h1.append(coor[0])
        elif right_roi[0]<coor[1]<right_roi[0]+right_roi[2] and \
            right_roi[1]<coor[0]<right_roi[1]+right_roi[3]:
            w2.append(coor[1])
            h2.append(coor[0])
        elif line1_left[0]<coor[1]<line1_left[0]+line1_left[2] and \
            line1_left[1]<coor[0]<line1_left[1]+line1_left[3]:
            W1.append(coor[1])
            H1.append(coor[0])
        elif line1_right[0]<coor[1]<line1_right[0]+line1_right[2] and \
            line1_right[1]<coor[0]<line1_right[1]+right_roi[3]:
            W2.append(coor[1])
            H2.append(coor[0])
    coor1 = (min(w1),min(h1))
    coor2 = (max(w2),min(h2))
    coor3 = (min(W1),min(H1))
    coor4 = (max(W2),min(H2))
    coor5 = (min(W1),min(h1))
    coor6 = (max(W2),min(h2))
    cv2.line(imCrop, coor1, coor2, (0,255,0), 2)
    cv2.line(imCrop, coor3, coor4, (0,255,0), 2)
    cv2.line(imCrop, coor5, coor3, (0,255,0), 2)
    cv2.line(imCrop, coor6, coor4, (0,255,0), 2)

    imCrop[corner>0.01*corner.max()] = [0,0,255]

    lines = cv2.HoughLinesP(dst, 1, np.pi/180, threshold, minLineLength, maxLineGap)
    if lines is not None:
        lines = lines[:,0,:]
        for x1,y1,x2,y2 in lines:
            if abs(y1-y2)<10 and abs(x1-x2)>30:
                cv2.line(imCrop, (x1,y1), (x2,y2), (255,0,0), 2)
            elif abs(y1-y2)>30 and abs(x1-x2)<10:
                cv2.line(imCrop, (x1,y1), (x2,y2), (255,0,0), 2)


def detect(lowThreshold):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))   # 定义内核
    gray = cv2.cvtColor(imCrop, cv2.COLOR_BGR2GRAY)  # 灰度化
    # showCrosshair = False
    # fromCenter = False
    # r = cv2.selectROI("Image", imCrop, fromCenter, showCrosshair)
    # print(r)
    r1 = (240, 455, 70, 75)
    r2 = (635, 455, 70, 75)
    ROI_L = gray[int(r1[1]):int(r1[1]+r1[3]), int(r1[0]):int(r1[0]+r1[2])]
    ROI_R = gray[int(r2[1]):int(r2[1]+r2[3]), int(r2[0]):int(r2[0]+r2[2])]
    # detect circle
    detect_circle(ROI_L, r1)
    detect_circle(ROI_R, r2)
    # detect line
    detect_line(gray, lowThreshold)
    cv2.imwrite("./data/canny.tiff", imCrop)
    cv2.imshow('canny demo',imCrop)
    cv2.waitKey(1000)

            

if __name__ == "__main__":
    """
    """
    imgs_list = glob.glob("./imgs"+"/*")
    for imgpath in imgs_list:
        img = cv2.imread(imgpath)
        height, width = img.shape[:2]
        img = cv2.resize(img, (int(width/2), int(height/2)), interpolation=cv2.INTER_LINEAR)
        # print("shape: ", img.shape)

        imCrop = img.copy()
        # cv2.createTrackbar('Min threshold','canny demo', lowThreshold, max_lowThreshold, CannyThreshold)
        detect(16)
        # cv2.imshow("a", img)