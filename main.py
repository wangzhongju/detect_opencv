######################
#
# Demo for detect
#
######################


import cv2
import os
import glob
import time
import numpy as np



def detect_line(r):
    # showCrosshair = False
    # fromCenter = False
    # r = cv2.selectROI("Image", imCrop, fromCenter, showCrosshair)
    # print(r)
    ROI_line = imCrop[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    ROI_line_bgr = imCrop_bgr[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    ROI_line_f32 = np.float32(ROI_line)
    # ROI_line_bgr_gray = cv2.cvtColor(ROI_line_bgr, cv2.COLOR_BGR2GRAY)
    # corner_uint8 = cv2.cornerHarris(ROI_line_bgr_gray, 2, 3, 0.04)
    corner = cv2.cornerHarris(ROI_line_f32, 2, 3, 0.04)
    ROI_line_bgr[corner>0.01*corner.max()] = [0,0,255]
    cv2.imwrite("./32corner.tiff", ROI_line_bgr)
    index = np.where(np.array(corner)>0.01*corner.max())
    corner_pixel_coordinate = []
    for i in range(len(index[0])):
        if index[0][i] > r[3]//2:
            corner_pixel_coordinate.append([index[1][i], index[0][i]])
    
    left,right,lt,rt = [0,0],[0,0],[],[]
    lt = corner_pixel_coordinate[0]
    for i in range(len(corner_pixel_coordinate)):
        if abs(corner_pixel_coordinate[i][1]-lt[1])<30 and abs(corner_pixel_coordinate[i][0]-lt[0])>50:
            rt = corner_pixel_coordinate[i]
            break
    w_index = [v[0] for v in corner_pixel_coordinate]
    r_index = np.argmax(w_index)
    l_index = np.argmin(w_index)
    left = corner_pixel_coordinate[l_index]
    right = corner_pixel_coordinate[r_index]

    lt = (lt[0]+r[0], lt[1]+r[1])
    rt = (rt[0]+r[0], rt[1]+r[1])
    left = (left[0]+r[0], left[1]+r[1])
    right = (right[0]+r[0], right[1]+r[1])
    obj_left_bot = (lt[0], left[1])
    obj_left_top = lt
    obj_right_bot = (rt[0], right[1])
    obj_right_top = rt

    # 100, 100, 20
    threshold = 100
    minLineLength = 100
    maxLineGap = 20
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    gray = cv2.cvtColor(ROI_line_bgr, cv2.COLOR_BGR2GRAY)
    gaus = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gaus, 16, 80, apertureSize=3)
    dst = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    lines = cv2.HoughLinesP(dst, 1, np.pi/180, threshold, minLineLength, maxLineGap)
    left_line_coor = np.empty(shape=[0,4], dtype=int)
    right_line_coor = np.empty(shape=[0,4], dtype=int)
    if lines is not None:
        lines = lines[:,0,:]
        for x1,y1,x2,y2 in lines:
            # cv2.line(ROI_line_bgr, (x1,y1), (x2,y2), (255,0,0), 2)
            if abs(x1-x2) < 10 and abs(x1+r[0]-obj_left_top[0]) < 5:
                left_line_coor = np.append(left_line_coor, [[x1,y1,x2,y2]], axis=0)
            elif abs(x1-x2) < 10 and abs(x1+r[0]-obj_right_top[0]) < 5:
                right_line_coor = np.append(right_line_coor, [[x1,y1,x2,y2]], axis=0)
    if left_line_coor.size != 0:
        left_line_arg = np.argmin(left_line_coor[:,-1])
        x1,y1,x2,y2 = left_line_coor[left_line_arg]
        obj_left_bot = (x1+r[0], obj_left_bot[1])
        obj_left_top = (x2+r[0], y2+r[1])
    if right_line_coor.size != 0:
        right_line_arg = np.argmin(right_line_coor[:,-1])
        x1,y1,x2,y2 = right_line_coor[right_line_arg]
        obj_right_bot = (x1+r[0], obj_right_bot[1])
        obj_right_top = (x2+r[0], y2+r[1])
        

    cv2.line(imCrop_bgr, lt, rt, (0,255,0), 1)
    cv2.line(imCrop_bgr, left, right, (0,255,0), 1)
    cv2.line(imCrop_bgr, obj_left_bot, obj_left_top, (0,0,255), 1)
    cv2.line(imCrop_bgr, obj_right_bot, obj_right_top, (0,0,255), 1)

    return (left, right, lt, rt)


def detect_circle(r):
    # showCrosshair = False
    # fromCenter = False
    # r = cv2.selectROI("Image", imCrop, fromCenter, showCrosshair)
    # print(r)
    ROI_Circle = imCrop[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    ROI_Circle_bgr = imCrop_bgr[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    cv2.imshow("circle", ROI_Circle)
    height, width = ROI_Circle.shape[:2]
    ROI_Circle_f32 = np.float32(ROI_Circle)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # ROI_Circle_f32 = cv2.erode(ROI_Circle_f32, kernel, iterations=3)
    gradient = np.zeros((height, width))
    for x in range(height-1):
        for y in range(width-1):
            gx = abs(ROI_Circle_f32[x+1,y]-ROI_Circle_f32[x,y])
            gy = abs(ROI_Circle_f32[x,y+1]-ROI_Circle_f32[x,y])
            gradient[x,y] = gx+gy
    _, gradient = cv2.threshold(gradient, 500, 255, cv2.THRESH_BINARY)

    test = 1
    if test:
        # 2. filter gradient
        for i in range(height//kernel.shape[0]):
            for j in range(width//kernel.shape[0]):
                v1 = kernel[1,1] * gradient[3*i+1, 3*j+1] > 0
                v2 = kernel[0,1] * gradient[3*i, 3*j+1] > 0
                v3 = kernel[2,1] * gradient[3*i+2, 3*j+1] > 0
                v4 = kernel[1,0] * gradient[3*i+1, 3*j] > 0
                v5 = kernel[1,2] * gradient[3*i+1, 3*j+2] > 0
                if v1*v2*v3*v4*v5 == False:
                    gradient[3*i, 3*j]     = 0
                    gradient[3*i, 3*j+1]   = 0
                    gradient[3*i, 3*j+2]   = 0
                    gradient[3*i+1, 3*j]   = 0
                    gradient[3*i+1, 3*j+1] = 0
                    gradient[3*i+1, 3*j+2] = 0
                    gradient[3*i+2, 3*j]   = 0
                    gradient[3*i+2, 3*j+1] = 0
                    gradient[3*i+2, 3*j+2] = 0

    with open("./test_ori.txt", "w") as fp:
        for i in range(height):
            for j in range(width):
                fp.writelines("%04d"%(gradient[i,j])+" ")
            fp.writelines("\n")
    # gradient = cv2.convertScaleAbs(gradient)
    # _, contours, hierarchy = cv2.findContours(gradient, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(ROI_Circle_bgr, contours, -1, (0,0,255), 3)
    # cv2.imshow("contours", ROI_Circle_bgr)

    # 3. build bbox with extreme coordinate
    arg_x = np.argmax(gradient, 0)
    arg_y = np.argmax(gradient, 1)
    for i,v in enumerate(arg_x):
        if v != 0 and i > 10:
            left = (i,v)
            break
    for i,v in enumerate(arg_x[::-1]):
        if v != 0 and i > 15:
            right = (len(arg_x)-1-i,v)
            break
    for i,v in enumerate(arg_y):
        if v != 0 and i > 15:
            top = (v,i)
            break
    for i,v in enumerate(arg_y[::-1]):
        if v != 0 and i > 10:
            bot = (v,len(arg_y)-1-i)
            break
    # left_top = (left[0]+roi[0], top[1]+roi[1])
    # right_bot = (right[0]+roi[0], bot[1]+roi[1])
    left_top = (left[0]+r[0], top[1]+r[1])
    right_bot = (right[0]+r[0], bot[1]+r[1])
    cv2.rectangle(imCrop_bgr, left_top, right_bot, (0,255,0),2)




def detect():
    # roi_obj = (51, 91, 812, 649)
    roi_obj = (95, 95, 755, 655)
    coor = detect_line(roi_obj)  #(left, right, lt, rt)

    roi_circle_left = (237, 446, 83, 86) # (lt_w, lt_h, w, h)
    coor_x = coor[3][0] - roi_circle_left[0] - roi_circle_left[2] + coor[2][0]
    coor_y = roi_circle_left[1]
    coor_w = roi_circle_left[2]
    coor_h = roi_circle_left[3]
    roi_circle_right = (coor_x, coor_y, coor_w, coor_h)
    # roi_circle_right = (633, 446, 90, 92)
    detect_circle(roi_circle_left)
    detect_circle(roi_circle_right)







if __name__ == "__main__":
    imgs_list = glob.glob("./imgs"+"/*")
    start = time.time()

    test = 1

    if test:
        for imgpath in imgs_list:
            img = cv2.imread(imgpath, -1)
            img_bgr = cv2.imread(imgpath)
            imgname = os.path.basename(imgpath)
            print("imgname:", imgname)
            height, width = img.shape[:2]
            imCrop = cv2.resize(img, (int(width/2), int(height/2)), interpolation=cv2.INTER_LINEAR)
            imCrop_bgr = cv2.resize(img_bgr, (width//2, height//2), interpolation=cv2.INTER_LINEAR)
            detect()
            cv2.imshow("res", imCrop_bgr)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

    elif not test:
        imgpath = "./imgs/d_1.tiff"
        img = cv2.imread(imgpath, -1)
        img_bgr = cv2.imread(imgpath)
        height, width = img.shape[:2]
        imCrop = cv2.resize(img, (int(width/2), int(height/2)), interpolation=cv2.INTER_LINEAR)
        imCrop_bgr = cv2.resize(img_bgr, (width//2, height//2), interpolation=cv2.INTER_LINEAR)
        detect()
        cv2.imshow("res", imCrop_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    end = time.time()
    print("use time:", end - start)