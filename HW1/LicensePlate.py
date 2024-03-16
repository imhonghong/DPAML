# paste your code below
import cv2
import ImageProcess
import numpy as np

def order_points_new(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    if leftMost[0,1]!=leftMost[1,1]:
        leftMost=leftMost[np.argsort(leftMost[:,1]),:]
    else:
        leftMost=leftMost[np.argsort(leftMost[:,0])[::-1],:]
    (tl, bl) = leftMost
    if rightMost[0,1]!=rightMost[1,1]:
        rightMost=rightMost[np.argsort(rightMost[:,1]),:]
    else:
        rightMost=rightMost[np.argsort(rightMost[:,0])[::-1],:]
    (tr,br)=rightMost
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="int")

def main(ksize:list,thre:list,itera:list,image_list:list)->None:
    #ksize=[4]
    #thre=[80]
    #itera=[1]
    
    result = []


    for k in range(len(ksize)):
        for t in range(len(thre)):
            for i in range(len(itera)):
                for image in image_list:
                    ori_img = cv2.imread(image) 

                    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
                    img=ImageProcess.Image_Filter(img,'bilateralFilter',show_image=False,size=ksize[k])             ### ksize
                    post_img=ImageProcess.Edge_Detection(img,'Sobel',gray=False, show_image=False)
                    # post_img=ImageProcess.Image_Filter(post_img,'GaussianBlur',show_image=True,size=4)
                    ret, th1 = cv2.threshold(post_img,thre[t],255,cv2.THRESH_BINARY)                               ### thre

                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    #執行影象形態學

                    # 閉運算
                    closed = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
                    closed = cv2.dilate(closed, None, iterations=itera[i])                  ### iterations
                    closed = cv2.erode(closed, None, iterations=itera[i])                   ### iterations

                    (cnts, _) = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                    possible_img = ori_img.copy()
                    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
                    rect = cv2.minAreaRect(c)
                    Box = np.int0(cv2.boxPoints(rect))
                    Final_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)


                    possible_img = ori_img.copy()

                    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
                        #print (c)
                    
                        rect = cv2.minAreaRect(c)
                        Box = np.int0(cv2.boxPoints(rect))
                        Box=order_points_new(Box) # return  左上/右上/右下/左下 (x,y)                
                        possible_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)

                    possible_img = ori_img.copy()

                    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
                    
                        rect = cv2.minAreaRect(c)
                        Box = np.int0(cv2.boxPoints(rect))
                        Box=ImageProcess.order_points_new(Box)
                    
                        # determine by 工人智慧,指定方框條件設定,同學們可以在這裡調整條件
                    
                        if (((Box[1][0]-Box[0][0])-(Box[3][1]-Box[0][1]))>40) and 20<abs(Box[1][1]-Box[2][1])<100  and 60<abs(Box[0][0]-Box[1][0])<200  and abs(Box[0][1]-Box[1][1])<20  :
                            possible_img = cv2.drawContours(possible_img, [Box], -1, (0, 0, 255), 3)

                            break

                    result.append(possible_img)
                
                cv2.imwrite('output/01.jpg', result[0])
                cv2.imwrite('output/02.jpg', result[1])
                cv2.imwrite('output/03.jpg', result[2])
############################################################################################
if __name__ == '__main__':
    image_list = ['image/01.jpg', 'image/02.jpg', 'image/03.jpg']
    ksize = [4]
    thre=[80]
    itera=[1]
    main(ksize,thre,itera,image_list)
