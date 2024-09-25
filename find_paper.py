import cv2
import numpy as np


""" def find_paper(image)
        - image: image name of the image containing the qrcode for the detection of it
"""
def find_qrcode(image):

    # open the image in the "origin/" directory
    img = cv2.imread("origin/"+image)

    # convert it from color to black/white
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # create a gamma table to apply to the image above
    invGamma = 1.0 / 0.3
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    gray = cv2.LUT(gray, table)

    # apply threshold in the detection, adjusts have been made to get the best results by trial/error
    ret,thresh1 = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)

    #thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

    # find the contours of the qrcode
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # function to find the biggest qrcode
    def biggestRectangle(contours):
        #biggest = None
        max_area = 0
        indexReturn = -1
        # iterate through all
        for index in range(len(contours)):
                i = contours[index]
                #calculates area of the current qrcode
                area = cv2.contourArea(i)
                if area > 100:
                    perimeter = cv2.arcLength(i,True)
                    approx = cv2.approxPolyDP(i,0.1*perimeter,True)
                    #update when current area is bigger than the max area obtained until this point at time
                    if area > max_area:
                            #biggest = approx
                            max_area = area
                            indexReturn = index
        #return the index of the biggest qrcode
        return indexReturn

    # find the qrcode
    indexReturn = biggestRectangle(contours)

    # draw the contour
    hull = cv2.convexHull(contours[indexReturn])

    # fill the contour with color
    cv2.drawContours(img, [hull], 0, (0,0,255),3)
    cv2.fillPoly(img, pts=[hull], color=(0, 0, 255))

    # save the image of the contour and of the threshold(just for data gathering)
    cv2.imwrite('./contours/contoured_' + image, img)
    cv2.imwrite('./thresh/thresh_' + image,thresh1)

if __name__ == '__main__':
    find_qrcode("7long_perto.jpg")