 
# This code was taken from "https://github.com/sourabhkhemka/DocumentScanner/blob/main/RScan/Python/scan/scan.py"
import cv2
import numpy as np

def map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def highPassFilter(img,kSize):
    if not kSize%2:
        kSize +=1
    kernel = np.ones((kSize,kSize),np.float32)/(kSize*kSize)
    filtered = cv2.filter2D(img,-1,kernel)
    filtered = img.astype('float32') - filtered.astype('float32')
    filtered = filtered + 127*np.ones(img.shape, np.uint8)
    filtered = filtered.astype('uint8')
    return filtered

def blackPointSelect(img, blackPoint):
    img = img.astype('int32')
    img = map(img, blackPoint, 255, 0, 255)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO)
    img = img.astype('uint8')
    return img

def whitePointSelect(img,whitePoint):
    _,img = cv2.threshold(img, whitePoint, 255, cv2.THRESH_TRUNC)
    img = img.astype('int32')
    img = map(img, 0, whitePoint, 0, 255)
    img = img.astype('uint8')
    return img

def blackAndWhite(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    (l,a,b) = cv2.split(lab)
    img = cv2.add( cv2.subtract(l,b), cv2.subtract(l,a) )
    return img
def scan_effect(img):
    #img = cv.imread("/home/anju_chhetri/Desktop/PLS/scan_49.jpg")
    blackPoint = 66
    whitePoint = 127
    image = highPassFilter(img,kSize = 51)
    image_white = whitePointSelect(image, whitePoint)
    img_black = blackPointSelect(image_white, blackPoint)
#     image=blackPointSelect(img)
#     white = whitePointSelect(image)
#     img_black = blackAndWhite(white)
    return img_black

#img = cv2.imread("/home/anju_chhetri/Desktop/flowers1/flower_all/flower1.jpg")
#image = scan_effect(img)
#cv2.imshow("fd",image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
