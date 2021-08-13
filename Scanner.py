import cv2
from pathlib import Path
from PIL import Image
import os
import numpy as np
import argparse
from Scan_effect import *
#cap = cv2.VideoCapture('rtsp://admin:123456@192.168.1.103:8080/H264?ch=1&subtype=0')
def basic(pdf_name, orientation, perspective):
    url = 'http://192.168.1.103:8080/video'
    path = str(Path.home())
    path = path + "/Desktop/Mobile-to-PC-Scanner/"
    folder_name = "PLS"
    folder = os.path.join(path, folder_name)
    pdf_folder = os.path.join(folder, "pdf")
    camera_image_folder = os.path.join(folder, "camera_image")
    file_exists = os.path.isdir(folder)
    camera_image_exists = os.path.isdir(camera_image_folder)
    if(not file_exists):
        os.mkdir(folder)
    pdf_exists = os.path.isdir(pdf_folder)
    if(not pdf_exists):
        os.mkdir(pdf_folder)
    if(not camera_image_exists):
        os.mkdir(camera_image_folder)
    image_list = []
    vid = cv2.VideoCapture(url)
    img_count = len(os.listdir(camera_image_folder))
    count_increase = img_count
    while(True):
        ret, frame = vid.read()
        width = int(frame.shape[1]*(50/100))
        height = int(frame.shape[0] * (60/100))

        frame = cv2.resize(frame, (width,height),interpolation=cv2.INTER_AREA)
        if(orientation == 'n'):
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        for_display = frame.copy()
        cv2.imshow("video",for_display)
        key = cv2.waitKey(1)
        if  key==113:
            break
        if key == 32:
            image_name = camera_image_folder + "/scan_" +str(count_increase) +  ".jpg"
            count_increase +=1
            cv2.imwrite(image_name,frame)
            image_list.append(frame)
    vid.release()
    cv2.destroyAllWindows()
    if(len(image_list)==0):
        print("No picture.")
        exit()
    scan_image_list =[]
    if(perspective == 'y'):
        for image in image_list:
            ima = process_image(image)
            ima = Image.fromarray(ima)
            scan_image_list.append(ima)
    else:
        for image in image_list:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ima = Image.fromarray(image)
            scan_image_list.append(ima)
    first_image = scan_image_list[0]
    scan_image_list = scan_image_list[1:]
    pdf = pdf_folder +"/"+ pdf_name
    first_image.save(pdf, save_all = True, append_images = scan_image_list)


# image = cv2.imread("/home/anju_chhetri/Desktop/scan_images/scan_image6.jpg")
# image = cv2.resize(image, (700, 800))
# returned_image = process_image(image)
# cv2.imshow("win", returned_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
def return_one_contour(contour , image):
    max_contour=0
    j=0
    coordinate = np.array([])
    for i in contour:
        area = cv2.contourArea(i)
        perimeter = cv2.arcLength(i, True)
        vertices = cv2.approxPolyDP(i,perimeter*0.02,True)
        if (area>1000) and (len(vertices)==4):
            max_contour = area
            coordinate = vertices
    if(len(coordinate) == 4):
        scanned = futher(coordinate, image)
    else:
        scanned = scan_effect(image)
        scanned = cv2.resize(scanned, (600,700))
         # scanned = image
        #_,scanned = cv2.threshold(im,127,255,cv2.THRESH_BINARY)

    return scanned

def futher(coordinate, image):
    scanned = scan_effect(image)
    points = coordinate.reshape((4,2))
    points = reorder(points)
    point1 = np.float32(points)
    point2 = np.float32([[0,0],[600,0],[0,700],[600,700]])
    matrix = cv2.getPerspectiveTransform(point1, point2)
    wrap_image = cv2.warpPerspective(scanned,matrix,(600,700))
    img = wrap_image.astype(np.uint8)
    return  img

# def scan_effect(img):
#     dilated_img = cv2.dilate(img, np.ones((7, 7), np.uint8))
#     bg_img = cv2.medianBlur(dilated_img, 15)
#     diff_img = 255 - cv2.absdiff(img, bg_img)
#     norm_img = diff_img.copy()
#     cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#     _, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
#     cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret, th = cv2.threshold(img,0,  255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     denoised = cv2.fastNlMeansDenoising(th, 11, 31, 9)
#     th = cv2.adaptiveThreshold(blur, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,3)
#     return th


def reorder(points):
    points_copy = np.zeros((4,1,2), dtype = np.int32)
    sum_p = points.sum(1)
    points_copy[0] = points[np.argmin(sum_p)]
    points_copy[3] = points[np.argmax(sum_p)]
    diff = np.diff(points,axis = 1)
    points_copy[1] = points[np.argmin(diff)]
    points_copy[2] = points[np.argmax(diff)]
    return points_copy

def process_image(image):
    im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.uint8)
    img_dilate = cv2.dilate(im_gray, kernel, 5)
    im_blur = cv2.GaussianBlur(img_dilate,(3,3),0)
    blur= cv2.erode(im_blur,kernel,iterations =5)
    im_canny = cv2.Canny(blur,100,255)
    cont,h=cv2.findContours(im_canny.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    j=0
    scanned = return_one_contour(cont, image)
    return scanned

if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument("pdf_name", help = "Name of the pdf. Must include .pdf extension", type = str)
    parser.add_argument("--o", help = "Rotate the display screen by 90 degree anti-clockwise (y/n)", default = "n",type = str)
    parser.add_argument("--ps", help = "Apply Perspective Transfrom and scan effect (y/n)",default = "y",type = str)
    args = parser.parse_args()
    if((args.o== 'y' or args.o=='n') and (args.ps== 'y' or args.ps=='n')):
        basic(args.pdf_name, args.o, args.ps)
    else:
        print("Invalid argument")
