It is a simple OpenCV based program which sends image from your phone to your PC, scans it and creates a pdf. It uses IP Webcam application to connect your phone camera to your PC. It is available [here](https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en&gl=US) .

<br/> 

Before using this you need to change the url in the [Scanner.py](https://github.com/anju-chhetri/Mobile-To-PC-Scanner/blob/master/Scanner.py) with your camera url. 

``` 
url = 'Your camera IP/video' 
``` 

To run the program  

``` 
python3 Scanner.py your_pdf_name.pdf 
``` 

To apply perspective transform the document in your image should have clear four edges. If the four edges are not detected  properly in your document, scan effect is applied to your whole image without any perspective transform. 

<br/> 

Optional arguments can be passed to turn off the perspective transform, scan effect and orientation. 

``` 
python3 Scanner.py your_pdf_name.pdf --o y --ps n 
``` 
![](https://github.com/anju-chhetri/Mobile-To-PC-Scanner/blob/master/IMAGE/scanner.jpg)

Required Libraries:
* [opencv-python](https://docs.opencv.org/4.5.2/d6/d00/tutorial_py_root.html)
* [Pillow](https://pillow.readthedocs.io/en/stable/reference/Image.html)
* [numpy](https://numpy.org/install/)

Space bar is used to take image(multiple images can be taken) and we can press 'q' to terminate the program. 
