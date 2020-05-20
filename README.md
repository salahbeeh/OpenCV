# OpenCV
In this document, I will go throgh my journy with OpenCV staring from noob to face and eyes detection which still basics but I consider it as the first step to master OpenCV with Python.

![Frist step](http://www.hotel-r.net/im/hotel/gb/first-step-5.jpeg)


## Installing OpenCV
* for linux users click [here](https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/)
* for window users click [here](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html)
* for Mac users click [here](https://www.pyimagesearch.com/2018/08/17/install-opencv-4-on-macos/)

## Resources

* Homepage: https://opencv.org
    * Courses: https://opencv.org/courses
* Docs: https://docs.opencv.org/master/
* Q&A forum: http://answers.opencv.org
* Issue tracking: https://github.com/opencv/opencv/issues

## Introduction to OpenCV
OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library. OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in the commercial products. Being a BSD-licensed product, OpenCV makes it easy for businesses to utilize and modify the code.

The library has more than 2500 optimized algorithms, which includes a comprehensive set of both classic and state-of-the-art computer vision and machine learning algorithms. These algorithms can be used to detect and recognize faces, identify objects, classify human actions in videos, track camera movements, track moving objects, extract 3D models of objects, produce 3D point clouds from stereo cameras, stitch images together to produce a high resolution image of an entire scene, find similar images from an image database, remove red eyes from images taken using flash, follow eye movements, recognize scenery and establish markers to overlay it with augmented reality, etc. OpenCV has more than 47 thousand people of user community and estimated number of downloads exceeding 18 million. The library is used extensively in companies, research groups and by governmental bodies.

Along with well-established companies like Google, Yahoo, Microsoft, Intel, IBM, Sony, Honda, Toyota that employ the library, there are many startups such as Applied Minds, VideoSurf, and Zeitera, that make extensive use of OpenCV. OpenCV’s deployed uses span the range from stitching streetview images together, detecting intrusions in surveillance video in Israel, monitoring mine equipment in China, helping robots navigate and pick up objects at Willow Garage, detection of swimming pool drowning accidents in Europe, running interactive art in Spain and New York, checking runways for debris in Turkey, inspecting labels on products in factories around the world on to rapid face detection in Japan.

It has C++, Python, Java and MATLAB interfaces and supports Windows, Linux, Android and Mac OS. OpenCV leans mostly towards real-time vision applications and takes advantage of MMX and SSE instructions when available. A full-featured CUDAand OpenCL interfaces are being actively developed right now. There are over 500 algorithms and about 10 times as many functions that compose or support those algorithms. OpenCV is written natively in C++ and has a templated interface that works seamlessly with STL containers.

## Intro
* First thing first, we need to understand that the core of image and video analysis is frames as videos consist of static images called frames.

* It's important to simplyify the image or the video source as much as possible and that can be done by converting the source into grayscale,filtring and many other ways depends on the task you wish to preform.

### 01.Loading images and videos

**1.1 Loading images**

This is the most basic operation to be done by **OpenCV** using images or videos or video webcams,and handling frames from a video is identical to handling for images.

let's see an example :

```python
import cv2

# careful with the identions

"""
# reading an image can be done in opencv with many ways:
# IMREAD_GRAYSCALE , we can basicly just replace it with 0
# IMREAD_COLOR , or just replace it with 1
 """

# reading a randam image from my labtop using opencv
image = cv2.imread('programming.jpg',cv2.IMREAD_GRAYSCALE)

# showing the image i just read
cv2.imshow('Image' , image)

# litarly waiting for any button to be presed
cv2.waitKey(0)

cv2.destroyAllWindows()
```

First, we import OpenCV module and to read an image we use cv2.imread(image path , alpha channel), this function simply takes the path of the image which you wanna load, and in the second prameter we pass an alpha channel we wanna apply on the image, by  default is going to be IMREAD_COLOR, which is color without any alpha channel, there are some other channels like IMREAD_GRAYSCALE for example.

Once loaded, we use cv2.imshow(title,image) to show the image. From here, we use the cv2.waitKey(0) to wait until any key is pressed. Once that's done, we use cv2.destroyAllWindows() to close everything.

Here is the image we loaded

<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/programming.jpg?raw=true">
</p>


and the output for the former code will be like:
<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/programming-gray.jpg?raw=true">
</p>


You don't need to write something like IMREAD_GRAYSCALE at each time you wanna convert the image using a spacific alpha channel, insted you can can also use simple numbers. You should be familiar with both options, so you understand what the person is doing. For the second parameter, you can use -1, 0, or 1. Color is 1, grayscale is 0, and the unchanged is -1.

let's try it with an example:
```python
image = cv2.imread('programming.jpg',0)
```
and it should give the same output as the pervious one.

the comming part is (optional), if you want to save the image you just converted from BGR (in OpenCV it's called ==BGR== insted of \~\~RGB\~\~ but they are the same) to grayscale we can write the next line of code.
```python
cv2.imwrite('programming-gray.jpg',image)
```
easy enough!

**1.2 Loading videos**

It's somehow the same thing with images exept the method used but the same concept as handling frames from a video is identical to handling for images.

```python
import cv2

# 0 means that you are capturing from the first cam
# 1 means that you are capturing from the second cam
# if you have 2 or whatever
video = cv2.VideoCapture(1)

# this the the codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')

while True:
    # reading the rgb video
    ret, frame = video.read()
    # reading the rgb video and convert it into grayscale
    gray = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)
    # well, writing means 'saving' pretty much simple
    out.write(frame)

    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)

    # the video will wait for any key to be pushed
    # but with '&' it will wait for the the hecadecimal of the key q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()
```
cv2.VideoCapture(0).0 will return video from the first webcam on your computer,in case if you have another webcam you would need to change 0 to 1 or 2 or whatever how many cam you have connected to your computer.

so we pass the video feed to an infinite loop and read() the feed into varible frame, and as I said before it's important to simplyify the image or the video source as much as possible and that can be done by converting the source into grayscale and this is exactly doing here:

```python
gray = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)
```
here I basicly take the feed which I saved in *frame* variable and convert it into grayscale and save the grayscale version to *gray* variable.

In order to save a the video you are capturing from your webcam, you need to define the video codec before intering the loop to save the video like here:
```python
fourcc = cv2.VideoWriter_fourcc(*'XVID')
```
then use **out.write()** to output the frame and after the loop ends we just need to release the Resources which are the webcam by **video.release()** and release the output via **out.release()** then we destroy all Windows.



### 02.Drawing and Writing on Image
usaully to draw shapes on images when it comes to python most of us will choose to use **matplotlib** insted of OpenCV but since OpenCV has this feature then it will be more optimal to use it, it's something similar to native/react native development programming as drawing with OpenCV is the native and using matplotlib is react native, I hope I made it clear enough.

**2.1 Drawing a line**

```python
import cv2

# just reading the image
image = cv2.imread('programming.jpg',cv2.IMREAD_COLOR)

# drawing a line
# cv2.line(image,srart point,end point,line_color, line width)
cv2.line(image,(5,5),(60,60),(255,255,255),5)

cv2.imshow('line',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
So first thing first, as usaule we import OpenCV model and read the image we wanna draw the line on and then simply we use cv2.line an we pass the parameters which are:

*cv2.line(image,srart point,end point,line_color, line width)*

* image = the varible you saved the image in.
* start point = which pixel where it should start drawing the line.
* end point = the pixel where it will stop drawing the line.
* line color = the color that the line is going to appear like. note : colors in OpenCV are a mix of BGR ( Blue , Green , Red ).
* line width = a scale of the line thickness.


This how to draw a line by OpenCV, but is it the only shape that you can draw by OpenCV?! no it's not, we can draw a punsh of other shapes a cricle for example.

**2.2 Drawing a circle**

Here is how to draw a cricle in OpenCV:

```python
# drawing a circle
# cv2.circle(image , center, ridus,circle_color, width{if it -1 it will fill the circle})
cv2.circle(image, (200,200), 30, (255,0,0),3)
```
unlike lines, cirles does not work the same way with starting point and an ending point, you need to specifiy where to put the center of it and to spacifiy the ridus then to start drawing.
let's disscus the parameters of the cricle fun:
* image = the variable you saved the image in.
* center = the pixle where the cricle located.
* ridus = the ridus of the cricle to be drawn.
* cricle color = the color of the line used in the cricle drawing.
* width = the thickness of the line used to draw the cricle. note: in case of width = -1, it means the cricle will be filled with the color you choosed.


**2.3 Drawing a rectangle**
```python
# drawing a rectangle
# cv2.rectangle(image,highest lift point ,lowest right point,rectangle_color, rectangle width)
cv2.rectangle(image,(100,100) ,(350,350),(0,255,0), 5)
```
In order to draw a rectangle in OpenCV you would need to spacifiy many parameters other than the image you are going to draw on itself, like:
* Highest lift point = if you wishing to draw a rectangle with your mouse you will start with a point this point or this pixle is the highest lift point.
* lowest right point = the point where you will stop drawing the rectangle with your mouse.
* and as usaule the line color and width, I don't think that they need explanation anymore.

**2.4 Drawing a polygone**

Simply what is polygones? well,it's a plane shape (two-dimensional) with straight sides.

Examples: triangles, rectangles and pentagons.

(Note: a circle is not a polygon because it has a curved side)
also it hasn't to be a complete shape it could be a collection of connected lines as long as it has a straight sides like the following figure.
<p align="center">
  <img src="https://warmaths.fr/MATH/geometr/Polygone_fichiers/image008.jpg">
</p>



```python
# polygones consist of points and you gonna connect those dots
# you have the option to close the polygone
# the points for the polygone
points = np.array([[50,45],[60,47],[69,78],[39,85],[82,64]],np.int32)
points = points.reshape((-1,1,2))
#cv2.polylines(image, [points],wither or not to connect the first point to  the last one, color, line width)
cv2.polylines(image,[points],True,(0,255,255),6)
```
In order to draw a polygon we need to spacifiy the points to connet then to pass them to *cv2.polylines()* and it should draw the lines between them and in order to connet the last point to the first one you should make the third prameter *True* as it's about whether or not you want to connct them to have a closed shape.

as if it True it will look somethig like this:
<p align="center">
  <img src="https://warmaths.fr/MATH/geometr/Polygone_fichiers/image010.jpg">
</p>

**2.5 Writing on an image**

All you need to do to write on an image is to spacifiy the font to be written with and what to be written.
```python
# writing on the image
# define a font
font = cv2.FONT_HERSHEY_SIMPLEX
#                            start,font, size,color, space between charcters, alusing
cv2.putText(image,'oh! wow it works',(0,130),font, 1, (100,220,150), 2,cv2.LINE_AA)
```
### 03.Image Operations
**3.1 ROI**

ROI tends to reigon of interest, this defininition is widly used in many areas. For example, in medical imaging, the boundaries of a tumor may be defined on an image or in a volume, for the purpose of measuring its size. so how how can we do an operation on a spacific reigon which has our interst, Well simply we select it:
```python
px = image[100:200,100:200]
 ```
and apply the operation we want.

**3.2 Image arithmetics and Logic**

**Addition** :Well, I was tring the other day to add two images and I was amazed, We can litarly add two images by a the plus sign, For example if I want to add the following 2 images:
<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/1.png">
</p>
<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/2.png">
</p>


```python
import cv2

image1 = cv2.imread('1.png')
image2 = cv2.imread('2.png')

addition = image1 + image2

cv2.imshow('addition', addition)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
we are going to get this output:
<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/addition.png">
</p>

the output is not the optimal, quite missy but it still exists anyway it's not the only addition there we can use the fun *cv2.add()* like:

```python
import cv2

image1 = cv2.imread('1.png')
image2 = cv2.imread('2.png')

add = cv2.add(image1,image2)

cv2.imshow('addition', add)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
and its output will look like:
<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/add.png">
</p>

the output is kinda white as it litarly adds the pixel's values together, For example,(120,150,200) + (10, 150, 200) = 130, 300, 400...translated to (130, 255,255).

**Images Overlaping**
What if we want to overlap 2 images like to make one of them as a victor art or a png file on the other (those who know photoshop will understand those terms if you did't, don't worry, you will learn by an example) like for example we want to put the python logo image on the output of the addition operation.

If I add two images, it will change color. If I blend it, I get an transparent effect. But I want it to be opaque. If it was a rectangular region, I could use ROI as we did in last time. But python logo is a not a rectangular shape. So you can do it with [bitwise](https://docs.opencv.org/4.1.2/d0/d86/tutorial_py_image_arithmetics.html) operations as below:

let's just take a quick look at the images we are working on:

Here is the first one, and yes as you noticed it's the same one we produced last time with the addition operation.

<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/addition.png">
</p>

and here is the image i want to blind it into the addition image.

<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/mainlogo.png">
</p>

```python
import cv2
import numpy as np

image1 = cv2.imread('samples/1.png')
image3 = cv2.imread('samples/mainlogo.png')


# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = image3.shape
ROI = image1 [0:rows,0:cols]

# now we are going to create a mask which is a initial of conversion to the grayscale
image_gray = cv2.cvtColor(image3,cv2.COLOR_BGR2GRAY)
# now we are thresholding
# so any value above 220 will be converted into 255 which is white
# THRESH_BINARY_INV is going to do the inverse so what is lower than 220 will be converted
# into 0 which is black
# add a threshold
ret , mask = cv2.threshold(image_gray, 220 ,255, cv2.THRESH_BINARY_INV)
mask_inv = cv2.bitwise_not(mask)
cv2.imshow('mask_inv', mask_inv)

# Now black-out the area of logo in ROI
img1_background = cv2.bitwise_and(ROI,ROI,mask = mask_inv)
cv2.imshow('img1_background',img1_background)
# Take only region of logo from logo image.
im2_forground = cv2.bitwise_and(image3,image3,mask = mask)
cv2.imshow('im2_forground',im2_forground)
distination = cv2.add(img1_background,im2_forground)
cv2.imshow('distination', distination)
image1[0:rows,0:cols] = distination
cv2.imshow('result', image1)
#cv2.imshow('mask', mask)
cv2.imwrite('samples/transplant.png',image1)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

and here is the output:

<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/transplant.png">
</p>


### 04.Thresholding
Well, what is threshold and thresholding? **Thresholding** is a basic operation where we make any pixle white(255) or black(0) when we hit a spasific pixel value this pixel value called **Threshold** i.e. if we assumed that the threshold is (150), it means that any value below it will be  automatically converted to black (0) and any value above(150) will be converted into white (255) and so on, i call it a manual grayscaling.

there is alot of thresholding types we will go throgh some of them, the following image is a book page

<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/bookpage.jpg">
</p>

but it's really dim let's try to fix it with some thresholding.

```python
import cv2 as cv

image = cv.imread('bookpage.jpg')

# frist threshold is the binary
# cv2.threshold( image , critical point, converted values, type of threshold)
# the critical point is where any thing above it will be converted to the (converted value)
retval, threshold = cv.threshold(image, 12, 255, cv.THRESH_BINARY)
cv.imshow('binary threshold', threshold)
```
**4.1 binary Thresholding**

so the first Threshold type is the binary and from its name the output value is either black(0) or white(255).

let's take a quick look on the parameters:

cv2.threshold( image , critical point, converted values, type of threshold)

  * image = the variable where you keep the image in
  * critical point = the value where to convert to black or white depending on if the pixel is lower or above it.
  * converted value = the value to be converted to if the pixle value is above the threshold.
  * type of the threshold = here we used the binary one.

let's take a quick look at the output.

<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/binary%20threshold.jpg">
</p>

oh wow!, wait it's not supposed to look like that, it's supposed to look black and white, well yeah it's quite right but have passed a colored image to the threshold and any pixel is above 12 will be white but what about those which below it. yeah now it's more clear they will not be changed and that is what caused those colors.



Have you ever thought of grayscaling is a thresholding? well yeah but not quite right but it might help with our situation. let's try it and see it's output.
```python
# let's try convert the original image to grayscale to see maybe it might help
grayscale = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
retval, threshold2 = cv.threshold(grayscale , 12, 255, cv.THRESH_BINARY)
cv.imshow('grayscaled', threshold2)
```
and here is the output:
<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/grayscaled%20binary.jpg">
</p>

Mmm, not bad but still missy and hard to read, anyway we were just trying.

**4.2 [Adaptive Thresholding](https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=adaptivethreshold)**

let's try this one on our grayscaled image.

```python
# adaptive threshold
# there's 2 types of adaptive threshold (mean,gaussian)
# cv.AdaptiveThreshold(src, dst, maxValue, adaptive_method=CV_ADAPTIVE_THRESH_MEAN_C, thresholdType=CV_THRESH_BINARY, blockSize=3, param1)
mean_adaptive = cv.adaptiveThreshold(grayscale,255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 115, 1)
cv.imshow('mean_adaptive',mean_adaptive)
cv.imwrite('mean_adaptive.jpg',mean_adaptive)

gaussian_adaptive = cv.adaptiveThreshold(grayscale,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 115, 1)
cv.imshow('gaussian_adaptive',gaussian_adaptive)
cv.imwrite('gaussian_adaptive.jpg',gaussian_adaptive)
```
In simple thresholding, the threshold value is global, i.e., it is same for all the pixels in the image. Adaptive thresholding is the method where the threshold value is calculated for smaller regions and therefore, there will be different threshold values for different regions.

Opencv supports two types of adaptive thresholding algorithm for image processing namely gaussian and adaptive mean. The basic difference between these two algorithms is that in adaptive mean to calculate the threshold value for a sub region we make use of mean and for gaussian we use weighted mean.

Anyway aren't you currious how the output of them both will look like! I will show the mean then the gaussian.


<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/mean_adaptive.jpg">
</p>

<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/gaussian_adaptive.jpg">
</p>

## 05.Color Filtering

In this chapter, I'm attempting to filter a color from the whole image or video, as only stuff with this color will show up and others won't, have you watched how marvel makes its movies? it's the same idea here with the green background stuff.

First thing first, I tryied to filter the color based on the BGR colors but that was hard and had many problems, but I found a tutorial how to filter by [HSV](https://en.wikipedia.org/wiki/HSL_and_HSV) Hue for color, saturation for the strength of the color, and value for light.
<p align="center">
  <img src="https://i.pinimg.com/564x/ad/b5/16/adb5169fcd9530d5010f43a1258223c6.jpg">
</p>



```python
import cv2 as cv
import numpy as np


video = cv.VideoCapture(0)

while True:
    _ , frame = video.read()
    hsv = cv.cvtColor(frame ,cv.COLOR_BGR2HSV)

    # hsv hue sat Value
    lower  = np.array([130,0,0])
    higher = np.array([200,255,255])

    mask   = cv.inRange(hsv,lower,higher)
    result = cv.bitwise_and(frame,frame,mask =mask)
    cv.imshow('result', result)

      k = cv.waitKey(5) & 0xFF
      if k == 27:
          break

  video.release()
  cv.destroyAllWindows()
```
Here I'm trying to target a pink shirt that's why I wrote those spacific values, please feel free to change them depending on what color you are trageting, when you do you won't be satisfied with the results as some nosie will appear, don't worry this is the topic of the next chapter.

## 06.Blurring and Smoothing

**6.1 Smoothing**

Smoothing is an afective way to remove background noise, and where we do a sort of averaging per block of pixels. In our case, let's do a 15 x 15 square, which means we have 225 total pixels.

```python
import cv2 as cv
import numpy as np


video = cv.VideoCapture(-1)

while True:
    _ , frame = video.read()
    hsv = cv.cvtColor(frame ,cv.COLOR_BGR2HSV)

    # hsv hue sat Value
    lower  = np.array([130,0,0])
    higher = np.array([200,255,255])

    mask   = cv.inRange(hsv,lower,higher)
    result = cv.bitwise_and(frame,frame,mask =mask)


    # now if we wanna get rid of the noise in the background
    # let's try smoothing with averaging
    kernel   = np.ones((15,15), np.float32)/225
    smoothed = cv.filter2D(result, -1, kernel)
    cv.imshow('smoothed', smoothed)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

video.release()
cv.destroyAllWindows()
```
**6.2 Blurring**

There are many types of Blurring, but I only tryed two of them, [Gaussian Blur](https://en.wikipedia.org/wiki/Gaussian_blur), and [Median Blur](https://www.tutorialspoint.com/opencv/opencv_median_blur.htm).

```python
# let's try gaussian bluring
  blur = cv.GaussianBlur(result,(15,15),0)
  cv.imshow('gaussian', blur)

  # let's try median
  median = cv.medianBlur(result,15)
  cv.imshow('median', median)

  #cv.imshow('mask',mask)
  cv.imshow('result', result)
  ```

  ## 0.7 Morphological Transformations

  [Morphological transformations](https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html) are some simple operations based on the image shape. It is normally performed on binary images. It needs two inputs, one is our original image, second one is called structuring element or kernel which decides the nature of operation. Two basic morphological operators are Erosion and Dilation. Then its variant forms like Opening, Closing, Gradient etc also comes into play.

  **7.1 Erosion & Dilation**

  The basic idea of erosion is just like soil erosion only, it erodes away the boundaries of foreground object (Always try to keep foreground in white). So what it does? The kernel slides through the image (as in 2D convolution). A pixel in the original image (either 1 or 0) will be considered 1 only if all the pixels under the kernel is 1, otherwise it is eroded (made to zero).

  But Dilation is just opposite of erosion. Here, a pixel element is '1' if atleast one pixel under the kernel is '1'. So it increases the white region in the image or size of foreground object increases. Normally, in cases like noise removal, erosion is followed by dilation. Because, erosion removes white noises, but it also shrinks our object. So we dilate it. Since noise is gone, they won't come back, but our object area increases. It is also useful in joining broken parts of an object.

```python
import cv2 as cv
import numpy as np


video = cv.VideoCapture(-1)

while True:
    _ , frame = video.read()
    hsv = cv.cvtColor(frame ,cv.COLOR_BGR2HSV)

    # hsv hue sat Value
    lower  = np.array([130,0,0])
    higher = np.array([200,255,255])

    mask   = cv.inRange(hsv,lower,higher)
    result = cv.bitwise_and(frame,frame,mask =mask)

    kernel  = np.ones((5,5),np.uint8)
    erosion = cv.erode(mask,kernel,iterations = 1)
    dilation = cv.dilate(mask,kernel,iterations = 1)
    cv.imshow('erosion', erosion)
    cv.imshow('dilation', dilation)
    k = cv.waitKey(5) & 0xFF
      if k == 27:
          break

video.release()
cv.destroyAllWindows()
```
Do you know noise types? here, I'll explain. Well we call noise "false", For example, The last attempt we did we were trying to filter by color any pink object and only show it. but we got some noise, nosie on the pink item itself, the item were are trying to filter and those noises we call "False Negatives", also we detected some noise in the background and those we call "False positives", to know more about the False positives and False Negatives you can check this [link](https://en.wikipedia.org/wiki/False_positives_and_false_negatives). And to deal with those we use Opening and Closing.

**7.2 Opening & Closing**

Opening is just another name of erosion followed by dilation. It is useful in removing noise, as we explained above. Closing is reverse of Opening, Dilation followed by Erosion. It is useful in closing small holes inside the foreground objects, or small black points on the object.

```python
mport cv2 as cv
import numpy as np


video = cv.VideoCapture(-1)

while True:
    _ , frame = video.read()
    hsv = cv.cvtColor(frame ,cv.COLOR_BGR2HSV)

    # hsv hue sat Value
    lower  = np.array([130,0,0])
    higher = np.array([200,255,255])

    mask   = cv.inRange(hsv,lower,higher)
    result = cv.bitwise_and(frame,frame,mask =mask)

    kernel  = np.ones((5,5),np.uint8)
        # opening removes the false positives which are the noise in the background
    opening = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel)
    # closing removes the false negatives which are mistaken detectied in the object we tring to filter
    closing = cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel)

    cv.imshow('opening', opening)
    cv.imshow('closing', closing)

    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

video.release()
cv.destroyAllWindows()
```
**7.3 Morphological Gradient**

It is the difference between dilation and erosion of an image.

```python
gradient = cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)
```
## 08.Edge Detection

Well, This chapter needs you know some Image processing terms, don't worry, I will try my best to summrize those terms.

**Edge** is a boundary between two regions with relatively distinct gray level properties, Edges are pixels where the brightness function changes abruptly, Edge detectors are a collection of very important local image pre-processing methods used to locate (sharp) changes in the intensity function.

Edge detection steps: -
1. Filtering (filter the noise to improve the performance of the edge detector).
2. Enhancement (emphasize that the pixels having change in their local intensity).
3. Detection – Identify edges – thresholding.
4. Localization (locating the edge and estimate the orientation).

**METHODS OF EDGE DETECTION**
* First derivative (gradient methods)
  - Roberts
    - Detects horizontal and vertical edges only.
  - **Sobel**
    - Detect (horizontal, vertical, someangels) edges.
    - Detect thicker edges.
    - Less sensitive to noise.
    - Most popular detector.
    - Uses 2 masks.
    - Focus on the edges in the center.
  - Prewitt
    - Use 8 masks.
    - Similar to sobel.
    - Use angle 45.
* Second Order Derivative Methods:
  - **Laplacian**
    - very sensitive to noise
  - Laplacian of Gaussian(LOG)
  - Difference of Gaussian(DOG)
* **Canny Edge Detector**:
  - Steps:
    1. filter the noise “use the Gaussian filter, choose the width carefully”.
    2. find the edge strength (take the gradient “Roberts and sobel”).
    3. find edge direction.
    4. Non-maxima suppression – trace along the edge direction and suppress any pixel value not considered to be an edge. Gives a thin line for edge.
    5. Use double / hysterisis thresholding to eliminate streaking.

so Edge detectors are three 2 levels gradiant and canny. We will start with gradiant.

```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    # frist edge detector
    laplacian = cv2.Laplacian(frame,cv2.CV_64F)
    # sobel detector is complex than laplacian as it require 2 mask
    # one is horizontal and the other is virtical
    # cv2.CV_64F is, that's the data type. ksize is the kernel size. We use 5, so 5x5 regions are consulted.
    sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)


    cv2.imshow('Original',frame)
    cv2.imshow('Mask',mask)
    cv2.imshow('laplacian',laplacian)
    cv2.imshow('sobelx',sobelx)
    cv2.imshow('sobely',sobely)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

```
Have you wondered what is *CV_64F*, I had a hard time with it myself, Well, In OpenCV the main matrix class is called *Mat* and is contained in the OpenCV-namespace cv. This matrix is not templated but nevertheless can contain different data types. These are indicated by a certain type-number. Additionally, OpenCV provides a templated class called Mat_, which is derived from Mat. CV_64F is a data type for simple grayscale image and has only 1 channel.

More generally, type name of a Mat object consists of several parts. Here's example for CV_64FC1:
* CV_ - this is just a prefix
* 64 - number of bits per base matrix element (e.g. pixel value in grayscale image or single color element in BGR image)
* F - type of the base element. In this case it's F for float, but can also be S (signed) or U (unsigned)
* Cx - number of channels in an image as I outlined earlier.


Now, Let's try [Canny](https://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html):
```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    # the last edge detector is canny ( the optimal one)
    edges = cv2.Canny(frame,100,200)


    cv2.imshow('Original',frame)
    cv2.imshow('Mask',mask)
    cv2.imshow('Edges',edges)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
```

## 09.Object Matching
The idea here is to find identical regions of an image that match a template we provide, giving a certain threshold. For exact object matches, with exact lighting/scale/angle, this can work great. An example where these conditions are usually met is just about any GUI on the computer.

so we will work on a cat template, I hope you are a cat lover.

Main image:
<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/catfamily.jpg">
</p>

Template:
<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/cat.jpg">
</p>

Well, It'a obvious that there is no any cat identical to our template cat so we do have a threshold option, where if something is maybe an 80% match for example, then we say it's a match.

```python
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np

image = cv.imread('catfamily.jpg')
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
# cv2.imread(path, flag)
# Parameters:
# path: A string representing the path of the image to be read.
# flag: It specifies the way in which image should be read. It’s default value is cv2.IMREAD_COLOR
""" All three types of flags are described below:

    1- cv2.IMREAD_COLOR: It specifies to load a color image. Any transparency of
       image will be neglected. It is the default flag. Alternatively, we can pass
       integer value 1 for this flag.
    2- cv2.IMREAD_GRAYSCALE: It specifies to load an image in grayscale mode.
       Alternatively, we can pass integer value 0 for this flag.
    3- cv2.IMREAD_UNCHANGED: It specifies to load an image as such including
       alpha channel. Alternatively, we can pass integer value -1 for this flag.
    """
temp = cv.imread('cat.jpg',0)
width, hight = temp.shape[::-1]

result =cv.matchTemplate(gray,temp, cv.TM_CCOEFF_NORMED)
threshold = 0.7

location = np.where(result >= threshold)
# Here, we call res the matchTemplate between the img_gray
# (our main image), the template, and then the matching method
#  we're going to use. We specify a threshold, here 0.7 for 70%.
#  Then we find locations with a logical statement, where the
# res is greater than or equal to 70%.

Finally, we mark all matches on the original image, using the coordinates we found in the gray image:
for point in zip(*location[::-1]):
    cv.rectangle(image, point,(point[0]+width,point[1]+hight),(255,0,255),2)

cv.imshow('image', image)
cv.waitKey(0)
cv.destroyAllWindows()

```
We loaded in both images, converted them to gray. kept the original RGB image, and create a grayscale version because we do all of the processing in the grayscale, then use the same coordinates for labels and such on the color image.

With the main image, we just have the color version and the grayscale version. We load the template and note the dimensions.

after applying the threshold let's see the output:

<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/catmatch.jpg">
</p>

## 10.GrabCut Foreground Extraction

GrabCut algorithm was designed by Carsten Rother, Vladimir Kolmogorov & Andrew Blake from Microsoft Research Cambridge, UK. in their paper, ["GrabCut": interactive foreground extraction using iterated graph cuts](https://dl.acm.org/doi/10.1145/1186562.1015720) . An algorithm was needed for foreground extraction with minimal user interaction, and the result was GrabCut.

How it works from user point of view ? Initially user draws a rectangle around the foreground region (foreground region should be completely inside the rectangle). Then algorithm segments it iteratively to get the best result. Done. But in some cases, the segmentation won't be fine, like, it may have marked some foreground region as background and vice versa. In that case, user need to do fine touch-ups. Just give some strokes on the images where some faulty results are there. Strokes basically says *"Hey, this region should be foreground, you marked it background, correct it in next iteration"* or its opposite for background. Then in the next iteration, you get better results.

Example:

<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/me.jpg">
</p>

Here I will try to cut my face from the image as it's the forground to be cut. So as usalle I'll import the modules I will use, will load the image above to work on, and will create a mask.

```python
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('samples/me.jpg')
mask = np.zeros(img.shape[:2],np.uint8)
```
Then I'll specify the background and foreground model, which is used by the algorithm internally. Then to spacify the rectangle to be cut where rectangle = (start_x, start_y, width, height).

```python
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
# rect = It is the coordinates of a rectangle which includes the foreground object in the format (x,y,w,h)
rect = (700,200,200,300)
```
So far, rect = It is the coordinates of a rectangle which includes the foreground object in the format (x,y,w,h).



```python
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img)
plt.colorbar()
plt.show()
```
 So here we used cv2.grabCut, which took quite a few parameters. First the input image, then the mask, then the rectangle for our main object, the background model, foreground model, the amount of iterations to run, and what mode you are using.

 let's take a quick look on the output:

 <p align="center">
   <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/grubcut%20forground.jpg">
 </p>

 So what happens in background ?

* User inputs the rectangle. Everything outside this rectangle will be taken as sure background (That is the reason it is mentioned before that your rectangle should include all the objects). Everything inside rectangle is unknown. Similarly any user input specifying foreground and background are considered as hard-labelling which means they won't change in the process.
* Computer does an initial labelling depending on the data we gave. It labels the foreground and background pixels (or it hard-labels).
* Now a Gaussian Mixture Model(GMM) is used to model the foreground and background.
* Depending on the data we gave, GMM learns and create new pixel distribution. That is, the unknown pixels are labelled either probable foreground or probable background depending on its relation with the other hard-labelled pixels in terms of color statistics (It is just like clustering).
* A graph is built from this pixel distribution. Nodes in the graphs are pixels. Additional two nodes are added, Source node and Sink node. Every foreground pixel is connected to Source node and every background pixel is connected to Sink node.
* The weights of edges connecting pixels to source node/end node are defined by the probability of a pixel being foreground/background. The weights between the pixels are defined by the edge information or pixel similarity. If there is a large difference in pixel color, the edge between them will get a low weight.
* Then a mincut algorithm is used to segment the graph. It cuts the graph into two separating source node and sink node with minimum cost function. The cost function is the sum of all weights of the edges that are cut. After the cut, all the pixels connected to Source node become foreground and those connected to Sink node become background.
* The process is continued until the classification converges.

## 11.Corner Detection

A corner can be defined as the intersection of two edges. A corner can also be defined as a point for which there are two dominant and different edge directions in a local neighbourhood of the point.

The purpose of detecting corners is to track things like motion, do 3D modeling, and recognize objects, shapes, and characters.

We will do the usales, importing the modules and load the image to be used.

<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/corners.png">
</p>



```python
import numpy as np
import cv2

img = cv2.imread('corners.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
# cv2.goodFeaturesToTrack(image ,num corners to be detected , corner quality, minDistance)
# minDistance = Minimum possible Euclidean distance between the returned corners.
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)
for corner in corners:
    x,y = corner.ravel()
    cv2.circle(img,(x,y),3,255,-1)

cv2.imshow('corner',img)
if cv2.waitKey(0) & 0xff ==27:
    cv2.destroyAllWindows()

```
[cv.goodFeaturesToTrack()](https://docs.opencv.org/master/d4/d8c/tutorial_py_shi_tomasi.html) :It finds N strongest corners in the image by Shi-Tomasi method (or Harris Corner Detection, if you specify it). As usual, image should be a grayscale image. Then you specify number of corners you want to find. Then you specify the quality level, which is a value between 0-1, which denotes the minimum quality of corner below which everyone is rejected. Then we provide the minimum euclidean distance between corners detected.

goodFeaturesToTrack(image ,num corners to be detected , corner quality, minDistance)
minDistance = Minimum possible Euclidean distance between the returned corners.

<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/corner_detected.png">
</p>


## 12.Feature Matching (Homography) Brute Force

We start with the image that we're hoping to find, and then we can search for this image within another image. The beauty here is that the image does not need to be the same lighting, angle, rotation...etc. The features just need to match up.

<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/logos.jpg">
</p>

<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/BMW1.jpg">
</p>


```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('BMW1.jpg',0)
img2 = cv2.imread('logos.jpg',0)
```
Now we're going to use a form of "brute force" matching. We're going to find all features in both images. Then we match these features. We then can draw out as many as we want. Careful though. If you draw say 500 matches, you're going to have a lot of false positives.

```python
# Initiate SIFT detector
orb = cv2.ORB_create()
```

* SIFT = Scale-invariant feature transform
* SIFT detectors allow us to know the matches even if the 2 images we are comparing are not the same or the matched objects are rotated it still find the matches unless the object feature are too small to be matched.

Here, we find the key points and their descriptors with the orb detector.
```python

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
```
This is our BFMatcher object.
```python
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
```
* BFMatche returns the best match.
* NORM_HAMMING Hamming use distance as measurement

* crossCheck which is false by default. If it is true, Matcher returns only those matches with value (i,j) such that i-th descriptor in set A has j-th descriptor n set B as the best match and vice-versa.

```python
matches = bf.match(des1,des2)
# sort matches based on the dictances
matches = sorted(matches, key = lambda x:x.distance)

# drawing the matches
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
plt.imshow(img3)
plt.show()
```
The output will be:

<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/brute%20force%20match.jpg">
</p>

## 13.MOG Background Reduction

Background subtraction (BS) is a common and widely used technique for generating a foreground mask (namely, a binary image containing the pixels belonging to moving objects in the scene) by using static cameras.

As the name suggests, BS calculates the foreground mask performing a subtraction between the current frame and a background model, containing the static part of the scene or, more in general, everything that can be considered as background given the characteristics of the observed scene.

Basiclly, We will read data from videos or image sequences by using cv.VideoCapture, Then, Create and update the background model by using cv.BackgroundSubtractor class, And finally Get and show the foreground mask by using cv.imshow.

``` python
import numpy as np
import cv2

cap = cv2.VideoCapture('samples/vtest.avi')
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

      cv2.imshow('fgmask',frame)
      cv2.imshow('frame',fgmask)


      k = cv2.waitKey(30) & 0xff
      if k == 27:
          break


  cap.release()
  cv2.destroyAllWindows()
```
let's disscus BackgroundSubtractorMOG2 parameters

fgmask	=	cv.BackgroundSubtractorMOG2.apply(	image[, fgmask[, learningRate]]	)

Computes a foreground mask.

Parameters
  * *image*:	      Next video frame. Floating point frame will be used without scaling and should be in range [0,255].
  * *fgmask*:	      The output foreground mask as an 8-bit binary image.
  * *learningRate*: The value between 0 and 1 that indicates how fast the background model is learnt. Negative parameter value makes the algorithm to use some automatically chosen learning rate. 0 means that the background model is not updated at all, 1 means that the background model is completely reinitialized from the last frame.

<p align="center">
    <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/vtest_fig.png">
</p>

<p align="center">
    <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/frame_mog.png">
</p>

## 14.Haar Cascade Object Detection Face & Eye

Object Detection using Haar feature-based cascade classifiers is an effective object detection method proposed by Paul Viola and Michael Jones in their paper, "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001. It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.

Here we will work with face detection. Initially, the algorithm needs a lot of positive images (images of faces) and negative images (images without faces) to train the classifier. Then we need to extract features from it. For this, Haar features shown in the below image are used. They are just like our convolutional kernel. Each feature is a single value obtained by subtracting sum of pixels under the white rectangle from sum of pixels under the black rectangle.

we're going to discuss object detection with Haar Cascades. We'll do face and eye detection to start. In order to do object recognition/detection with cascade files, you first need cascade files. For the extremely popular tasks, these already exist. Detecting things like faces, cars, smiles, eyes, and license plates for example are all pretty prevalent.

First, I will show you how to use these cascade files, then I will show you how to embark on creating your very own cascades, so that you can detect any object you want, which is pretty darn cool!

We will use a [Face cascade](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) and [Eye cascade](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml). You can find a few more at the root directory of Haar cascades. Note the license for using/distributing these Haar Cascades.

Now we begin our typical loop, the only new thing here is the creation of faces. For more information, visit the documentation for the detectMultiScale functionality. Basically, it finds faces! We also want to find eyes, but, in a world of false positives, wouldn't it be prudent to logically make it so that we only find eyes in faces? Let's hope we're not looking for eyes that aren't in faces! In all seriousness, "eye detection" probably wouldn't find an eyeball laying around. Most eye detection uses the surrounding skin, eye lids, eye lashes, and eye brows to also make the detection. Thus, our next step is to break down the faces first, before getting to the eyes:

```python
import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)

while 1:
    # returns the feed from the camera
    ret, img = cap.read()
    # convert the feed into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # HaarDetectObjects(image, scale_factor=1.3, min_neighbors=5)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    for (x,y,w,h) in faces:
        # drawing a rectangle around the face
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # writing on the first pixel of the rectangle
        cv2.putText(img,'Face',(x,y), font, 0.5, (11,255,255), 2, cv2.LINE_AA)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        # HaarDetectObjects(image)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            # drawing a rectangle around the eyes
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            # writing on the first pixel of the rectangle
            cv2.putText(img,'Eye',(ex+x,ey+y), font, 0.5, (11,0,255), 2, cv2.LINE_AA)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
```
I tried on a video of mine, let's see the output:

<p align="center">
  <img src="https://github.com/salahbeeh/OpenCV/blob/master/samples/face%26eyes%20detected.png">
</p>
