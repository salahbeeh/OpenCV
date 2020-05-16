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

![input](https://github.com/salahbeeh/OpenCV/blob/master/samples/programming.jpg?raw=true)

and the output for the former code will be like:

![output](https://github.com/salahbeeh/OpenCV/blob/master/samples/programming-gray.jpg?raw=true)


You don't need to write something like IMREAD_GRAYSCALE at each time you wanna convert the image using a spacific alpha channel, insted you can can also use simple numbers. You should be familiar with both options, so you understand what the person is doing. For the second parameter, you can use -1, 0, or 1. Color is 1, grayscale is 0, and the unchanged is -1.

let's try it with an example:
```python
image = cv2.imread('programming.jpg',0)
```
and it should give the same output as the pervious one.

the comming part is (optional), if you want to save the image you just converted from BGR (in OpenCV it's called BGR insted of RGB but they are the same) to grayscale we can write the next line of code.
```python
cv2.imwrite('programming-gray.jpg',image)
```
easy enough!
