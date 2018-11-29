## Advanced Lane Finding Project ##

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/camera_cal.png "Undistorted"
[undistortedimage]: ./output_images/undistorted3.jpg "undistortedimage"
[actualimage]: ./test_images/test4.jpg "actual image"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/final_process.jpg "Binary Example"
[image4]: ./output_images/wraped4.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[colortransformed]: ./output_images/gradcolor4.jpg "color transformed image"
[outputofline]: ./output_images/lines4.jpg "output of line function"
[finaloutputimage]: ./output_images/final_output4.jpg "final output image"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

```python
def get_image_points(image):
    image_size = (image.shape[1],image.shape[0])
    objectpoints  = np.zeros((6*9,3),np.float32)
    objectpoints[:,:2] =  np.mgrid[0:9, 0:6].T.reshape(-1,2)
    ret,pointsonimage = cv2.findChessboardCorners(image,(9,6),flags = cv2.CALIB_CB_ADAPTIVE_THRESH)
    if ret == True:
        cv2.drawChessboardCorners(image,(9,6),pointsonimage,ret)
        return ret,objectpoints,pointsonimage
    else:
        return False,objectpoints,pointsonimage

def calibrate_camera(image_path=None):
    images = glob.glob('camera_cal/calibration*.jpg')
    objectpoints_=[]
    pointsonimage_ = []
    for index,image in enumerate(images):
        image_ = mpimg.imread(image)
        ret,objectpoints,pointsonimage = get_image_points(image_)
        if ret == True:
            objectpoints_.append(objectpoints)
            pointsonimage_.append(pointsonimage)
                                  
    test_image = mpimg.imread(image_path)
    image_size = (test_image.shape[1],test_image.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectpoints_, pointsonimage_, image_size,None,None)
    dst = cv2.undistort(test_image, mtx, dist, None, mtx)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    ax1.imshow(test_image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(dst)
    ax2.set_title('Undistorted and Warped Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    return mtx,dist
    
mtx,dist = calibrate_camera('camera_cal/TestImage.jpg')
```
![alt text][image1]

Steps for this pipeline are as follows:-

1. We found the corners for the chess board using the following function.This function returns the corner points of the image which is further used for calibrating camera in later steps.The *CALIB_CB_ADAPTIVE_THRESH* helps us just feeding the image instead it converting to grayscale and then feeding it.

```python
 cv2.findChessboardCorners(image,(9,6),flags = cv2.CALIB_CB_ADAPTIVE_THRESH)
 
```
2. We use minimum 10 images to gather the corner points on the images.These number of corner helps us getting a better camera matrix and then for further process.

3. We then feed these corner points to camera calibration functions to obtain the camera matrix and distortion coefficient.These two parameters helps us undistorting the distorted images.

```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectpoints_, pointsonimage_, image_size,None,None)
```

4. The Following function undistorts the image using the camera matrix and the distortion coefficent .

```python
dst = cv2.undistort(test_image, mtx, dist, None, mtx)
```

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][actualimage] 
**Undistored**
![alt text][undistortedimage]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of gradient and channel thresholding.
I used sobel gradient along x axis and for channel thresholding I changed RGB to HLS and then set threshold of S and L channel

```python
def absgradient(image,thres=(20,100)):
    image_ = np.copy(image)
    image_gray = cv2.cvtColor(image_,cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(image_gray,cv2.CV_64F,1,0)
    abs_image = np.absolute(sobelx)
    scaled = np.uint8(255*abs_image/np.max(abs_image))
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled > thres[0]) & (scaled <= thres[1])] = 1
    return binary_output

def HLSprocess(image,thres=(170,255)):
    hlsimage = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    hlsimage = hlsimage[:,:,2]
    binary_output = np.zeros_like(hlsimage)
    binary_output[(hlsimage > thres[0]) & (hlsimage <= thres[1])] =1
    return binary_output

def l_channel(image,thres=(150,255)):
        hlsimage = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
        hlsimage = hlsimage[:,:,1]
        binary_output = np.zeros_like(hlsimage)
        binary_output[(hlsimage > thres[0]) & (hlsimage <= thres[1])] =1
        return binary_output

test_image_ = mpimg.imread('output_images/undistorted1.jpg')
test_image_grad = absgradient(test_image_)
test_image_s = HLSprocess(test_image_)
test_image_l = l_channel(test_image_)
test_binary_ = np.zeros_like(test_image_l)

##Combining all of them ###
test_binary_[((test_image_l == 1) & (test_image_s == 1)) | (test_image_grad == 1)] =1
```
Example image.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

For Prespective I took 4 points on the undistorted image (Source Points) and 4 Points that would be the destination points after prespective transform, using following function.
```python
src_corners = np.float32([[235,665],[1130,665],[555,480],[765,480]])
dest_corners = np.float32([[235,665],[1130,665],[285,0],[1130,0]])
M = cv2.getPerspectiveTransform(src_corners, dest_corners)
###Here M_Inv is for the inverse prespective transform 
M_inv = cv2.getPerspectiveTransform(dest_corners,src_corners)

def perspective_transform(image):
    warped = cv2.warpPerspective(image, M, (image.shape[1],image.shape[0]))
    return warped
```
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image

**Actual Image**
![alt text][actualimage]

**Warped Image**
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

##### Steps #####
1. To Identify the edges of the warped image.I employed the steps of point 2 and got the following output.

![alt text][colortransformed]

2. Next using the sliding window approach I marked all the boxes that are having lanes enclosed in them.The window correct itself depending upon the previous window. 

3. After finding all the windows enclosing the left and the right lane we can find the all the pixels contained in them. We feed those pixel points to the ployfit of degree 2 to obtain a line that runs though the mid of the pixel and hence enclosing the lane. Here the output after the previous two steps.

Here the *green* lane denotes the enclosed lane and the boxes denotes the pixel that belong to those lanes.Here the blue denotes the left lane and the red the right lane.

![alt text][outputofline]

The following code engulfs these previous steps [code](http://localhost:8888/notebooks/Final.ipynb#Lane-Line-detection)
#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The Radius of curvature and the position of vehicle from the center has been calculated as follows:-

1.  Radius was calculated by 

 `R = ((1 + (2Ay+B)^2)^1.5) / |2A|`
 
2. But this formulae returns us the radius in pixel values So to covert it into meter I have considered meter/pixel for particular direction 
```python
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```  
Code for Radius is :-

```python
# Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
#     print(left_curverad, 'm', right_curverad, 'm')
    radius = (left_curverad+right_curverad)//2 
``` 
3. For distance from the center.I have assumed the center of the lane to be the center of the image. Now the using the fit equation i found out the left lane position and right lane position and their average gave me the cars lane position.
Difference between the center of the lane and cars lane position gives us the position of the car from the center.

    code for this function follows as 

    ```python
    car_position = image.shape[1]/2
    left_fit_center = left_fit[0]*image.shape[0]**2 + left_fit[1]*image.shape[0] + left_fit[2]
    right_fit_center = right_fit[0]*image.shape[0]**2 + right_fit[1]*image.shape[0] + right_fit[2]
    lane_center_position = (right_fit_center + left_fit_center) /2
    center_dist = (car_position - lane_center_position) * xm_per_pix
    ```
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The processed warped imaged was rewarped back to the original image with the lane marked.

Code that governs this function follows as :-

```python
newwarp = cv2.warpPerspective(final_output, M_inv, (image.shape[1], image.shape[0])) 
#     # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
```

Output

![alt text][finaloutputimage]





---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

**Points I considered Important for discussion**

1.  I added running average for last 10 frames to smooth out the lane line.This was done to smooth out the output lane mark.
2. This pipe fails when the lines darker than the lane lines appear as that of challenge video.The pipeline fails to distinguish between these line and the actual lane line.
3.  For harder challenge video.The range or the point to change prespective will fail as the length of the lane keeps changing.giving specific points to be considered for prespecitive transform will fail.
4. If a vehicles comes within view of the car ie inside the polygon created for marking the lane ,the pipeline wont be able handle that case.

5.  Problem I faced while developing this pipeline.
    
    5.1 To chose points for making quadrilateral for prespective transform and lane line marking.The points had to chosen to make sure that no outside noise doesnot deflects the lane lines.

    5.2 Understanding the logic behind the sliding windows and using the previous marked windows to generate new window was getting difficult as the flags that needs to be set to make that work was not very clear and hence I worked on taking the running average of the left fit and the right fit curve to make the lane line smoothing. Adding the function to take the previous windows and generating new window would have saved time but was not required for recorded video.
    
    
