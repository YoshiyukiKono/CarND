**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

[//]: # (Image References)

[image1](./writeup_images/chess_dist.png) "Car and Not Car"

[image2](./writeup_images/hog_visualization.png) "HOG Example"

[image3](./writeup_images/thres_bin.png) "Sliding Windows"

[image4](./writeup_images/warped_straight.png) "Sliding Window"

[image5](./writeup_images/compare_src_dst.png) "BBoxes and Heat"

[image6](./writeup_images/poly.png) "Labels Map"

[image7](./writeup_images/lane_proj.png) "Output BBoxes"

[video1](./output_images/project_video_output.mp4) "Project Output"



## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  
Images are taken from [Here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) for vehicles and [https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip] for non-vehicles
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![](./writeup_images/car_noncar_example.png)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![](./writeup_images/RGB_histogram.png)

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and used the following parameter in my project.

| Parameter        | Value   | Remarks   | 
|:-------------:|:-------------:| :-------------|
|color_space  |HSV  | Can be RGB, HSV, LUV, HLS, YUV, YCrCb |
|orient       |9  | HOG orientations |
|pix_per_cell |8  | HOG pixels per cell |
|cell_per_block |2  | HOG cells per block |
|hog_channel |ALL | Can be 0, 1, 2, or "ALL" |

To reach the above combination, I compared each accuracy for some combinations, as follows:
* HSV is the best of color spaces
* hog_channel: All is better than 1
* for the other parameters, I chose these values from sample code or forum recomendation

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG, Color Histogram, and Spatial Binding futures.

| Parameter        | Value   | Remarks   | 
|:-------------:|:-------------:| :-------------|
|spatial_size |(16, 16)  | Spatial binning dimensions |
|hist_bins |16 | Number of histogram bins |
|spatial_feat |True  | Spatial features on or off |
|hist_feat |True  | Histogram features on or off |
|hog_feat |True  | HOG features on or off |

* hist_bin: 16 is better than 32
* for the other parameters, I chose these values from sample code or forum recomendation

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to use two types of scale when searching cars: 0.9, 2. First of all, I used the bottom half of the image to search cars, as it covers the area of the load. Then, the first scale, 0.9, is used to search in the top half of the trimed image where cars look small. Then the second scale was used for the entire image (of the half of the original image).

![](./writeup_images/scales_apply.png)

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using HSV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![](./writeup_images/first_scale.png)
![](./writeup_images/second_scale.png)
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_images/project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

When processing the video, I used deque to use the advantage of heatmap values in some frames.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

* Scale modification for distant car
* 
