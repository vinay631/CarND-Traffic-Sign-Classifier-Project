# Traffic Sign Recognition

---

## Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1a]: ./sign_vis.png "Dispaly some random signs"
[image1b]: ./dist_vis.png "Distribution of sample size across sign types"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./my_signs/caution.jpg "Traffic Sign 1"
[image5]: ./my_signs/Do-Not-Enter.jpg "Traffic Sign 2"
[image6]: ./my_signs/fifty.jpg "Traffic Sign 3"
[image7]: ./my_signs/rightturn.jpg "Traffic Sign 4"
[image8]: ./my_signs/yield.jpg "Traffic Sign 5"

## Rubric Points
### In this section I will address the [rubric points](https://review.udacity.com/#!/rubrics/481/view).  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This document aims to satisfy the requirement of a "Writeup" file. The project code can be found in my [notebook](https://github.com/vinay631/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The image below displays some signs selected randomly from the training dataset.

![alt text][image1a]

The image below shows the distribution of sample size for each traffic sign type.

![alt text][image1b]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale to reduce the extra information that might be not required for classification. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Then I normalized the images so that the values are from -1 to 1. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 RGB image   							|
| Convolution 1     	| 1x1 stride, VALID padding, output = 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, output = 14x14x6   |
| Convolution 2  	    | 1x1 stride, VALID padding, output = 10x10x16  |
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, output = 5x5x16    |
| Flatten				| output = 400									|
| Fully connected		| input = 400, output = 120       	            |
| RELU					|												|
| Fully connected		| input = 120, output = 84       	            |
| RELU					|												|
| Fully connected		| input = 84, output = 10       	            |

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used LeNet architecture with learning rate of 0.001, epoch of 120 and the batch size of 128. Adam optimizer was used. The final accuracy of my model on test set was 93.17%.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I used the LeNet architecture for my model. I started with learning rate of 0.01 which didn't converge the model. I reduced the learning rate to 0.001 and changed my number of epoch to 120.
My final model results were:
* validation set accuracy of 94.6 
* test set accuracy of 93.17

If a well known architecture was chosen:
* What architecture was chosen?
  I chose LeNet-5 architecture first and it seemed to give a good result. I will be implementing other architectures too to compare them with the LeNet architecture.
* Why did you believe it would be relevant to the traffic sign application?
The original LeNet architecture performs well on MNIST data which has a lot of similarity with traffic sign data.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The validation accuracy is slightly higher than
 
If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I chose LeNet5 architecture as I was most familiar with it from the previous lessons and exercises.
* What were some problems with the initial architecture?
For me the intial challenge was coming up with the initial parameters.
* How was the architecture adjusted and why was it adjusted?
I added dropouts but that didn't seem to work. From the lessons I learnt that dropouts method help generalize the network well, but due to lack of suffcient time, I didn't explore more dropout probabilities.
* Which parameters were tuned? How were they adjusted and why?
Epoch, batch size, learning rate and dropout probabilities were adjusted. The parameters were tuned mostly by trial and error method. I had a high learning rate to start with which didn't help the model accuracy with lower value of epoch. So I increased the number of epochs and lowered the learning rate to 0.0001 which seem to have helped the model surpass the 93% accuracy on test dataset.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The third image was misclassified perhaps because of lower sample size for the class.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General caution      		| General caution   									| 
| No entry			| No entry      							|
| 50 km/h     			| 30 km/h 										|
| Right turn ahead					| Right turn ahead											|
| Yield	      		| Yield				 				|



The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 19th and 20th celsl of the Ipython notebook.

For the first image, the model predicts accurately the traffic sign as "No entry". Here are the top 5 softmax probabilities.

No entry:
-------------------------
No entry: 100.00%
Beware of ice/snow: 0.00%
Traffic signals: 0.00%
Turn left ahead: 0.00%
Priority road: 0.00%
