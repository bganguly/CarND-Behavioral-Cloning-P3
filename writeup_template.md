#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_for_writeup/model.png "Model Visualization"
[image2]: ./output_for_writeup/raw_image.png "raw image"
[image3]: ./output_for_writeup/raw_image_rescaled.png "raw image rescaled"
[image4]: ./output_for_writeup/raw_image_trimmed.png "raw image trimmed"
[image5]: ./output_for_writeup/raw_image_trimmed_and_rescaled.png "raw image trimmed and rescaled"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* Keras_behavioral_learning_car_simulator.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_template.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The Keras_behavioral_learning_car_simulator.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Architecture and Training Documentation

####1. Thoroughly discuss the approach taken for deriving and designing a model architecture fit for solving the given problem.  

The overall strategy for deriving a model architecture was to start with a Conv layer and add in dropout and flatten as needed.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. With the basic approach above the losses from model.fit() weren't narrowing down quickly enough with reasonable epochs. I then added in the MaxPooling and ran the simulation with fairly poor results. Based on some discussions on the slack channels i decided to addin the Droput layer, but the parameter to that still needed some tweaking over several iterations. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Provide sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

I choose to use the Keras Sequential Model as that seemed more declarative and easier to use than the Functional API construct.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Describe how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset should be included.

I started off by using Udacity's provided raw dataset , as i had trouble using the simulator to keep the car on the track using the keyboard alone. Some sample images are as follows.  
raw image . 
![alt text][image2] . 
raw image scaled  
![alt text][image3] . 
raw image trimmed . 
![alt text][image4] . 
raw image trimmed and scaled . 
![alt text][image5] . 

The images were both trimmed (take out the upper half or so, as that should not contribute to the ability to predict the steering angle, and scaled, as we want the classifer to be able to work off of 'rougher' images.
The features were obtained by adding a small delta to the left camera and subtracting a small delta from the right camera.
I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I found an epoch of 20 wieth a batch_size of 128 to be adequate, after several iterations. I used an adam optimizer so that manually training the learning rate wasn't necessary.
