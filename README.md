# Colorful Image Colorization (DD2424 Deep Learning in Data Science - Group Project)

## Introduction
In this report, the main goal is replicate the results of the paper Colorful Image Colorization (R. Zhang, P. Isola, and A. A. Efros, “Colorful image colorization,” CoRR, vol. abs/1603.08511, 2016.) to produce plausible color versions of photographs given a grayscale input. 
The original model has been trained using the whole <a href="https://www.image-net.org/">ImageNet dataset</a> that contains more than a milion images. 
Given our resource (and time) constraints, we decided to use only 50'000 images as to speed up the training, at the expenses of suboptimal results.
However, the results we obtained were satisfactory considering the limited time and resources we had available, but are not close to the ones obtained by the authors. 
We also explored different network structures to see how the results could be improved.
At the end we performed what the authors refer to as the "Colorization Turing Test", where people are asked to choose between the original image and the one generated by the network, pointing to the one they thing is true.


## CNN structure
We show the results we obtained when trying to replicate the results of the paper by recreating the same CNN network structure implemented by the authors using Tensorflow Keras. 
The CNN model takes as input the L channel of the image, encoded using the Lab color space, and learns a probability distribution over the quantized ab channels to return a prediction on the ab channels. 

<img src="https://user-images.githubusercontent.com/63954877/211788643-c9b9d8b6-7c04-4d71-9505-ce342da76092.png" width=40%>
<figcaption><b>Fig: Top: examples of L channels used as input for the network; bottom: real version of the images.</b></figcaption><br /> 

<img src="https://user-images.githubusercontent.com/63954877/211787523-05c521cb-03bf-4c8a-869b-0da1ebecfd45.png" width=90%>
<figcaption><b>Fig: Structure of the CNN network.</b></figcaption><br /> 

## Loss function
Compared to previous colorization methods, this model is made in a way that encourages the production of vibrant and realistic colors, as opposed to dull and desaturated ones. 
This is possible through class rebalancing, which is a way to give different importance to colors that are more or less common, accordingly; this is then used in a custom loss function that helps us take this into account. 

<img src="https://user-images.githubusercontent.com/63954877/211790980-d4c3c689-5af7-4227-91fc-49ddfca7519c.png" width=60%>
<figcaption><b>Fig: Brief description of the loss function and various components.</b></figcaption><br /> 


## Output examples
Once they are put together, we obtain a realistic representation of how the image could have been with colors. The model is trained using only a subset of the original dataset, given our time constraints, but it is still able to obtain remarkable results. 

<img src="https://user-images.githubusercontent.com/63954877/211792249-c161ae19-a738-4169-8f98-923794bf10d5.png" height=60%>
<figcaption><b>Fig: Some of the best output samples we obtained, next to the ground truth version of the image.</b></figcaption><br /> 

## Modifications to the network structures and experiments

One of the tasks of the project was also try out different implementations of the network, with the purpose of improving the results. 
What were combinations of the following techniques:
* using some regularization technique (eg. L2)
* changing optimizer for Gradient Descent (eg. Adam, AdaDelta)
* reducing the number of layers
* adding dropout
* reducing the resolution of the output

The results of the experiments can be read in the report, but what we noticed could help the most was reducing the output resolution, so the model had less pixels to predict and the colors were more accurate.

<img src="https://user-images.githubusercontent.com/63954877/211795252-0f99e151-dea7-4d5a-a4bb-b905d632b939.png" width=90%>
<figcaption><b>Fig: Results we obtained with different implementations of the network.</b></figcaption><br /> 
