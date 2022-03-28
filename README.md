# Face-Mask-Detection-using-TensorFlow-and-OpenCV

### Construct a CNN2D model to detect if a person is wearing a face mask or not with your webcam

In most public meetings, such as malls, theaters, and parks, it is becoming increasingly vital to verify if persons in the audience are wearing face masks. The creation of an AI system that can detect if someone is wearing a face mask and allow them admission would be extremely beneficial to society. A basic Face Mask identification system is constructed using Convolutional Neural Networks, a Deep Learning approach (CNN). The TensorFlow framework and the OpenCV library were used to create this CNN Model, which is widely utilized in real-time applications.

### CNN Architecture

![picture alt](https://github.com/jaybfn/Face-Mask-Detection-using-TensorFlow-and-OpenCV/blob/main/models/model_architecture.png)

The Face Mask detection model is constructed using the keras library's Sequential API in this suggested technique. This allows us to incrementally add new layers to our model. The numerous layers that we employed in our CNN model are listed below.

The first layer is the Conv2D layer with 6 filters and the filter size or the kernel size is set to 5X5 and activation is set to relu
for the first layer,MaxPooling2D is used with the pool size of 2X2.

The second layer is the Conv2D layer with 16 filters and the filter size or the kernel size is set to 5X5 and activation is set to relu
In the second layer, the MaxPooling2D is used with the pool size of 2X2.

Next step, we use the Flatten() layer to flatten all the layers into a single 1D layer.

After the Flatten layer, 
we use 4 sets of Dropout (0.2) layer and dense layer with 256,128,64,32 and 16 units with activation as relu 
finally a single dense layer with 2 units and softmax function was built.

The softmax function returns a vector that represents each of the input units' probability distributions. Two input units are utilized in this example. The softmax function returns a vector containing two values from the probability distribution.

After building the model, we compile the model and define the loss function and optimizer function. For training purposes, we employ the 'Adam' Optimizer and the 'Binary Cross Entropy' as the Loss function.

### Tech requirements:
1. python 3.9 
2. Tensorflow-gpu
3. keras
4. opencv
5. pandas
6. numpy








