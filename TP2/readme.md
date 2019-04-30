# CamShift

### Aim
The aim for this project is to track the hand in the camera and take a picture of it. We chose Camshift to track our object since it can adapt the size of the object. 
### Problems
But still we have two problmes to solve:
* How do we get the color histogram (which is the color distribution of the object we want to track) of our hand?
* How can we eliminate the effect of our face?

### How to solve
Fortunately, we can solve the two problems together by using the **Face detect**.

 1. Using face detect and get the histogram(hope that the color for our face and hands are similar), we use this histogram to find our hand
 2. Set the probability to zero in the area of face(see [back project](https://github.com/wangzeyao/ComputerVision/blob/master/Markdowns/%E5%8F%8D%E5%90%91%E6%8A%95%E5%BD%B1.md))

So now the CamShift is responsible for the tracking(see also [meanshift and camshift](https://github.com/wangzeyao/ComputerVision/blob/master/Markdowns/MeanShift.md))  and the face detection find the face and wipe its probability of it, problem solved.

### Result
**LOVE & PEACE**
![enter image description here](https://lh3.googleusercontent.com/COe2BDRsrUkzFu8kQ7_owF7PAwuzftFNP2CMXnhpAUPND-Z23veVAi_6VeYWOxnklT4I6-qk8POi)
small pic
![enter image description here](https://lh3.googleusercontent.com/ucfznrvRkR2gOcV3CTDeSMNVg8RMAupYQE3l7KxX6UG9s1Wm8Z8kU2CPTF677D9f1Mh0f_74BdyM)


And the matrix which contain the probabiltiy after resized to 16X16![enter image description here](https://lh3.googleusercontent.com/aQelvcsRgM2wn7uPCuxZxa8l-eqPfAwmnhaLyKiGmLr9_HbllHoW8iDX7J1GScJvnd_1Az0CAUQn)
It's for the next step: **Train a neural network to recogenize our hand gesture.**

The code is in the same directory called `camshift.py` with comments.



# Gesture recognition by Deep Neural Network

## Previously on Camshift
In the last TP, we used the camshift algorithme to track the hand and take picture of it. Then we resize the picture to 16X16 matrix and store it as the data set to train our neural network.  
I spend an hour to capture the gesture of my hand and I only got like 1500 pictures. So in order to have more data to train the network. I randomly select 1000 pictures I have and rotate each of them with a random angle and add them into the data.(The code for selecting and rotating them is [here](https://github.com/wangzeyao/ComputerVision/blob/master/TP2/img_process.py)). So now I have a data set with 2500 pictures.
![The data set we have](https://lh3.googleusercontent.com/6nrSCWnXf75wqYINjmOW35lKdmufyrk-GmxiSHLdex9ZjCvHKF6_qfRer5DcR2GRQJuv2JzzBUVd "The data set")
We have four letters to predict C,V,I and O 
We will use two kinds of Network, MLP and CNN

## 1.Multilayer Perceptron
A **multilayer perceptron** (MLP) is a class of feedforward artificial neural network. A MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer. For MLP, we will using an existing python file in the sample of openCV *letter_recog.py*

### 1.1 Load data and preprocessing
First, we define a function to load the data and do some preprocessing.
```python
def load_base(path):  
    data = np.loadtxt(path, np.float32, delimiter=',', converters={0: lambda ch : convertFun(ord(ch))})  
    index = [i for i in range(len(data))]  
    np.random.shuffle(index)  
    data = data[index]  
    samples, responses = data[:,1:], data[:,0]  
    return samples, responses
    
def convertFun(letter):  
    if chr(letter) == 'C':  
        return 0  
  elif chr(letter) == 'V':  
        return 1  
  elif chr(letter) == 'I':  
        return 2  
  elif chr(letter) == 'O':  
        return 3
```
We first load the data from .txt file and use a convert function to transfer letter to numbers. Then we split the data into **samples** and **responses** or you can call it X and Y.
Then in class **LetterStatModel**, I modified the function **unroll_responses** in order to create the one-hot array for dimension 4(the previous code is for 26 letters,but now i only have 4) with the help of *np_utils.to_categorical()* in *keras.utils*
```python
def unroll_responses(self, responses):  
    new_responses = responses  
    new_responses = np_utils.to_categorical(responses, 4)  
    return new_responses
```
### 1.2 Structure of the neural network
We can only modify two things for this MLP, the number of hidden layer and how many neurons in each layer. Since there are simple pictures and we only have four letters, I tried one and two layers with number of neurons equals to (5,10,15,20,.....200)
#### single hidden layer
![enter image description here](https://lh3.googleusercontent.com/1RgPvYWinIKGAq8gT-dp2bbEjkK-CE5ZGQ2SCHdWwyP6RxkrGUmnHwcEJM3OWcIsZaYcEBi0dfHh)
For the single layer MLP with 40 nerons, we have the best result of 90.89% accuracy in the validation. And with the number of nerons becomes lager, the accuracy decreases.

#### two hidden layer
![enter image description here](https://lh3.googleusercontent.com/QRqlWXPYOrnbFARfhTIHt93RToUQWclyRtHwgVCU00PbdzInz4g2Sgu-hVqTqF-p1P0s_ruHe2ii)
For two hidden layers, the best result showed up at 55 nerons for each layers and its accuracy was 91.29%. We don't see too much difference with the two kind of MLP but I still choose to use the second one.

The code for train the MLP is here

## 2. Convolutional Neural Network
The introduction for CNN is in my another [markdown](https://github.com/wangzeyao/ComputerVision/blob/master/Markdowns/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md)
This time I used keras with backend tensorflow to build the CNN.

### 2.1 Load data and preprocessing
It's the same with MLP for data loading but a little bit difference in preprocessing. This time we will have more works to do.
#### shuffle and split the data
```python
data = np.loadtxt(path, np.float32, delimiter=',', converters={0: lambda ch : convertFun(ord(ch))})  
index = [i for i in range(len(data))]  
np.random.shuffle(index)  
data = data[index]  
x_train, y_train = data[:2000,1:], data[:2000,0]  
x_test, y_test = data[2000:2521,1:], data[2000:2521,0]
```
The most important part here is to **shuffle** the data. When I first try to use the CNN and the results were not satisfying. I tried to modifer everything but none of them worked. Then I realised that when I took the picture, I usually press like 50 times button C to store the gesture of C and 50 times for O, and V and so on. So the data I feed to the network are in orderded, that may influence the performence of the network. So I tried to shuffle the data first and the accuracy becames much better than before.
After the shuffling, I split the data so that I have 80% data for training and 20% of data for testing.
#### reshape the format of the picture
For a set of pictures, we have two kinds of format to represent it. One is **channels_first**, like this (100,3,16,16). The first number represent the number of samples, second one is the number of channels, and the last two is the lenght and height of the picture.
But we still have **channels_last** format like (100,16,16,3) which the number of the channels is the last one.
After that, I convert the data into float32, and do the normalization in order to speed up the compute.
```python
from keras import backend as K  
img_row, img_col = 16,16  
  
if K.image_data_format() == 'channels_first':  
    shape_ord = (1,img_row,img_col)  
else:  
    shape_ord = (img_row,img_col,1)
    
x_train = x_train.reshape((x_train.shape[0],)+shape_ord)  
x_test = x_test.reshape((x_test.shape[0],)+shape_ord)  
x_train = x_train.astype(np.float32)  
x_test = x_test.astype(np.float32)  
x_train /= 255  
x_test /= 255
```
Then I convert the Y into one-hot encoding
```python
nb_class = 4  
y_train = np_utils.to_categorical(y_train,nb_class)  
y_test = np_utils.to_categorical(y_test,nb_class)
```

### 2.2 Structure of the neural network
For CNN I tried two structure, one is quit simple with only one convolution layer(call it simple CNN),the other is the classic LeNet.
#### simple CNN
```python
def simpleCNN(kernel_size=(3,3),activation='relu'):  
	model = Sequential()  
	model.add(Conv2D(nb_filter,  
			kernel_size=kernel_size,  
			padding='valid',  
			input_shape=shape_ord,  
			activation=activation))  
	model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))  
	model.add(Dropout(0.2))  
	model.add(Flatten())  
	model.add(Dense(units=128,activation=activation))  
	model.add(Dense(units=4,activation='softmax'))  
	return model
```
As we can see it's quit simple.  One Conv2d with 16 kernels, one Maxpooling and one Dense with 128 units. The kernel size is 3X3

#### LeNet
```python
def LeNet(kernel_size=(5,5),activation='relu'):  
	model = Sequential()  
	model.add(Conv2D(filters=6,  
			kernel_size=kernel_size,  
			strides=(1, 1),  
			padding='valid',  
			input_shape=shape_ord,  
			activation=activation,  
			name='Conv1'))  

	model.add(AveragePooling2D(pool_size=(2, 2),  
			strides=(1, 1),  
			padding='valid'))  

	model.add(Conv2D(16,  
			kernel_size=kernel_size,  
			strides=(1, 1),  
			padding='valid',  
			activation=activation,  
			name='Conv2'))  

	model.add(AveragePooling2D(pool_size=(2, 2),  
			strides=(2, 2),  
			padding='valid'))  

	model.add(Conv2D(120,  
			kernel_size=(5, 5),  
			strides=(1, 1),  
			padding='valid',  
			activation=activation,  
			name='Conv3'))  

	model.add(Flatten())  

	model.add(Dense(units=120,  
			activation='tanh'))  

	model.add(Dense(units=84,  
			activation='tanh'))  

	model.add(Dense(units=4,  
			activation='softmax'))  
	return model
```
click here to see the svg picture of the network.
It has 3 Conv2d layer, 2 average pooling layer and 2 dense layer.

### 2.3 choose of epoch
In order to prevent them from overfitting, I tried to train both of them 100 epoch and observr at which epoch the accuracy of validation stop to increase.

LeNet
![enter image description here](https://lh3.googleusercontent.com/9NbRfy0y4WcSKjE9OruHB866Oh_lLKqeQJs0gbLdFTr36fVeHRV6CMn2p-bG_4AIz4nsXu0SOdvm)

Simple CNN
![enter image description here](https://lh3.googleusercontent.com/0qRbLT9dibFRGUxa7fZUSrHjmtsyfp7tGJTsgEvhUxBZsG3rvR6xrTbY9IB5Y6q4-k8IPFKoMHUG)
So for LeNet, it stopped increasing at 30th epoch. So I decided to train LeNet and simple CNN with  30 epoches.
### 2.4 choose of parameters
Here I want to difference performence between the different optimizer and different kernel size.
#### Optimizer
For the introduciton of different gradient decent method, see my other [markdown](https://github.com/wangzeyao/ComputerVision/blob/master/Markdowns/%E5%B8%B8%E7%94%A8%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%96%B9%E6%B3%95.md).
Here I tried sgd and adam.
##### 1. SGD
![enter image description here](https://lh3.googleusercontent.com/TliBjMrX1Cv45ZGJ_SuKgeciMf1MlXLNs1-AbdjzZ2n8S-C5lJCZUp1afqrOk_p_zl4lkRPeHzZi)
![enter image description here](https://lh3.googleusercontent.com/q2BG9Du9r738zOU4a-LCh-8S0bKiilyIExf2J5IhlWCBbzNBVxY6QWqbyN5CEq0WKYBZt-WOM55d)

##### 2.ADAM
![enter image description here](https://lh3.googleusercontent.com/9VFdWaogCnwwuUVK4nJfiywuQ8-OPRNZtgIjEi64K1xx0qmcD9Q3yEHFy6pB2jLAO1WzbO8oZiBV)
![enter image description here](https://lh3.googleusercontent.com/Skv8zjUmN1JPYsgCy0IlR5ooad_xMbZ7mlLuBQ8w88g5eF44YtwrmjWlX6R5P2MsQo4BiV4TMvY9)

We can see that SGD converge faster, but adam is much smoother and more stable.

#### Kernel size
I tried 3 x 3 and 5 x 5 kernel for the two sturcture.
BUT for LeNet I have to modify the last convolution layer if the padding option is *valide*
The formula to calculate the out put size for each layer is:
FOR valid
$$
\left\lceil\frac{(W-F+1)}{S}\right\rceil
$$
 FOR same
$$
\left\lceil\frac{W}{S}\right\rceil
$$
 W is the input size, F is the size of kernel, S is the strde.
##### 3 X 3 kernel 
Conv1:$\left\lceil\frac{(16-3+1)}{1}\right\rceil=14$
Averagepool1:$\left\lceil\frac{(14-2+1)}{1}\right\rceil=13$
Conv2:$\left\lceil\frac{(13-3+1)}{1}\right\rceil=11$
Averagepool2:$\left\lceil\frac{(11-2+1)}{2}\right\rceil=5$
Conv3:$\left\lceil\frac{(5-5+1)}{1}\right\rceil=1$
![enter image description here](https://lh3.googleusercontent.com/pqqMhQKlkB1NqVVIX_OfVEcZGFBB6590zPcqXdRZ9YV5ZX3fp2a5LLWlDqLCEYFqohtLCcIZvKfN)
![enter image description here](https://lh3.googleusercontent.com/Ax9P9w8tjMklfX0Hn5IDfbncF5eBa0t8OCYPO5-tfDNgQuPFs9mXnKP_kUVID2RT_NKgaPN3xUAJ)
##### 5 X 5 kernel 
Conv1:$\left\lceil\frac{(16-5+1)}{1}\right\rceil=12$
Averagepool1:$\left\lceil\frac{(12-2+1)}{1}\right\rceil=11$
Conv2:$\left\lceil\frac{(11-5+1)}{1}\right\rceil=7$
Averagepool2:$\left\lceil\frac{(7-2+1)}{2}\right\rceil=3$
Conv3:$\left\lceil\frac{(3-3+1)}{1}\right\rceil=1$
![enter image description here](https://lh3.googleusercontent.com/mKQCXtjXjAtK7u4o9qPy48eV_xaUJ1_vBG1kgd69J-cynyxuhHgY34lTupszi--fFKQdZjtVymMW)
![enter image description here](https://lh3.googleusercontent.com/lCvVc4z0Ex9BVpn82AKafSRb5d4h8McJOtJZNi6b2VCgkBSDajOXARjrDae532z7jUV0kWb-3u7B)

But I don't see any evident difference for 5x5 kernel and 3x3 kernel.
### 2.5  K-fold Cross-Validation
I used k-fold cross validation to test the models.
![enter image description here](https://lh3.googleusercontent.com/Mk3q2oDIHpkQKcoBupBc-k8I3nu19HytvKa5ztcXRN0SOGAhzkuNDA8s1Z0hgu4xeghYq4kEnzyu)
K-fold cross validation
  1. Split the data into k parts(usually 5 or 10)
  2. Pick one part as test set and others as training set
  3. Compute the mse for this iteration
  4. Compute the average mse for k mse

I first tried to use the function in package scikitlearn called *~~StratifiedKFold().split()~~（should use function kfold() instead, thanks to melissa）* to split the data in k parts but it turns out that this function can only handle binary or multiclass so it didn't work.
So I have to write it by myself, fortunately it isn't diffcult.(If I wrote it right)
```python
def kfoldSplit(X,Y,k=10):  
    total_size = X.shape[0]  
    precentage = 1/k  
    size = int(total_size * precentage)  
    start = 0  
    end = size  
    x_train = []  
    y_train = []  
    x_test = []  
    y_test = []  
    for i in range(k):  
        x_test.append(X[start:end,:,:,:])  
        x_train.append(np.concatenate((X[:start,:,:,:],X[end:2522,:,:,:]),axis=0))  
        y_test.append(Y[start:end, :])  
        y_train.append(np.concatenate((Y[:start,:],Y[end:2522,:]),axis=0))  
        start = end  
        end += size  
    return [x_train,y_train,x_test,y_test]
```
and with this function I can apply the k-fold cross validation.
```python
for i in range(k):  
    model = CNN.LeNet()  
    model.compile(loss='categorical_crossentropy',  
		      optimizer=CNN.adam,  
			  metrics=['accuracy']  # 评价函数  
			  )  
    model.fit(kfold[0][i],kfold[1][i],  
			  epochs=e,  
		      batch_size=size,  
		      verbose=0)  
    score = model.evaluate(kfold[2][i],kfold[3][i],verbose=0)  
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))  
    cv_scores.append(score[1]*100)
```
#### Evaluation the two CNN models I saved before(k=5):
![enter image description here](https://lh3.googleusercontent.com/D2PYrCeFkBBVi6QBA4MrpuKgfRQqwF8AkZ6iXRuuZsGWAFwwWojsB3suxDvG0x8U0jmuS7gxpOTz)

#### Using cross-validation to find hyper parameters
We can also using the kfold validation to find the best hyper parameter for the model.
I tried to find best batch size with best epoch by grid search:
![enter image description here](https://lh3.googleusercontent.com/oTJmYbdV3DoL5YKDXHTnv_EOp1IoQDr1xpqhl_NjSwn24XyDvBUKmX8Bg4Fm1Nz86EPnjkgR1Abn)
The result is evident. But the question is why none of them can compare with the 2 models I built before even with the same parameters.(Haven't figure it out)

### 2.5 Visualization
#### Test Visualization
Randomly chose 5 pics from the test set and make the prediction
![enter image description here](https://lh3.googleusercontent.com/EYcFVxOKt0RDmSSVL4aRFqlVhJ5WH5oXoY4rkqlq6x1FXIiwzvnBcqIknxeaM1XEZWebLVSqxqYl)

#### Output of Convolution layer
Output of the first Conv2d
![enter image description here](https://lh3.googleusercontent.com/mvsMVr46hR8sktpGTXPX7rX4JqKsella6NIo3qgps29jgH8jBr86JsGcneONtK8Mla1as-yzsZ9e)
Output of the second Conv2d
![enter image description here](https://lh3.googleusercontent.com/0ii84ywdaEGxNW11tm-duE32m_-vv_bjcz8uTI54MNsT5tiQOqjmoxOgz7kWavThnRg9pYARy3P0)
Output of the third Conv2d
![enter image description here](https://lh3.googleusercontent.com/sPUL3cZ7_wmHjjn8bUZnl9MdZI3euk9D0pva9oYu6kkhgAMaQpyCB5GR2XVBFtnmYabyg7mkbDgd)

## 3. Apply the model on Camshift
After I had the model it's easy to apply it on the camshift
```python
import letter_recog_NN as lcn
elif ch == ord('p'):  
    xs0, ys0, xs1, ys1 = self.track_window  
    small_pic = cv.resize(prob[ys0:ys1 + ys0, xs0:xs1 + xs0], dsize=(16, 16))  
    small_pic = small_pic.astype(np.float32)  
    lcn.showResult(small_pic)
```
code in letter_recog_NN is [here](https://github.com/wangzeyao/ComputerVision/blob/master/TP2/letter_recog.py)

So now I can make a gesture and press button p to make the prediction by the two nerual network,MLP and CNN. And the result is the captured picture with the prediction on the top.
![enter image description here](https://lh3.googleusercontent.com/4zA-hfZR0rGgcceJWDu_Rl3V_h8vk05YGK290h0AW0KK9GxJhPdXU_j_aJhVycXYBmaDp_jQuhLh)
![enter image description here](https://lh3.googleusercontent.com/PdxpmzWy2031tJGatiFJoBVuSaU4NO4baudxmUtbCX258CBtkdjC2CTNPetXQovljSL1WuUggkST)
![enter image description here](https://lh3.googleusercontent.com/WQph9eqj0FrmNyjeP29CZtVoEGfhigB8qPwbYh5GTO2ty6xvlr0mzE6fdXaUBg95KYnlpVtdqDID)
![enter image description here](https://lh3.googleusercontent.com/it-0MkDhfqrRbKZC8wsXPUkdlhb0GAXVPZ8ertGiaJ_alx86UFVGrSepmun59TfXEV_HEfr2m_Qc)

### Problems
**Why the accuracy is worse when I using LeNet than using the SimpleCNN with only one convolution layer?**

It may because my data set was gathered under the same light conditon,or I have few letters(Just four). So the "function" I want to fit is simple(for exemple a stright line). With more Convolution layer(or other layer), I have more parameters. So what I'm doing is to use a comlicated function to simulate a simple one, which means many of the parameters are useless and should be set to zero. And I don't have enough data to train them to zero.
So, if I want the LeNet model to have the same performance of the simpleCNN, what I can do are

 - Gathering more data under the light conditon, so that I will have enought data to force some parameters to zero. But I think this can only make the LeNet getting closer to the simpleCNN but it will never perform better than simple one.
 - Make the function more complicated. Since the cause of my problem is the function is too simple, I can make the funcion more complex by adding more letters or taking more photos in different light conditions. And in this case the LeNet could get better generalization ability than the simple one which is what I'm looking for.

### Works todo

  1. debug:some times for cross validation the accuracy for the first trian is very low.(check the *kfoldsplit* function)
  2. Adding more letters and gathering more data under different light conditons
  3. Try to use different CNN.