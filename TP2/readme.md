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

![colorful photo](https://lh3.googleusercontent.com/6HxHrem6aAD9bM8Zk2A43b5z9dFM8MF4_9QFZbzFZ9JEnbPemBy4QOeYSCcjK68ld1Hi1CgBwyQf "colorful")
![enter image description here](https://lh3.googleusercontent.com/H3EuGk0ECrOhvjLbq4DKDU9XV6lOUndeTKMwqzHbEO4gyKYm_iVdlINsL77aSY7loC1343-2vzOd)
**LOVE & PEACE**

And the matrix which contain the probabiltiy after resized to 16X16![enter image description here](https://lh3.googleusercontent.com/aQelvcsRgM2wn7uPCuxZxa8l-eqPfAwmnhaLyKiGmLr9_HbllHoW8iDX7J1GScJvnd_1Az0CAUQn)
It's for the next step: **Train a neural network to recogenize our hand gesture.**

The code is in the same directory called `camshift.py` with comments.
