import cv2 as cv
import numpy as np


# this script is for create more data by rotating some existing picture

def convertFun(letter):
    if chr(letter) == 'C':
        return 0
    elif chr(letter) == 'V':
        return 1
    elif chr(letter) == 'I':
        return 2
    elif chr(letter) == 'O':
        return 3


def numToLetter(num):
    if num == 0:
        return 'C'
    elif num == 1:
        return 'V'
    elif num == 2:
        return 'I'
    elif num == 3:
        return 'O'


path = 'D:/pythonProject/ComputerVision/TP2/files/csv/training_set.txt'
data = np.loadtxt(path, np.float32, delimiter=',', converters={0: lambda ch: convertFun(ord(ch))})

index = np.random.randint(low=0, high=1511, size=20000)  # make random index to get picture

for i in index:
    image = data[i, 1:258]
    letter = numToLetter(data[i, 0])
    image = image.reshape((16, 16))
    image = image.astype(np.uint8)  # convert it from int to uint8, 8 bit for 0-255
    angel = np.random.randint(low=10, high=270)  # randomly choose an angle to rotate
    M = cv.getRotationMatrix2D((8, 8), angel, 1.0)  # get the rotation matrix
    image = cv.warpAffine(image, M, (16, 16))  # apply the rotation matrix
    file = open('D:/pythonProject/ComputerVision/TP2/files/csv/training_set.txt', 'a')
    samples = image.reshape((1, 256))
    file.write(letter + ',')
    for j in range(256):
        if j == 255:
            file.write(str(samples[0][j]))
        else:
            file.write(str(samples[0][j]) + ',')
    file.write('\n')
    file.close()
