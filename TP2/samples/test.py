import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
np.random.seed(199)
def convertFun(letter):
    if chr(letter) == 'C':
        return 0
    elif chr(letter) == 'V':
        return 1
    elif chr(letter) == 'I':
        return 2
    elif chr(letter) == 'O':
        return 3
path = 'D:/pythonProject/ComputerVision/TP2/files/csv/training_set.txt'
data = np.loadtxt(path, np.float32, delimiter=',', converters={0: lambda ch: convertFun(ord(ch))})
index = [i for i in range(len(data))]
np.random.shuffle(index)
data = data[index]
print(data[1])