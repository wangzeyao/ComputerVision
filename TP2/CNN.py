import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.optimizers import adam

path = 'D:/pythonProject/ComputerVision/TP2/files/csv/training_set.txt'



def convertFun(letter):
    if chr(letter) == 'C':
        return 0
    elif chr(letter) == 'V':
        return 1
    elif chr(letter) == 'I':
        return 2
    elif chr(letter) == 'O':
        return 3

# load data
data = np.loadtxt(path, np.float32, delimiter=',', converters={0: lambda ch : convertFun(ord(ch))})
x_train, y_train = data[:1000,1:], data[:1000,0]
x_test, y_test = data[1000:1511,1:], data[1000:1511,0]

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


nb_class = 4
y_train = np_utils.to_categorical(y_train,nb_class)
y_test = np_utils.to_categorical(y_test,nb_class)
print('Training X:',x_train.shape,'\n','Training Y:',y_train.shape)

# CNN网络的建立
nb_epoch = 2

batch_size = 64
nb_filter = 16  # filter的数量
nb_pool = 2  # max池化的大小
size_conv = (3,3)  # 卷积filter的大小

sgd = SGD(lr=0.1,  # 学习率
          decay=1e-6,  # 每次更新后的学习率衰减值
          momentum=0.9,  # 动量
          nesterov=True  #使用Nesterov动量
          )
adam = adam()

model =Sequential()
model.add(Conv2D(nb_filter,
                 kernel_size=size_conv,
                 padding='valid',
                 input_shape=shape_ord,
                 ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(nb_class))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy']  # 评价函数,不懂
              )

hist = model.fit(x_train,y_train,
                 batch_size=batch_size,
                 epochs=nb_epoch,
                 verbose=1,  # 日志显示模式
                 validation_data=(x_test, y_test)
                 )

import matplotlib.pyplot as plt

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['Training','Validation'])

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['Training','Validation'],loc='lower right')
plt.show()

loss,accuracy = model.evaluate(x_test,y_test,verbose=0)
print('Test Loss: ',loss)
print('Test acc: ',accuracy)

model.save('CNN_model')