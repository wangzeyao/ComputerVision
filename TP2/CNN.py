import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import AveragePooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.optimizers import adam
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
import Conv_vis
from keras import backend as K
import matplotlib.pyplot as plt

# np.random.seed(2019)
path = 'D:/pythonProject/ComputerVision/TP2/files/csv/training_set.txt'
img_row, img_col = 16, 16
nb_class = 4


def convertFun(letter):
    if chr(letter) == 'C':
        return 0
    elif chr(letter) == 'V':
        return 1
    elif chr(letter) == 'I':
        return 2
    elif chr(letter) == 'O':
        return 3


def dataPreprocess(data_set):
    # load data
    data = np.loadtxt(path, np.float32, delimiter=',', converters={0: lambda ch: convertFun(ord(ch))})
    # shuffle the data
    index = [i for i in range(len(data))]
    np.random.shuffle(index)
    data = data[index]
    X = data[:, 1:]
    Y = data[:, 0]

    if K.image_data_format() == 'channels_first':  # channel first or channel last
        shape_ord = (1, img_row, img_col)
    else:
        shape_ord = (img_row, img_col, 1)

    print(shape_ord)
    X = X.reshape((X.shape[0],) + shape_ord)  # reshape the data
    X = X.astype(np.float32)
    X = X / 255  # normalization

    Y = np_utils.to_categorical(Y, nb_class)  # convert into one-hot encoding
    print('X:', X.shape, '\n', 'Y:', Y.shape)
    return [X, Y, shape_ord]


X, Y, shape_ord = dataPreprocess('D:/pythonProject/ComputerVision/TP2/files/csv/training_set.txt')

# split the data into training set and testing set
x_train, x_test = X[:2000, :, :, :], X[2000:2521, :, :, :]
y_train, y_test = Y[:2000, :], Y[2000:2521, :]

# bulit CNN

nb_epoch = 30
batch_size = 64
nb_kernels = 16  # filter的数量
kernel_size = (3, 3)
activation_fun = 'relu'
nb_pool = 2  # pooling seize
size_conv = (3, 3)  # kernel size

sgd = SGD(lr=0.1,  # learning rate
          decay=1e-6,  # 每次更新后的学习率衰减值
          momentum=0.9,  # 动量
          nesterov=True  # 使用Nesterov动量
          )
adam = adam()
ada_grad = Adagrad()
rmsprop = RMSprop()


def LeNet(kernel_size=(5, 5), activation='tanh'):
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

    # model.add(Conv2D(120,
    #                  kernel_size=(3, 3),
    #                  strides=(1, 1),
    #                  padding='valid',
    #                  activation=activation,
    #                  name='Conv3'))

    model.add(Flatten())

    model.add(Dense(units=120,
                    activation=activation))

    model.add(Dense(units=84,
                    activation=activation))

    model.add(Dense(units=4,
                    activation='softmax'))

    # model.summary()
    return model


def simpleCNN(kernel_size=(3, 3), activation='relu'):
    model = Sequential()
    model.add(Conv2D(nb_kernels,
                     kernel_size=kernel_size,
                     padding='valid',
                     input_shape=shape_ord,
                     activation=activation
                     ))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units=128, activation=activation))
    model.add(Dense(units=4, activation='softmax'))
    return model


def main():
    Le_Net = LeNet(kernel_size=(3, 3), activation='relu')
    simple_CNN = simpleCNN()

    model = Le_Net  # choose the model

    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy']  # 评价函数
                  )

    hist = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=nb_epoch,
                     verbose=0,  # log
                     validation_data=(x_test, y_test)
                     )

    # draw the loss plot
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.title('3x3 SimpleCNN')

    # draw the accuracy plot
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.title('LeNet')
    plt.show()

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print('Test Loss: ', loss)
    print('Test acc: ', accuracy)

    # plot 5 random picture with the prediction
    x_test_org = x_test.reshape(x_test.shape[0], img_row, img_col)
    nb_predict = 5
    x_pred = x_test
    prediction = model.predict(x_pred)
    prediction = prediction.argmax(axis=1)
    letter_index = np.random.randint(low=0, high=220, size=nb_predict)
    letter_predict = []
    for i in letter_index:
        if prediction[i] == 0:
            letter_predict.append('C')
        elif prediction[i] == 1:
            letter_predict.append('V')
        elif prediction[i] == 2:
            letter_predict.append('I')
        elif prediction[i] == 3:
            letter_predict.append('O')

    letter_img = []
    for i in letter_index:
        letter_img.append(x_test_org[i])

    plt.figure(figsize=(16, 8))
    for i in range(nb_predict):
        plt.subplot(1, nb_predict, i + 1)
        plt.imshow(letter_img[i])
        plt.text(0, -3, letter_predict[i], color='black', size=50)
        plt.axis('off')

    plt.show()
    # model.save('CNN_model_simple')

    # plot the out put of convolution layer
    Conv_vis.showConvOutput('CNN_model_LeNet', letter_img[1], ('Conv1', 'Conv2', 'Conv3'))


if __name__ == '__main__':
    main()
