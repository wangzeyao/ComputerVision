import numpy as np
import cv2 as cv
from keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('CNN_model_simple')


class LetterStatModel(object):
    class_n = 26
    train_ratio = 0.5

    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)

    def unroll_samples(self, samples):
        sample_n, var_n = samples.shape
        new_samples = np.zeros((sample_n * self.class_n, var_n + 1), np.float32)
        new_samples[:, :-1] = np.repeat(samples, self.class_n, axis=0)
        new_samples[:, -1] = np.tile(np.arange(self.class_n), sample_n)
        return new_samples

    def unroll_responses(self, responses):
        sample_n = len(responses)
        new_responses = np.zeros(sample_n * self.class_n, np.int32)
        resp_idx = np.int32(responses + np.arange(sample_n) * self.class_n)
        new_responses[resp_idx] = 1
        return new_responses


class MLP(LetterStatModel):
    def __init__(self):
        self.model = cv.ml.ANN_MLP_create()

    def train(self, samples, responses, hyperparameters):
        _sample_n, var_n = samples.shape
        new_responses = self.unroll_responses(responses).reshape(-1, self.class_n)
        layer_sizes = np.int32([var_n, hyperparameters, self.class_n])

        self.model.setLayerSizes(layer_sizes)
        self.model.setTrainMethod(cv.ml.ANN_MLP_BACKPROP)
        self.model.setBackpropMomentumScale(0.0)
        self.model.setBackpropWeightScale(0.001)
        self.model.setTermCriteria((cv.TERM_CRITERIA_COUNT, 20, 0.01))
        self.model.setActivationFunction(cv.ml.ANN_MLP_SIGMOID_SYM, 2, 1)

        self.model.train(samples, cv.ml.ROW_SAMPLE, np.float32(new_responses))

    def predict(self, samples):
        _ret, resp = self.model.predict(samples)
        return resp.argmax(-1)


def MLP_prediction(image):
    mlp = MLP()
    mlp.load('MLP_model')
    image = image.reshape(1, 256)
    result = mlp.predict(image)
    if result == 0:
        return 'C'
    elif result == 1:
        return 'V'
    elif result == 2:
        return 'I'
    elif result == 3:
        return 'O'


def CNN_prediction(image):
    image = image.reshape((1,) + (16, 16, 1))
    result = model.predict(image)
    result = result.argmax(axis=1)
    if result == 0:
        return 'C'
    elif result == 1:
        return 'V'
    elif result == 2:
        return 'I'
    elif result == 3:
        return 'O'


def showResult(image):
    letter_predict = [MLP_prediction(image), CNN_prediction(image)]
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.text(0, -3, 'MLP: ' + letter_predict[0], color='black', size=50)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.text(0, -3, 'CNN: ' + letter_predict[1], color='black', size=50)
    plt.axis('off')
    plt.show()
