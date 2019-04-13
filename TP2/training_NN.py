from __future__ import print_function

import numpy as np
import cv2 as cv
from keras.utils import np_utils

def load_base(path):
    # data = np.loadtxt(path, np.float32, delimiter=',', converters={ 0 : lambda ch : ord(ch)-ord('A') })
    data = np.loadtxt(path, np.float32, delimiter=',', converters={0: lambda ch : convertFun(ord(ch))})
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
class LetterStatModel(object):
    class_n = 4
    train_ratio = 0.5

    def load(self, fn):
        self.model = self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

    def unroll_samples(self, samples):
        sample_n, var_n = samples.shape
        new_samples = np.zeros((sample_n * self.class_n, var_n+1), np.float32)
        new_samples[:,:-1] = np.repeat(samples, self.class_n, axis=0)
        new_samples[:,-1] = np.tile(np.arange(self.class_n), sample_n)
        return new_samples

    def unroll_responses(self, responses):
        new_responses = responses
        # sample_n = len(responses)
        # new_responses = np.zeros(sample_n*self.class_n, np.int32)
        # resp_idx = np.int32( responses + np.arange(sample_n)*self.class_n )
        # new_responses[resp_idx] = 1
        new_responses = np_utils.to_categorical(responses, 4)
        return new_responses

class MLP(LetterStatModel):
    def __init__(self):
        self.model = cv.ml.ANN_MLP_create()

    def train(self, samples, responses,hyperparameters):
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


if __name__ == '__main__':
    import getopt
    import sys

    print(__doc__)

    models = [MLP] # NBayes
    models = dict( [(cls.__name__.lower(), cls) for cls in models] )


    args, dummy = getopt.getopt(sys.argv[1:], '', ['model=', 'data=', 'load=', 'save='])
    args = dict(args)
    args.setdefault('--model', 'mlp')
    args.setdefault('--save', 'trained_nn')

    print('loading data %s ...')
    samples, responses = load_base('D:/pythonProject/ComputerVision/TP2/files/csv/training_set.txt')
    Model = models[args['--model']]
    result = {}
    hyperparameters = range(47,48)
    for i in range(len(hyperparameters)):
        model = Model()
        train_n = int(len(samples) * model.train_ratio)
        if '--load' in args:
            fn = args['--load']
            print('loading model from %s ...' % fn)
            model.load(fn)
        else:
            # print('training %s ...' % Model.__name__)
            model.train(samples[:train_n], responses[:train_n], hyperparameters[i])

        # print('testing...')
        train_rate = np.mean(model.predict(samples[:train_n]) == responses[:train_n].astype(int))
        test_rate = np.mean(model.predict(samples[train_n:]) == responses[train_n:].astype(int))
        sample = samples[train_n:]
        result[hyperparameters[i]] = test_rate
        predict = model.predict(samples[train_n:])
        print(predict)
        # print('train rate: %f  test rate: %f' % (train_rate * 100, test_rate * 100))

        if '--save' in args:
            fn = args['--save']
            # print('saving model to %s ...' % fn)
            model.save('model')
        cv.destroyAllWindows()
    max = 0
    best = 0
    for i in range(len(result)):
        if result[hyperparameters[i]] > max:
            max = result[hyperparameters[i]]
            best = hyperparameters[i]
    print(max,'\n',best)