from keras.models import load_model
import numpy as np
import CNN

k = 5
path = 'D:/pythonProject/ComputerVision/TP2/files/csv/training_set.txt'

X, Y, shape = CNN.dataPreprocess(path)


def kfoldSplit(X, Y, k=10):  # split the data into k parts
    total_size = X.shape[0]
    precentage = 1 / k
    size = int(total_size * precentage)
    start = 0
    end = size
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(k):
        x_test.append(X[start:end, :, :, :])
        x_train.append(np.concatenate((X[:start, :, :, :], X[end:2522, :, :, :]), axis=0))
        y_test.append(Y[start:end, :])
        y_train.append(np.concatenate((Y[:start, :], Y[end:2522, :]), axis=0))
        start = end
        end += size
    return [x_train, y_train, x_test, y_test]


kfold = kfoldSplit(X, Y, k)
cv_scores = []  # store the score
batch_size = [64]
epoch = range(30, 31)
model = ['CNN_model_LeNet', 'CNN_model_simple']

for e in epoch:
    Y.append(e)
    for size in batch_size:
        for i in range(k):
            model = CNN.LeNet()
            model.compile(loss='categorical_crossentropy',
                          optimizer=CNN.adam,
                          metrics=['accuracy']  # 评价函数
                          )
            model.fit(kfold[0][i], kfold[1][i],
                      epochs=e,
                      batch_size=size,
                      verbose=0)
            score = model.evaluate(kfold[2][i], kfold[3][i], verbose=0)
            cv_scores.append(score[1] * 100)

        print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)), 'batch_size:', size, 'epoch:', e)

for mod in model:
    for i in range(k):
        model = load_model(mod)
        score = model.evaluate(kfold[2][i], kfold[3][i], verbose=0)
        cv_scores.append(score[1] * 100)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)), 'model:', mod)
