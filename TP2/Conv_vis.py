from keras.models import Model
from keras.models import load_model
import matplotlib.pyplot as plt


# this script is for plot the out put of the convolution layer

def conv_output(model, layer_name, img):
    input_img = model.input

    try:
        out_conv = model.get_layer(layer_name).output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))

    inter_layer_model = Model(inputs=input_img, outputs=out_conv)

    img = img.reshape((1, 16, 16, 1))

    inter_output = inter_layer_model.predict(img)

    return inter_output[0]


def showConvOutput(model, image, layer_name):
    for name in layer_name:
        out_put = conv_output(load_model(model), name, image)
        for i in range(6):
            show_img = out_put[:, :, i]
            plt.subplot(3, 2, i + 1)
            plt.imshow(show_img)
            plt.axis('off')
            plt.title('Conv1' + '-' + str(i))
        plt.show()
