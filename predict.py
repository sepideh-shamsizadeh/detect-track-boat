import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image

reconstructed_model = keras.models.load_model("my_model")


def predictImage(filenames):
    for filename in filenames:
        test = image.load_img(filename, target_size=(96, 96))
        Y = image.img_to_array(test)

        X = np.expand_dims(Y, axis=0)
        val = reconstructed_model.predict(X)
        print(val)
        if val[0] > 0.97:
            print('boat')
        else:
            print('not')


filenames = ["../segmentation/test/0.jpg", "../segmentation/test/1.jpg", "../segmentation/test/2.jpg", "../segmentation/test/3.jpg", "../segmentation/test/5.jpg", "../segmentation/test/8.jpg", "../segmentation/test/9.jpg",
             "../segmentation/test/10.jpg"]
print('boat', 'boat', 'boat', 'not', 'boat', 'not', 'not', 'not')
predictImage(filenames)
