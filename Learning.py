import numpy as np
from os import path
import os
import glob
import cv2
from xml.dom import minidom
import tensorflow as tf
from keras.layers import Dense, Flatten, Activation, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class CNN:

    def __init__(self, x_train, x_test, y_train, y_test, IMAGE_SIZE):
        self.X_train, self.X_test, self.Y_train, self.Y_test = x_train, x_test, y_train, y_test
        self.IMAGE_SIZE = IMAGE_SIZE
        self.model = None
        self.datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=True,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.2,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

    def structure_model(self):
        IMG_SHAPE = (self.IMAGE_SIZE, self.IMAGE_SIZE, 3)
        # Pre-trained model with MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights='imagenet'
        )
        # Freeze the pre-trained model weights
        base_model.trainable = False
        # Layer classification head with feature detector
        self.model = tf.keras.Sequential([
            base_model
        ])

        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format="channels_last"))
        self.model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

    def train_model(self, learning_rate, num_epochs, val_steps, BATCH_SIZE):
        # Compile the self.model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                           loss='binary_crossentropy',
                           metrics=['accuracy']
                           )
        self.model.summary()

        num_train = len(self.X_train)
        steps_per_epoch = round(num_train) // BATCH_SIZE
        self.model.fit(self.datagen.flow(self.X_train, self.Y_train, batch_size=32, shuffle=True),
                       epochs=num_epochs,
                       steps_per_epoch=steps_per_epoch,
                       validation_data=(self.X_test, self.Y_test),
                       validation_steps=val_steps)
        self.model.save("my_model")

    def predict(self):
        predictions = self.model.predict_classes(self.X_test)
        predictions = predictions.reshape(1, -1)[0]
        print(classification_report(self.Y_test, predictions, target_names=['boat (Class 0)', 'not_boat (Class 1)']))


def get_data(data_dir, img_size, labels):
    dataset = []
    for l in labels:
        p = os.path.join(data_dir, l)
        class_num = labels.index(l)
        for img in os.listdir(p):
            try:
                img_arr = cv2.imread(os.path.join(p, img))[..., ::-1]  # convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
                dataset.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(dataset)


def read_image(dir_images, location_file):
    num = 0
    for fl in glob.glob(dir_images + '/*.png'):
        img = cv2.imread(fl)
        mta = minidom.parse(path.join(location_file, path.splitext(path.basename(fl))[0]+'.xml'))

        for box in mta.getElementsByTagName('bndbox'):
            num += 1
            xmin = int(box.getElementsByTagName('xmin')[0].childNodes[0].data)
            xmax = int(box.getElementsByTagName('xmax')[0].childNodes[0].data)
            ymin = int(box.getElementsByTagName('ymin')[0].childNodes[0].data)
            ymax = int(box.getElementsByTagName('ymax')[0].childNodes[0].data)
    return img, 1, [xmin, ymin, xmax, ymax]


def draw_bounding_box(image, ymin, xmin, ymax, xmax, color=(255, 0, 0), thickness=5):
    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)


def draw_bounding_boxes(image, boxes, color=[], thickness=5):
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    for i in range(boxes_shape[0]):
        draw_bounding_box(image, boxes[i, 1], boxes[i, 0], boxes[i, 3],
                                   boxes[i, 2], color[i], thickness)


def draw_bounding_boxes_array(image, boxes, color=[], thickness=5):
    draw_bounding_boxes(image, boxes, color, thickness)

    return image



















if __name__ == '__main__':
    label = ['boat', 'not_boat']
    image_size = 96
    path = 'data/train/'
    batch = 64
    epochs = 40
    val_steps = 20
    learning_rate = 0.0001
    data = get_data(path, image_size, label)
    print(data.shape)
    X = []
    Y = []
    for feature, lbl in data:
        X.append(feature)
        Y.append(lbl)

    # Normalize the data
    X = np.array(X) / 255

    X.reshape(-1, image_size, image_size, 1)
    Y = np.array(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    my_cnn = CNN(X_train, X_test, Y_train, Y_test, image_size)
    my_cnn.structure_model()
    my_cnn.train_model(learning_rate, epochs, val_steps, batch)
    my_cnn.predict()
