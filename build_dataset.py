
from pickle import dump

import keras
from keras.applications.vgg16 import VGG16
import cv2
import  tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from tensorflow.python.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM

from  utils import util
import dataset_person as person
import numpy as np
import pandas as pd

# extract features from each photo in the directory
def extract_features(data):
    # load the model
    model = VGG16()
    # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # summarize
    print(model.summary())
    # extract features from each photo
    # features = dict()
    features = []
    for img in data[:20]:
        # load an image from file
        image = load_img(img, target_size=(224, 224))
        print(image)
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        feature = model.predict(image, verbose=0)
        # get image id
        # image_id = img.split('.')[0].split("\\")[2]
        # print(image_id)
        # store feature
        features.append(feature)
        # features[image_id] = feature
        print('>%s' % img)
    return features

dataset_dir="data/"

class Tracking(object):

    def __init__(self, dataset_dir, num_validation_y=0.2, seed=123):
        self._dataset_dir = dataset_dir
        self._num_validation_y = num_validation_y
        self._seed = seed

    def read_train(self):
        filenames, ids, camera_indices, _ = person.read_train_split_to_str(
            self._dataset_dir)
        train_indices, _ = util.create_validation_split(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        filenames = [filenames[i] for i in train_indices]
        ids = [ids[i] for i in train_indices]
        camera_indices = [camera_indices[i] for i in train_indices]
        return filenames, ids, camera_indices

    def read_validation(self):
        filenames, ids, camera_indices, _ = person.read_train_split_to_str(
            self._dataset_dir)
        _, valid_indices = util.create_validation_split(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        filenames = [filenames[i] for i in valid_indices]
        ids = [ids[i] for i in valid_indices]
        camera_indices = [camera_indices[i] for i in valid_indices]
        return filenames, ids, camera_indices

    # def read_test_filenames(self):
    #     filename = os.path.join(self._dataset_dir, "info", "test_name.txt")
    #     with open(filename, "r") as file_handle:
    #         content = file_handle.read()
    #         lines = content.splitlines()
    #
    #     image_dir = os.path.join(self._dataset_dir, "bbox_test")
    #     return [os.path.join(image_dir, f[:4], f) for f in lines]


def main():
    dataset = Tracking(dataset_dir, num_validation_y=0.2, seed=123)
    train_x, train_y, _ = dataset.read_train()
    print("Train set size: %d images, %d identities" % (
        len(train_x), len(np.unique(train_y))))
    valid_x, valid_y, camera_indices = dataset.read_validation()
    print("Validation set size: %d images, %d identities" % (
        len(valid_x), len(np.unique(valid_y))))
    train = extract_features(train_x)
    # print(list(zip(train,train_y[:20])))
    # data = zip(train,train_y[:20])
    # data = list(data)
    # # save to file
    # dump(train, open('train_x.pkl', 'wb'))
    # lookback = 3
    # pre = 1
    # inputs = np.zeros((len(data) - lookback, lookback, 3))
    # labels = np.zeros(len(data) - lookback)
    # for i in range(lookback, len(data) - (pre)):
    #     inputs[i - lookback] = data[0][i - lookback:i]
    #     labels[i - lookback] = data[1][i + pre]
    # inputs = inputs.reshape(-1, lookback, 1)
    # labels = labels.reshape(-1, 1)
    # print("input" ,inputs,"lavel",labels)
    train = np.asarray((train))
    train_y = np.asarray((train_y))
    print(train_y.shape)
    print(train.shape)

    model = Sequential()
    model.add(LSTM(units=64, activation='elu', input_shape=(None, 1)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    print(model.summary())
    train_data = tf.data.Dataset.from_tensor_slices((train, train_y[:20]))
    model.fit(train_data, epochs=10, batch_size=64)

if __name__ == "__main__":
    main()
