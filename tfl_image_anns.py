########################################################
# module: tfl_image_anns.py
# authors: vladimir kulyukin
# descrption: starter code for image ANN for project 1
# bugs to vladimir kulyukin in canvas
# to install tflearn to go http://tflearn.org/installation/
########################################################

import pickle
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

## we need this to load the pickled data into Python.
def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

## Paths to all datasets. Change accordingly.
PATH = './data/'
BEE1_gray_base_path    = PATH + 'BEE1_gray/'
BEE2_1S_gray_base_path = PATH + 'BEE2_1S_gray/'
BEE4_gray_base_path    = PATH + 'BEE4_gray/'

## let's load BEE1_gray
base_path = BEE1_gray_base_path
print('loading datasets from {}...'.format(base_path))
BEE1_gray_train_X = load(base_path + 'train_X.pck')
BEE1_gray_train_Y = load(base_path + 'train_Y.pck')
BEE1_gray_test_X = load(base_path + 'test_X.pck')
BEE1_gray_test_Y = load(base_path + 'test_Y.pck')
BEE1_gray_valid_X = load(base_path + 'valid_X.pck')
BEE1_gray_valid_Y = load(base_path + 'valid_Y.pck')
print(BEE1_gray_train_X.shape)
print(BEE1_gray_train_Y.shape)
print(BEE1_gray_test_X.shape)
print(BEE1_gray_test_Y.shape)
print(BEE1_gray_valid_X.shape)
print(BEE1_gray_valid_Y.shape)
print('datasets from {} loaded...'.format(base_path))
BEE1_gray_train_X = BEE1_gray_train_X.reshape([-1, 64, 64, 1])
BEE1_gray_test_X = BEE1_gray_test_X.reshape([-1, 64, 64, 1])

## to make sure that the dimensions of the
## examples and targets are the same.
assert BEE1_gray_train_X.shape[0] == BEE1_gray_train_Y.shape[0]
assert BEE1_gray_test_X.shape[0]  == BEE1_gray_test_Y.shape[0]
assert BEE1_gray_valid_X.shape[0] == BEE1_gray_valid_Y.shape[0]

## let's load BEE2_1S_gray
base_path = BEE2_1S_gray_base_path
print('loading datasets from {}...'.format(base_path))
BEE2_1S_gray_train_X = load(base_path + 'train_X.pck')
BEE2_1S_gray_train_Y = load(base_path + 'train_Y.pck')
BEE2_1S_gray_test_X = load(base_path + 'test_X.pck')
BEE2_1S_gray_test_Y = load(base_path + 'test_Y.pck')
BEE2_1S_gray_valid_X = load(base_path + 'valid_X.pck')
BEE2_1S_gray_valid_Y = load(base_path + 'valid_Y.pck')
print(BEE2_1S_gray_train_X.shape)
print(BEE2_1S_gray_train_Y.shape)
print(BEE2_1S_gray_test_X.shape)
print(BEE2_1S_gray_test_Y.shape)
print(BEE2_1S_gray_valid_X.shape)
print(BEE2_1S_gray_valid_Y.shape)
print('datasets from {} loaded...'.format(base_path))
BEE2_1S_gray_train_X = BEE2_1S_gray_train_X.reshape([-1, 64, 64, 1])
BEE2_1S_gray_test_X = BEE2_1S_gray_test_X.reshape([-1, 64, 64, 1])

assert BEE2_1S_gray_train_X.shape[0] == BEE2_1S_gray_train_Y.shape[0]
assert BEE2_1S_gray_test_X.shape[0]  == BEE2_1S_gray_test_Y.shape[0]
assert BEE2_1S_gray_valid_X.shape[0] == BEE2_1S_gray_valid_Y.shape[0]

## let's load BEE4_gray
base_path = BEE4_gray_base_path
print('loading datasets from {}...'.format(base_path))
BEE4_gray_train_X = load(base_path + 'train_X.pck')
BEE4_gray_train_Y = load(base_path + 'train_Y.pck')
BEE4_gray_test_X = load(base_path + 'test_X.pck')
BEE4_gray_test_Y = load(base_path + 'test_Y.pck')
BEE4_gray_valid_X = load(base_path + 'valid_X.pck')
BEE4_gray_valid_Y = load(base_path + 'valid_Y.pck')
print(BEE4_gray_train_X.shape)
print(BEE4_gray_train_Y.shape)
print(BEE4_gray_test_X.shape)
print(BEE4_gray_test_Y.shape)
print(BEE4_gray_valid_X.shape)
print(BEE4_gray_valid_Y.shape)
print('datasets from {} loaded...'.format(base_path))
BEE4_gray_train_X = BEE4_gray_train_X.reshape([-1, 64, 64, 1])
BEE4_gray_test_X = BEE4_gray_test_X.reshape([-1, 64, 64, 1])

assert BEE4_gray_train_X.shape[0] == BEE4_gray_train_Y.shape[0]
assert BEE4_gray_test_X.shape[0]  == BEE4_gray_test_Y.shape[0]
assert BEE4_gray_valid_X.shape[0] == BEE4_gray_valid_Y.shape[0]

### here's an example of how to make an ANN with tflearn.
### An ANN is nothing but a sequence of fully connected hidden layers
### plus the input layer and the output layer of appropriate dimensions.
def make_image_ann_model():
    input_layer = input_data(shape=[None, 64, 64, 1])
    fc_layer_1 = fully_connected(input_layer, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 128,
                                 activation='relu',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 2,
                                 activation='softmax',
                                 name='fc_layer_3')
    network = regression(fc_layer_3, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
    model = tflearn.DNN(network)
    return model

### Note that the load function must mimick the
### the archictecture of the persisted model!!!
def load_image_ann_model(model_path):
    input_layer = input_data(shape=[None, 64, 64, 1])
    fc_layer_1 = fully_connected(input_layer, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 128,
                                 activation='relu',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 2,
                                 activation='softmax',
                                 name='fc_layer_3')
    model = tflearn.DNN(fc_layer_3)
    model.load(model_path)
    return model

### test a tfl network model on valid_X and valid_Y.
def test_tfl_image_ann_model(network_model, valid_X, valid_Y):
    results = []
    for i in range(len(valid_X)):
        prediction = network_model.predict(valid_X[i].reshape([-1, 64, 64, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == \
                       np.argmax(valid_Y[i]))
    return float(sum((np.array(results) == True))) / float(len(results))

###  train a tfl model on train_X, train_Y, test_X, test_Y.
def train_tfl_image_ann_model(model, train_X, train_Y, test_X, test_Y, num_epochs=2, batch_size=10):
  tf.compat.v1.reset_default_graph()
  model.fit(train_X, train_Y, n_epoch=num_epochs,
            shuffle=True,
            validation_set=(test_X, test_Y),
            show_metric=True,
            batch_size=batch_size,
            run_id='image_ann_model')

### validating is testing on valid_X and valid_Y.
def validate_tfl_image_ann_model(model, valid_X, valid_Y):
    return test_tfl_image_ann_model(model, valid_X, valid_Y)


# Training and Saving Pipeline
def pipeline():
    epochs = 50
    max_accuracy = 0
    batch_vs_acc = {}
    img_ann = make_image_ann_model()
    for batch_size in [8,16,32]:
        # Train it on BEE1_gray
        train_tfl_image_ann_model(img_ann, BEE1_gray_train_X, BEE1_gray_train_Y, BEE1_gray_test_X, BEE1_gray_test_Y, num_epochs=epochs, batch_size=batch_size)


        # Train it on BEE2_1S_gray
        train_tfl_image_ann_model(img_ann, BEE2_1S_gray_train_X, BEE2_1S_gray_train_Y, BEE2_1S_gray_test_X, BEE2_1S_gray_test_Y, num_epochs=epochs, batch_size=batch_size)


        # Train it on BEE4_gray
        train_tfl_image_ann_model(img_ann, BEE4_gray_train_X, BEE4_gray_train_Y, BEE4_gray_test_X, BEE4_gray_test_Y, num_epochs=epochs, batch_size=batch_size)
        # Validate BEE1_GRAY
        bee1_acc = validate_tfl_image_ann_model(img_ann, BEE1_gray_valid_X, BEE1_gray_valid_Y)
        print("BEE1_gray", bee1_acc)
        # Validate BEE2_1S_gray
        bee2_acc = validate_tfl_image_ann_model(img_ann, BEE2_1S_gray_valid_X, BEE2_1S_gray_valid_Y)
        print("BEE2_1S_gray", bee2_acc)
        # Validate BEE4_gray
        bee4_acc = validate_tfl_image_ann_model(img_ann, BEE4_gray_valid_X, BEE4_gray_valid_Y)
        print("BEE4_gray", bee4_acc)
        mean_acc = (bee1_acc + bee2_acc + bee4_acc) / 3

        # save stats to dictionary
        batch_vs_acc[f"bee1_{batch_size}"] = bee1_acc
        batch_vs_acc[f"bee2_{batch_size}"] = bee2_acc
        batch_vs_acc[f"bee4_{batch_size}"] = bee4_acc

        if mean_acc > max_accuracy:
            img_ann.save("models/img_ann.tfl")
# Execute pipeline
def main():
    pipeline()
if __name__ == '__main__':
    main()
