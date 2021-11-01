#############################################################
# module: tfl_audio_convnets.py
# authors: vladimir kulyukin
# descrption: starter code for audio ConvNets for Project 1
# to install tflearn to go http://tflearn.org/installation/
#############################################################

import pickle
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

## we need this to load the pickled data into Python.
def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

## Paths to all datasets. Change accordingly.
base_path = './data'
BUZZ1_base_path = base_path + 'BUZZ1/'
BUZZ2_base_path = base_path + 'BUZZ2/'
BUZZ3_base_path = base_path + 'BUZZ3/'

## let's load BUZZ1
base_path = BUZZ1_base_path
print('loading datasets from {}...'.format(base_path))
BUZZ1_train_X = load(base_path + 'train_X.pck')
BUZZ1_train_Y = load(base_path + 'train_Y.pck')
BUZZ1_test_X = load(base_path + 'test_X.pck')
BUZZ1_test_Y = load(base_path + 'test_Y.pck')
BUZZ1_valid_X = load(base_path + 'valid_X.pck')
BUZZ1_valid_Y = load(base_path + 'valid_Y.pck')
print(BUZZ1_train_X.shape)
print(BUZZ1_train_Y.shape)
print(BUZZ1_test_X.shape)
print(BUZZ1_test_Y.shape)
print(BUZZ1_valid_X.shape)
print(BUZZ1_valid_Y.shape)
print('datasets from {} loaded...'.format(base_path))
BUZZ1_train_X = BUZZ1_train_X.reshape([-1, 4000, 1, 1])
BUZZ1_test_X  = BUZZ1_test_X.reshape([-1, 4000, 1, 1])

## to make sure that the dimensions of the
## examples and targets are the same.
assert BUZZ1_train_X.shape[0] == BUZZ1_train_Y.shape[0]
assert BUZZ1_test_X.shape[0]  == BUZZ1_test_Y.shape[0]
assert BUZZ1_valid_X.shape[0] == BUZZ1_valid_Y.shape[0]

## let's load BUZZ2
base_path = BUZZ2_base_path
print('loading datasets from {}...'.format(base_path))
BUZZ2_train_X = load(base_path + 'train_X.pck')
BUZZ2_train_Y = load(base_path + 'train_Y.pck')
BUZZ2_test_X = load(base_path + 'test_X.pck')
BUZZ2_test_Y = load(base_path + 'test_Y.pck')
BUZZ2_valid_X = load(base_path + 'valid_X.pck')
BUZZ2_valid_Y = load(base_path + 'valid_Y.pck')
print(BUZZ2_train_X.shape)
print(BUZZ2_train_Y.shape)
print(BUZZ2_test_X.shape)
print(BUZZ2_test_Y.shape)
print(BUZZ2_valid_X.shape)
print(BUZZ2_valid_Y.shape)
print('datasets from {} loaded...'.format(base_path))
BUZZ2_train_X = BUZZ2_train_X.reshape([-1, 4000, 1, 1])
BUZZ2_test_X = BUZZ2_test_X.reshape([-1, 4000, 1, 1])

assert BUZZ2_train_X.shape[0] == BUZZ2_train_Y.shape[0]
assert BUZZ2_test_X.shape[0]  == BUZZ2_test_Y.shape[0]
assert BUZZ2_valid_X.shape[0] == BUZZ2_valid_Y.shape[0]

## let's load BUZZ3
base_path = BUZZ3_base_path
print('loading datasets from {}...'.format(base_path))
BUZZ3_train_X = load(base_path + 'train_X.pck')
BUZZ3_train_Y = load(base_path + 'train_Y.pck')
BUZZ3_test_X = load(base_path + 'test_X.pck')
BUZZ3_test_Y = load(base_path + 'test_Y.pck')
BUZZ3_valid_X = load(base_path + 'valid_X.pck')
BUZZ3_valid_Y = load(base_path + 'valid_Y.pck')
print(BUZZ3_train_X.shape)
print(BUZZ3_train_Y.shape)
print(BUZZ3_test_X.shape)
print(BUZZ3_test_Y.shape)
print(BUZZ3_valid_X.shape)
print(BUZZ3_valid_Y.shape)
print('datasets from {} loaded...'.format(base_path))
BUZZ3_train_X = BUZZ3_train_X.reshape([-1, 4000, 1, 1])
BUZZ3_test_X = BUZZ3_test_X.reshape([-1, 4000, 1, 1])

assert BUZZ3_train_X.shape[0] == BUZZ3_train_Y.shape[0]
assert BUZZ3_test_X.shape[0]  == BUZZ3_test_Y.shape[0]
assert BUZZ3_valid_X.shape[0] == BUZZ3_valid_Y.shape[0]

def make_audio_convnet_model():
    input_layer = input_data(shape=[None, 4000, 1, 1])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=8,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    fc_layer_1 = fully_connected(pool_layer_1, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 3,
                                 activation='softmax',
                                 name='fc_layer_2')
    network = regression(fc_layer_2, optimizer='sgd',
                         loss='categorical_crossentropy',
                         learning_rate=0.01)
    model = tflearn.DNN(network)
    return model

def load_audio_convnet_model(model_path):
    input_layer = input_data(shape=[None, 4000, 1, 1])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=8,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    fc_layer_1 = fully_connected(pool_layer_1, 128,
                                 activation='relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 3,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(model_path)
    return model

def test_tfl_audio_convnet_model(network_model, valid_X, valid_Y):
    results = []
    for i in range(len(valid_X)):
        prediction = network_model.predict(valid_X[i].reshape([-1, 4000, 1, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == \
                       np.argmax(valid_Y[i]))
    return float(sum((np.array(results) == True))) / float(len(results))

###  train a tfl model on train_X, train_Y, test_X, test_Y.
def train_tfl_audio_convnet_model(model, train_X, train_Y, test_X, test_Y, num_epochs=2, batch_size=10):
  tf.compat.v1.reset_default_graph()
  model.fit(train_X, train_Y, n_epoch=num_epochs,
            shuffle=True,
            validation_set=(test_X, test_Y),
            show_metric=True,
            batch_size=batch_size,
            run_id='audio_cn_model')

### validating is testing on valid_X and valid_Y.
def validate_tfl_audio_convnet_model(model, valid_X, valid_Y):
    return test_tfl_audio_convnet_model(model, valid_X, valid_Y)

# Training and Saving Pipeline
def pipeline():
    epochs = 50
    max_accuracy = 0
    batch_vs_acc = {}
    img_ann = make_audio_convnet_model()

    for batch_size in [8,16,32]:
        # Train it on BUZZ1
        train_tfl_audio_convnet_model(img_ann, BUZZ1_train_X, BUZZ1_train_Y, BUZZ1_test_X, BUZZ1_test_Y, num_epochs=epochs, batch_size=batch_size)


        # Train it on BUZZ2_gray
        train_tfl_audio_convnet_model(img_ann, BUZZ2_gray_train_X, BUZZ2_gray_train_Y, BUZZ2_gray_test_X, BUZZ2_gray_test_Y, num_epochs=epochs, batch_size=batch_size)


        # Train it on BUZZ3
        train_tfl_audio_convnet_model(img_ann, BUZZ3_train_X, BUZZ3_train_Y, BUZZ3_test_X, BUZZ3_test_Y, num_epochs=epochs, batch_size=batch_size)
        # Validate BUZZ1
        bee1_acc = validate_tfl_audio_convnet_model(img_ann, BUZZ1_valid_X, BUZZ1_valid_Y)
        print("BUZZ1", bee1_acc)
        # Validate BUZZ2_gray
        bee2_acc = validate_tfl_audio_convnet_model(img_ann, BUZZ2_gray_valid_X, BUZZ2_gray_valid_Y)
        print("BUZZ2", bee2_acc)
        # Validate BUZZ3
        bee4_acc = validate_tfl_audio_convnet_model(img_ann, BUZZ3_valid_X, BUZZ3_valid_Y)
        print("BUZZ3", bee4_acc)
        mean_acc = (bee1_acc + bee2_acc + bee4_acc) / 3

        # save stats to dictionary
        batch_vs_acc[f"BUZZ1_{batch_size}"] = bee1_acc
        batch_vs_acc[f"BUZZ2_{batch_size}"] = bee2_acc
        batch_vs_acc[f"BUZZ3_{batch_size}"] = bee4_acc

        if mean_acc > max_accuracy:
            img_ann.save("models/aud_cn.tfl")
    for key, value in batch_vs_acc.items():
        print(key, value)
    with open("stats/BUZZ_cn_stats.pickle", "wb") as file:
        pickle.dump(batch_vs_acc, file)
# Execute pipeline
def main():
    pipeline()
if __name__ == '__main__':
    main()
