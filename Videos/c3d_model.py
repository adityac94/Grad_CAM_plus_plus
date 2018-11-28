from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD

'''
dim_ordering issue:
- 'th'-style dim_ordering: [batch, channels, depth, height, width]
- 'tf'-style dim_ordering: [batch, depth, height, width, channels]
'''

def get_model(summary=False, backend='tf'):
    """ Return the Keras model of the network
    """
    model = Sequential()
    if backend == 'tf':
        input_shape=(16, 112, 112, 3) # l, h, w, c
    else:
        input_shape=(3, 16, 112, 112) # c, l, h, w
    model.add(Convolution3D(64, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv1',
                            input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3a'))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4a'))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5a'))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropad5'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(487, activation='softmax', name='fc8'))

    if summary:
        print(model.summary())

    return model

def get_int_model(model, layer, backend='tf'):

    if backend == 'tf':
        input_shape=(16, 112, 112, 3) # l, h, w, c
    else:
        input_shape=(3, 16, 112, 112) # c, l, h, w

    int_model = Sequential()

    int_model.add(Convolution3D(64, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv1',
                            input_shape=input_shape,
                            weights=model.layers[0].get_weights()))
    if layer == 'conv1':
        return int_model
    int_model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='valid', name='pool1'))
    if layer == 'pool1':
        return int_model

    # 2nd layer group
    int_model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv2',
                            weights=model.layers[2].get_weights()))
    if layer == 'conv2':
        return int_model
    int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool2'))
    if layer == 'pool2':
        return int_model

    # 3rd layer group
    int_model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3a',
                            weights=model.layers[4].get_weights()))
    if layer == 'conv3a':
        return int_model
    int_model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3b',
                            weights=model.layers[5].get_weights()))
    if layer == 'conv3b':
        return int_model
    int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool3'))
    if layer == 'pool3':
        return int_model

    # 4th layer group
    int_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4a',
                            weights=model.layers[7].get_weights()))
    if layer == 'conv4a':
        return int_model
    int_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4b',
                            weights=model.layers[8].get_weights()))
    if layer == 'conv4b':
        return int_model
    int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool4'))
    if layer == 'pool4':
        return int_model

    # 5th layer group
    int_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5a',
                            weights=model.layers[10].get_weights()))
    if layer == 'conv5a':
        return int_model
    int_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5b',
                            weights=model.layers[11].get_weights()))
    if layer == 'conv5b':
        return int_model
    int_model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropad'))
    int_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool5'))
    if layer == 'pool5':
        return int_model

    int_model.add(Flatten())
    # FC layers group
    int_model.add(Dense(4096, activation='relu', name='fc6',
                            weights=model.layers[15].get_weights()))
    if layer == 'fc6':
        return int_model
    int_model.add(Dropout(.5))
    int_model.add(Dense(4096, activation='relu', name='fc7',
                            weights=model.layers[17].get_weights()))
    if layer == 'fc7':
        return int_model
    int_model.add(Dropout(.5))
    int_model.add(Dense(487, activation='softmax', name='fc8',
                            weights=model.layers[19].get_weights()))
    if layer == 'fc8':
        return int_model

    return None

if __name__ == '__main__':
    model = get_model(summary=True)
