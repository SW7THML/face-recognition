from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

TF_WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v1.0/rcmalli_vggface_tf_weights_tf_ordering.h5'

def VGGFace(weight_path=None, freeze=False, classes=2622):
  input_shape = (224, 224, 3)
   
  # Block 1
  model = Sequential()
  model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv1_1', input_shape=input_shape))
  model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv1_2'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool1'))

  # Block 2
  model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv2_1'))
  model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv2_2'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool2'))

  # Block 3
  model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_1'))
  model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_2'))
  model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_3'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool3'))

  # Block 4
  model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_1'))
  model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_2'))
  model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_3'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool4'))

  # Block 5
  model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_1'))
  model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_2'))
  model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_3'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool5'))

  # Classification block
  model.add(Flatten(name='flatten'))
  model.add(Dense(4096, activation='relu', name='fc6'))
  model.add(Dense(4096, activation='relu', name='fc7'))
  model.add(Dense(classes, activation='softmax', name='fc8')) # VGGFace Class Size

  if weight_path:
    model.load_weights(weight_path)

  if freeze:
    for layer in model.layers:
      layer.trainable = False

  return model




  