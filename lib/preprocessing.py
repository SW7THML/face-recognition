from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import glob
import numpy as np

class ImageAugmentator:
  def __init__(self, count=40):
    self.count = count

  def augment(self, X, y):
    datagen = ImageDataGenerator(
                rotation_range=25,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.4,
                zoom_range=0.1,
                channel_shift_range=0.4,
                horizontal_flip=True,
                fill_mode='nearest')

    aug_X = []
    aug_y = []

    for idx, x in enumerate(X):
      x = x.reshape((1,) + x.shape)

      step = 0
      for X_batch in datagen.flow(x, batch_size=1, save_format='jpg'):
        step += 1

        aug_X.append(X_batch.reshape(X_batch.shape[1:]))
        aug_y.append(y[idx])

        if step > self.count:
          break

    aug_X = np.asarray(aug_X)
    aug_y = np.asarray(aug_y)

    return aug_X, aug_y




