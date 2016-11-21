from scipy import misc
import copy
import random
import numpy as np
from munkres import Munkres, print_matrix, make_cost_matrix

class Loader:
  def load_image(self, image_path):
    im = misc.imread(image_path)
    im = misc.imresize(im, (224, 224)).astype(np.float32)
    aux = copy.copy(im)
    
    im[:, :, 0] = aux[:, :, 2]
    im[:, :, 2] = aux[:, :, 0]
    
    # Remove image mean
    im[:, :, 0] -= 93.5940
    im[:, :, 1] -= 104.7624
    im[:, :, 2] -= 129.1863

    return im

  def load_data(self, data):
    random.seed(1337)
    data = random.sample(data, len(data))
    
    X = []
    y = []
    
    for datum in data:
      image = load_image(datum['path'])

    X = [load_image(datum['path']) for datum in data]
    y = [datum['label'] for datum in data]
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    return X, y

class Maximizer:
  def hungarian(self, matrix):
    munkres = Munkres()

    cost_matrix = make_cost_matrix(matrix, lambda cost: 1 - cost)
    indexes = munkres.compute(cost_matrix)

    total = 0
    coord = []
    
    for row, column in indexes:
      value = matrix[row][column]
      total += value

      coord.append([row, column])
        
    return total, coord




