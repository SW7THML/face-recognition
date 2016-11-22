from .core import VGGFace
from keras.layers import Dense
from keras.optimizers import SGD
from .utils import Maximizer

class FaceRecognitionClassifier:
  def __init__(self, course_users):
    self.model = None
    self.course_users = course_users

  def fit(self, X, y):
    self.model = VGGFace()
    self.model.layers.pop()
    self.model.outputs = [self.model.layers[-1].output]
    self.model.layers[-1].outbound_nodes = []
    self.model.add(Dense(len(self.course_users), activation='softmax', name='new_fc8'))
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    self.model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=["accuracy"])

    # train
    self.model.fit(X, y, batch_size=32, nb_epoch=2, shuffle=True, verbose=1)

  def predict(self, X):
    prediction = self.model.predict(X)

    # hungarian
    maximizer = Maximizer()
    total, coord = maximizer.hungarian(prediction)

    return [y for _, y in coord]

  def load_weights(self, weight_path):
    self.model = VGGFace(weight_path, freeze=True, classes=len(self.course_users))

  def save_weights(self, weight_path):
    self.model.save_weights(weight_path)




