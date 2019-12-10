from __future__ import division, absolute_import
import cv2
import numpy as np
import tflearn
import os

FACIAL_EXPRESSIONS = ["neutral", "anger", "disgust", "fear", "happy", "sadness", "surprise"]
HAARCASCADE_PATH = os.path.abspath("") + "/haarcascade_frontalface_default.xml"


class NN:
  def __init__(self):
    pass

  def build_nn(self):
      self.nn = tflearn.layers.core.input_data(shape = [None, 40, 40, 1])
      self.nn = tflearn.layers.conv.conv_2d(self.nn, 64, 5, activation = "relu")
      self.nn = tflearn.layers.conv.max_pool_2d(self.nn, 3, strides = 2)
      self.nn = tflearn.layers.conv.conv_2d(self.nn, 64, 5, activation = "relu")
      self.nn = tflearn.layers.conv.max_pool_2d(self.nn, 3, strides = 2)
      self.nn = tflearn.layers.conv.conv_2d(self.nn, 128, 4, activation = "relu")
      self.nn = tflearn.layers.core.dropout(self.nn, 0.3)
      self.nn = tflearn.layers.core.fully_connected(self.nn, 3072, activation = "relu")
      self.nn = tflearn.layers.core.fully_connected(self.nn, len(FACIAL_EXPRESSIONS), activation = "softmax")
      self.nn = tflearn.layers.estimator.regression(self.nn,optimizer = "momentum",metric = "accuracy",loss = "categorical_crossentropy")
      self.model = tflearn.DNN(self.nn,checkpoint_path = "model",max_checkpoints = 1,tensorboard_verbose = 2)
      self.load_model()


  def load_model(self):
    if not os.path.isfile("model.tflearn.meta"):
        return None
    else:
      self.model.load("model.tflearn")


  def predict(self, photo):
    if photo is not None:
      photo = photo.reshape([-1, 40, 40, 1])
      return self.model.predict(photo)
    else:
        return None


def bound_box(photo):
    photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    cascade_classifier = cv2.CascadeClassifier(HAARCASCADE_PATH)
    photo = cv2.resize(photo, (40,40), interpolation = cv2.INTER_CUBIC) / 255
    return photo


nn = NN()
nn.build_nn()


if __name__ == "__main__":
    while True:
        # Classifier to draw bounding box around face
        _, img = cv2.VideoCapture(0).read()
        facecasc = cv2.CascadeClassifier(HAARCASCADE_PATH)
        face = facecasc.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.3, 5)

        # Calculating the network prediction
        newtork_prediction = nn.predict(bound_box(img))
        if newtork_prediction is not None:
            # Put text different FACIAL_EXPRESSIONS with soft max numbers
            for counter, expression in enumerate(FACIAL_EXPRESSIONS):
                cv2.putText(img, expression, (4, counter * 20 + 20), cv2.cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(img, "{0:.5f}".format(newtork_prediction[0][counter]), (85, counter * 20 + 20), cv2.cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            # Predict emotion with maximum probability (max value)
            cv2.putText(img,"Predicted emotion is: " + FACIAL_EXPRESSIONS[np.argmax(newtork_prediction[0])],(60,450),
                        cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),1,cv2.LINE_AA)

            (x,y,w,h) = face[0]
            img = cv2.rectangle(img,(x,y-50),(x+w,y+h+10),(0,255,0),1)

        cv2.imshow("Real-time Facial Expression Detection with automated Face-Filters", cv2.resize(img,None,fx=1,fy=1))
        wait_key = cv2.waitKey(20)
        if wait_key == 27:
            break

    cv2.VideoCapture(0).release()
    cv2.destroyWindow("Real-time Facial Expression Detection with automated Face-Filters")
