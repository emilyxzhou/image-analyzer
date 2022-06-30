import cv2
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split

from models import cnn
from tools import load_training_data, scale_pixels, CLASSES

BATCH_SIZE = 16
EPOCHS = 50

X_train, y_train = load_training_data()
X_train = scale_pixels(X_train)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
y_train = to_categorical(y_train)
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

model = cnn()
model.load_weights("recognizer_pixel_operator.h5")

for i in range(0, 10):
    img = X_train[i, :, :]
    test_input = np.reshape(img, (1, 28, 28, 1))
    prediction = model.predict(test_input)
    prediction = CLASSES[np.argmax(prediction)]
    actual = CLASSES[np.argmax(y_train[i])]
    print(f"Prediction: {prediction}; actual: {actual}")
    cv2.imshow("char", img)
    cv2.waitKey(0)
