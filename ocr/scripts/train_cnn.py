import cv2
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split

from models import cnn
from tools import load_training_data, scale_pixels, CHECKPOINTS_FOLDER, CLASSES

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

# print(y_val[0])
# cv2.imshow("frame", X_val[0])
# cv2.waitKey(0)

# create model
model = cnn()
# set up checkpoints
model_path = os.path.join(CHECKPOINTS_FOLDER, "weights-{epoch:02d}-{val_accuracy:.2f}.hdf5")
checkpoint = ModelCheckpoint(model_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
callbacks_list = [checkpoint]

model.fit(
    x=X_train, y=y_train,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks_list,
    validation_data=(X_val, y_val)
)

model.save_weights("recognizer_pixel_operator.h5")

for i in range(0, 7):
    img = X_test[i, :, :]
    test_input = np.reshape(img, (1, 28, 28, 1))
    prediction = model.predict(test_input)
    prediction = CLASSES[np.argmax(prediction)]
    actual = CLASSES[np.argmax(y_test[i])]
    print(f"Prediction: {prediction}; actual: {actual}")
    cv2.imshow("char", img)
    cv2.waitKey(0)
