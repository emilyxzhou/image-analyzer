import csv
import numpy as np


def load_a_z_dataset(file_path):
    # initialize the list of data and labels
    data = []
    labels = []
    # loop over the rows of the A-Z handwritten digit dataset
    for row in open(file_path):
        # parse the label and image from the row
        row = row.split(",")
        label = int(row[0])
        image = np.array([int(x) for x in row[1:]], dtype="uint8")
        # images are represented as single channel (grayscale) images
        # that are 28x28=784 pixels -- we need to take this flattened
        # 784-d list of numbers and reshape them into a 28x28 matrix
        image = image.reshape((28, 28))
        # update the list of data and labels
        data.append(image)
        labels.append(label)
    # convert the data and labels to NumPy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels, dtype="int")
    # return a 2-tuple of the A-Z data and labels
    return data, labels


if __name__ == "__main__":
    import os
    cwd = os.getcwd()
    a_z_data_path = os.path.join(
        cwd, "..", "..", "data", "ocr_data", "A_Z_handwritten.csv"
    )
    a_z_data, a_z_labels = load_a_z_dataset(a_z_data_path)
    print(len(a_z_data))
    print(a_z_data[0].shape)
    print(len(a_z_labels))
    print(a_z_labels[0].shape)
