import cv2
import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf

IMAGE_SIZE = 224


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    if img is None:
        return None
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

 
def create_data_record(out_filename, addrs, labels):
    print('\nprocessing {}...\n'.format(out_filename))
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in tqdm.tqdm(range(len(addrs))):
        # Load the image
        img = load_image(addrs[i])
        # cv2.imshow('ImageWindow', img)
        # cv2.waitKey()

        label = labels[i]

        if img is None:
            continue

        # Create a feature
        feature = {
            'image_raw': _bytes_feature(img.tostring()),
            'label': _int64_feature(label)
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        
    writer.close()


if __name__ == '__main__':
    with open('training.csv', 'r') as train:
        decoded = pd.read_csv(train, header=-1)
        decoded = decoded.reindex(np.random.permutation(decoded.index)).as_matrix()
        train_addrs, train_labels = decoded[:, 0], decoded[:, 1]
    with open('validation.csv', 'r') as val:
        decoded = pd.read_csv(val, header=-1)
        decoded = decoded.reindex(np.random.permutation(decoded.index)).as_matrix()
        val_addrs, val_labels = decoded[:, 0], decoded[:, 1]

    create_data_record('training.tfrecords', train_addrs, train_labels)
    create_data_record('validation.tfrecords', val_addrs, val_labels)