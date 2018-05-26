
import os
import cv2
import sys
from train import *
from termcolor import colored
from create_dataset import _bytes_feature
from random import shuffle
from generate_csv import labels, dataset_folder

model = tf.estimator.Estimator(model_dir='art-model',
                               model_fn=model_fn, params={"learning_rate": LEARNING_RATE})


def predict_input_fn(addr):
    img = cv2.imread(addr)
    if img is None:
        return None
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feature = {'image_raw': _bytes_feature(img.tostring())}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    written = example.SerializeToString()
    keys_to_features = {"image_raw": tf.FixedLenFeature([], tf.string)}
    parsed = tf.parse_single_example(written, keys_to_features)
    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    return {'image': image}


def predict(filename, top=3):
    result = next(model.predict(input_fn=lambda: predict_input_fn(filename)))
    # return labels[result]
    return sorted(list(zip(labels, result)), key=lambda x: -x[1])[:top]

def main():
    if len(sys.argv) == 1:
        dirs = os.listdir(os.path.join(dataset_folder, 'validation_set'))
        for dir in dirs:
            images = os.listdir(os.path.join(dataset_folder, 'validation_set', dir))
            shuffle(images)
            for image in images:
                if (image == '' or image is None) or (
                        not image.endswith('.jpeg') and not image.endswith('.jpg') and not image.endswith('.png')):
                    continue
                image = os.path.join(dataset_folder, 'validation_set', dir, image)
                read_image = cv2.imread(image)
                predicted = predict(image)
                color = 'green' if predicted[0][0] == dir else 'red'
                result = colored('\nTrue: {}; predicted: {}\n'.format(dir, predict(image)), color)
                print(result)
                cv2.imshow('predict playground', read_image)
                cv2.waitKey()
    elif sys.argv[1] == '--image' and sys.argv[2]:
        image = sys.argv[2]
        result = predict(image)
        print(result)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    main()

