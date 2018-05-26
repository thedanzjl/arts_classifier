
import tensorflow as tf


IMAGE_SIZE = 224
MODEL_FILENAME = 'art-model'
LEARNING_RATE = 1e-4


def parser(record):
    keys_to_features = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        "label":     tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    # image = tf.reshape(image, shape=[IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.cast(image, tf.float32)
    tf.summary.image('image', image)
    label = tf.cast(parsed["label"], tf.int32)

    return {'image': image}, label



def input_fn(filenames, shuffle_size):
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.shuffle(buffer_size=shuffle_size).repeat(5)
    dataset = dataset.map(map_func=parser)
    dataset = dataset.batch(batch_size=32)
    dataset = dataset.prefetch(buffer_size=shuffle_size)
    return dataset


def train_input_fn():
    return input_fn(filenames=["training.tfrecords"], shuffle_size=2000)


def val_input_fn():
    return input_fn(filenames=["validation.tfrecords"], shuffle_size=500)


def model_fn(features, labels, mode, params):
    num_classes = 5
    image = features["image"]

    # this architecture:   http://image-net.org/challenges/LSVRC/2012/supervision.pdf

    net = tf.reshape(image, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                           filters=96, kernel_size=11, strides=(4, 4),
                           padding='same', activation=tf.nn.relu)

    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                           filters=256, kernel_size=3,
                           padding='same', activation=tf.nn.relu)

    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    net = tf.layers.conv2d(inputs=net, name='layer_conv3',
                           filters=384, kernel_size=3,
                           padding='same', activation=tf.nn.relu)

    net = tf.layers.conv2d(inputs=net, name='layer_conv4',
                           filters=384, kernel_size=3,
                           padding='same', activation=tf.nn.relu)

    net = tf.layers.conv2d(inputs=net, name='layer_conv5',
                           filters=256, kernel_size=3,
                           padding='same', activation=tf.nn.relu)

    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)


    net = tf.contrib.layers.flatten(net)

    net = tf.layers.dense(inputs=net, name='layer_fc1',
                          units=4096, activation=tf.nn.relu)

    net = tf.layers.dropout(inputs=net)


    net = tf.layers.dense(inputs=net, name='layer_fc2',
                          units=4096, activation=tf.nn.relu)

    net = tf.layers.dropout(inputs=net)

    net = tf.layers.dense(inputs=net, name='layer_fc3',
                          units=num_classes)

    y_pred = tf.nn.softmax(logits=net)

    y_pred_cls = tf.argmax(y_pred, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred)
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=net)
        loss = tf.reduce_mean(cross_entropy)

        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        metrics = {
            "accuracy": tf.metrics.accuracy(labels, y_pred_cls)
        }

        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)
        
    return spec


def train_and_evaluate():

    run_config = tf.estimator.RunConfig(
        model_dir=MODEL_FILENAME,
        save_summary_steps=100,
        save_checkpoints_steps=100
    )
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params={"learning_rate": LEARNING_RATE},
        config=run_config
    )
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=2000
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=val_input_fn, steps=100, throttle_secs=300
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.estimator.RunConfig(model_dir=MODEL_FILENAME)

    train_and_evaluate()

