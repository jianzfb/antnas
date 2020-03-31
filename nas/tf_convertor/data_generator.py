import os
import tensorflow as tf
from nas.tf_convertor import preprocess_utils
from nas.tf_convertor.config import config_param


def get_filenames(is_training, data_dir):
    if is_training:
        return [os.path.join(data_dir, 'trainaug-*')]
    else:
        return [os.path.join(data_dir, 'val-*')]


def parse_function(example_proto):
    features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'label/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'label/format':
            tf.FixedLenFeature((), tf.string, default_value='png'),
    }
    parsed_features = tf.parse_single_example(example_proto, features)

    image = tf.image.decode_jpeg(parsed_features['image/encoded'], channels=3)
    image = tf.cast(image, tf.float32)

    label = tf.image.decode_png(parsed_features['label/encoded'], channels=1)
    label = tf.cast(label, tf.int32)

    return image, label


def preprocess_image(image, label):
    scale = preprocess_utils.get_random_scale(
        config_param.min_scale_factor, config_param.max_scale_factor, config_param.scale_factor_step_size)
    image, label = preprocess_utils.randomly_scale_image_and_label(image, label, scale)

    mean_pixel = tf.reshape(config_param.mean_rgb, [1, 1, 3])
    image, label = preprocess_utils.random_crop_or_pad(
        image, label, config_param.train_crop_size[0], config_param.train_crop_size[1],
        mean_pixel, config_param.ignore_label)

    image, label, _ = preprocess_utils.flip_dim(
        [image, label], 0.5, dim=1)

    return image, label


def input_fn(is_training, batch_size, num_epochs):
    filenames = get_filenames(is_training, config_param.dataset_dir)
    files = tf.gfile.Glob(filenames)
    dataset = tf.data.TFRecordDataset(files)

    if is_training:
        dataset = dataset.shuffle(buffer_size=config_param.num_train_images)

    dataset = dataset.map(parse_function)

    dataset = dataset.map(preprocess_image)

    dataset = dataset.prefetch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
   
    return images, labels
