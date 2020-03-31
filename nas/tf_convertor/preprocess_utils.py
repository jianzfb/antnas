import tensorflow as tf


def get_random_scale(min_scale_factor, max_scale_factor, step_size):
    if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
        raise ValueError('Unexpected value of min_scale_factor.')

    if min_scale_factor == max_scale_factor:
        return tf.cast(min_scale_factor, tf.float32)

    # When step_size = 0, we sample the value uniformly from [min, max).
    if step_size == 0:
        return tf.random_uniform([1], minval=min_scale_factor, maxval=max_scale_factor)

    # When step_size != 0, we randomly select one discrete value from [min, max].
    num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
    scale_factors = tf.lin_space(min_scale_factor, max_scale_factor, num_steps)
    shuffled_scale_factors = tf.random_shuffle(scale_factors)
    return shuffled_scale_factors[0]


def randomly_scale_image_and_label(image, label, scale=1.0):
    if scale == 1.0:
        return image, label

    image_shape = tf.shape(image)
    new_dim = tf.cast(tf.cast([image_shape[0], image_shape[1]], tf.float32) * scale, tf.int32)
    image = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(image, 0), new_dim, align_corners=True), [0])
    label = tf.squeeze(tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_dim, align_corners=True), [0])

    return image, label


def random_crop_or_pad(image, label, crop_height, crop_width, mean_pixel, ignore_label):
    image = image - mean_pixel
    label = label - ignore_label
    label = tf.cast(label, tf.float32)
    image_and_label = tf.concat([image, label], axis=-1)

    image_height = tf.shape(image_and_label)[0]
    image_width = tf.shape(image_and_label)[1]
    channel = tf.shape(image_and_label)[2]

    image_and_label = tf.image.pad_to_bounding_box(
        image_and_label, 0, 0,
        tf.maximum(crop_height, image_height),
        tf.maximum(crop_width, image_width)
    )
    image_and_label = tf.random_crop(image_and_label, [crop_height, crop_width, channel])

    image = image_and_label[:, :, :3]
    image += mean_pixel

    label = image_and_label[:, :, 3:]
    label += ignore_label
    label = tf.cast(label, tf.int32)

    return image, label


def flip_dim(tensor_list, prob=0.5, dim=1):
    random_value = tf.random_uniform([])

    def flip():
        flipped = []
        for tensor in tensor_list:
            if dim < 0 or dim >= len(tensor.get_shape().as_list()):
                raise ValueError('dim must represent a valid dimension.')
            flipped.append(tf.reverse_v2(tensor, [dim]))
        return flipped

    is_flipped = tf.less_equal(random_value, prob)
    outputs = tf.cond(is_flipped, flip, lambda: tensor_list)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    outputs.append(is_flipped)

    return outputs
