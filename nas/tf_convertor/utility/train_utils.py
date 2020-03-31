# @Time    : 2019-05-09 14:18
# @Author  : zhangchenming
import tensorflow as tf
from nas.tf_convertor.config import config_param


def get_cross_entry_loss(logits, labels, num_classes, ignore_label):
    labels_flat = tf.reshape(labels, [-1, ])
    not_ignore_mask = tf.cast(tf.not_equal(labels_flat, ignore_label), tf.float32)
    one_hot_labels = tf.one_hot(labels_flat, num_classes, on_value=1.0, off_value=0.0)
    loss = tf.losses.softmax_cross_entropy(
        one_hot_labels,
        tf.reshape(logits, shape=[-1, num_classes]),
        weights=not_ignore_mask
    )
    return loss


def get_init_fn_for_scaffold(ignore_missing_vars=True):
    if tf.train.latest_checkpoint(config_param.model_dir):
        tf.logging.info('Ignoring --checkpoint_path because a checkpoint already exists in %s.' % config_param.model_dir)
        return None

    exclusion_scopes = []
    if config_param.exclude_scopes:
        exclusion_scopes = [scope.strip() for scope in config_param.exclude_scopes.split(',')]

    variables_to_restore = []
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        excluded = False
        for exclusion in exclusion_scopes:
            if exclusion in var.op.name:  # .startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    # if checkpoint_model_scope is not None:
    #     if checkpoint_model_scope.strip() == '':
    #         variables_to_restore = {var.op.name.replace(model_scope + '/', ''): var for var in variables_to_restore}
    #     else:
    #         variables_to_restore = {var.op.name.replace(model_scope, checkpoint_model_scope.strip()): var for var in
    #                                 variables_to_restore}

    if tf.gfile.IsDirectory(config_param.pretrained_dir):
        checkpoint_path = tf.train.latest_checkpoint(config_param.pretrained_dir)
    else:
        checkpoint_path = config_param.pretrained_dir

    tf.logging.info('Fine-tuning from %s. Ignoring missing vars: %s.' % (checkpoint_path, ignore_missing_vars))

    if not variables_to_restore:
        raise ValueError('variables_to_restore cannot be empty')

    if ignore_missing_vars:
        reader = tf.train.NewCheckpointReader(checkpoint_path)
        if isinstance(variables_to_restore, dict):
            var_dict = variables_to_restore
        else:
            var_dict = {var.op.name: var for var in variables_to_restore}
        available_vars = {}
        for var in var_dict:
            if reader.has_tensor(var):
                available_vars[var] = var_dict[var]
            else:
                tf.logging.warning('Variable %s missing in checkpoint %s.', var, checkpoint_path)
        variables_to_restore = available_vars

    if variables_to_restore:
        saver = tf.train.Saver(variables_to_restore, reshape=False)
        saver.build()

        def callback(scaffold, session):
            saver.restore(session, checkpoint_path)

        return callback
    else:
        tf.logging.warning('No Variables to restore.')
        return None
