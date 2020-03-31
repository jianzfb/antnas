import tensorflow as tf
import argparse

from nas.tf_convertor.model import deeplabv3plus
from nas.tf_convertor.config import config_param
from nas.tf_convertor import data_generator

parser = argparse.ArgumentParser(description='test nas on voc')
parser.add_argument('-arch', action='store', type=str)
parser.add_argument('-model_dir', action='store', type=str)


def main():
    args = parser.parse_args()

    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig(
        save_summary_steps=100,
        session_config=session_config,
        keep_checkpoint_max=10,
        log_step_count_steps=10)

    model = tf.estimator.Estimator(
        model_fn=tf.contrib.estimator.replicate_model_fn(deeplabv3plus.deeplabv3_plus_model_fn),
        model_dir=args.model_dir,
        config=run_config,
        params={
            'arch': args.arch
        }
    )

    eval_model = tf.estimator.Estimator(
        model_fn=deeplabv3plus.deeplabv3_plus_model_fn,
        model_dir=args.model_dir,
        config=run_config,
        params={
            'arch': args.arch
        }
    )

    for i in range(config_param.train_epochs // config_param.epochs_per_eval):
        tensors_to_log = {
            'learning_rate': 'learning_rate',
            'cross_entropy': 'cross_entropy',
            'train_px_accuracy': 'train_px_accuracy'
        }

        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=5)
        train_hooks = [logging_hook]
        eval_hooks = None

        tf.logging.info("Start training.")
        model.train(
            input_fn=lambda: data_generator.input_fn(
                is_training=True, batch_size=config_param.batch_size, num_epochs=config_param.epochs_per_eval),
            hooks=train_hooks
        )

        tf.logging.info("Start evaluation: " + args.model_dir)
        eval_results = eval_model.evaluate(
            input_fn=lambda: data_generator.input_fn(is_training=False, batch_size=1, num_epochs=1),
            hooks=eval_hooks
        )
        print(eval_results)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
