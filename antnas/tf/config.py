# @Time    : 2019-05-07 14:34
# @Author  : zhangchenming

from easydict import EasyDict
import math

config_param = EasyDict()

# config.model_dir = './checkpoint'
config_param.dataset_dir = '/home/work/zhangchenming/voc2012/tfrecord'

config_param.num_train_images = 10582
config_param.batch_size = 32
config_param.train_crop_size = [512, 512]
config_param.min_scale_factor = 0.5
config_param.max_scale_factor = 2
config_param.scale_factor_step_size = 0.25
config_param.mean_rgb = [127.5, 127.5, 127.5]

config_param.training_number_of_steps = 30000
config_param.train_epochs = math.ceil(config_param.training_number_of_steps * config_param.batch_size / config_param.num_train_images)
config_param.epochs_per_eval = config_param.train_epochs

config_param.ignore_label = 255
config_param.num_classes = 21

config_param.base_learning_rate = 0.007

config_param.learning_power = 0.9
config_param.momentum = 0.9
config_param.weight_decay = 0.00004
