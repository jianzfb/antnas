# -*- coding: UTF-8 -*-
# @Time    : 2019-08-19 18:22
# @File    : imagenet_train_and_search.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import logging

import antvis.client.mlogger as mlogger
from antnas.dataset.datasets import get_data
from antnas.manager import *
from antnas.networks.Anchors import *
from antnas.utils.drawers.NASDrawer import *
from antnas.utils.argument_parser import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageNetOutLayer(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan):
        super(ImageNetOutLayer, self).__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.conv_1 = nn.Conv2d(in_chan, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_1 = nn.BatchNorm2d(960,
                                   momentum=1.0 if not NetworkBlock.bn_moving_momentum else 0.1,
                                   track_running_stats=NetworkBlock.bn_track_running_stats)

        self.conv_2 = nn.Conv2d(960, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.classifier = nn.Linear(1280, 1000)
        self.dropout = torch.nn.Dropout(p=0.9)

        self.params = {
            'module_list': ['ImageNetOutLayer'],
            'name_list': ['ImageNetOutLayer'],
            'ImageNetOutLayer': {'in_chan': in_chan},
            'out': 'outname',
            'in_chan': in_chan,
        }

    def forward(self, x, sampling=None):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = F.relu(x)

        x = self.global_pool(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


@antnas_argment
def main(*args, **kwargs):
    if torch.cuda.is_available():
        print('CUDA is OK')

    # start run and set environment
    logger.info('Starting run : {}'.format(kwargs['exp_name']))
    os.environ['CUDA_VISIBLE_DEVICES'] = kwargs['cuda']

    # 获得数据集
    kwargs.update({'img_size': 224, 'in_channels': 3, 'out_channels': 1000})
    train_loader, val_loader, test_loader, data_properties = \
        get_data(kwargs['dset'], kwargs['bs'], kwargs['path'], kwargs)

    # 创建NAS模型
    nas_manager = Manager(kwargs, data_properties, out_layer=ImageNetOutLayer(320))

    # pk mobilenetv2
    nas_manager.build(blocks_per_stage=[1, 1, 1, 2, 2],
                      cells_per_block=[[2], [2], [2], [2, 2], [2, 2]],
                      channels_per_block=[[8], [12], [24], [48, 96], [112, 160]])

    # # pk mobilenetv2
    # nas_manager.build(blocks_per_stage=[1, 1, 2, 2, 2],
    #                   cells_per_block=[[2], [2], [2, 2], [3, 3], [2, 2]],
    #                   channels_per_block=[[8], [12], [24, 32], [64, 96], [112, 160]])

    # 构建结构Anchor
    anchors = None
    index = None
    if len(kwargs['anchor_archs']) > 0:
        anchors = Anchors()
        anchors.load(kwargs['anchor_archs'], kwargs['anchor_states'], ImageNetOutLayer, [int(c) for c in kwargs['cuda'].split(',')])
        index = torch.as_tensor(list(range(kwargs['bs'])))

    nas_manager.supernetwork.anchors = anchors

    # logger initialize
    xp = mlogger.Container()
    xp.epoch = mlogger.metric.Simple(plot_title='epoch')

    xp.train = mlogger.Container()
    xp.train.classif_loss = mlogger.metric.Average(plot_title="classif_loss")
    xp.train.accuracy = mlogger.metric.Simple(plot_title="accuracy")
    xp.train.rewards = mlogger.metric.Average(plot_title="rewards")
    xp.train.objective_cost = mlogger.metric.Average(plot_title="objective_cost")
    xp.train.learning_rate = mlogger.metric.Simple(plot_title="LR mode(%s)"%kwargs['lr_decay'])

    xp.test = mlogger.Container()
    xp.test.accuracy = mlogger.metric.Simple(plot_title="val_test_accuracy")

    for cost in kwargs['cost_evaluation']:
        xp.train.__setattr__('train_sampled_%s' % cost,
                             mlogger.metric.Average(plot_title='train_%s' % cost))
        xp.train.__setattr__('train_pruned_%s' % cost,
                             mlogger.metric.Average(plot_title='train_%s' % cost))

    # initialize supernetwork
    logging.info('initialize supernetwork basic info')
    nas_manager.init(shape=(2, data_properties['in_channels'], data_properties['img_dim'], data_properties['img_dim']),
                     arc_loss=kwargs['cost_evaluation'][0],
                     data_loader=test_loader,
                     supernetwork_manager=nas_manager,
                     architecture=kwargs['architecture'])

    # initialize parameter
    logging.info('initialize supernetwork parameter')
    if len(kwargs.get('model_path', '')) != 0:
        logging.info('load supernetwork parameter')
        if torch.cuda.is_available():
            nas_manager.supernetwork.to(torch.device("cuda:%d" % int(kwargs['cuda'].split(',')[0])))

        if torch.cuda.is_available():
            nas_manager.supernetwork.load_state_dict(torch.load(kwargs.get('model_path'),
                                                                map_location=torch.device("cuda:%d"%int(kwargs['cuda'].split(',')[0]))), False)
        else:
            nas_manager.supernetwork.load_state_dict(torch.load(kwargs.get('model_path'),
                                                                map_location=torch.device('cpu')), False)
    # set model input
    x = torch.Tensor()
    y = torch.LongTensor()

    if torch.cuda.is_available():
        logger.info('Running with cuda (GPU {})'.format(kwargs['cuda']))
        nas_manager.cuda([int(c) for c in kwargs['cuda'].split(',')])
        x = x.cuda(torch.device("cuda:%d"%int(kwargs['cuda'].split(',')[0])))
        y = y.cuda(torch.device("cuda:%d"%int(kwargs['cuda'].split(',')[0])))
    else:
        logger.warning('Running *WITHOUT* cuda')

    # training and searching iterately
    # 只针对迭代搜索模式下使用
    if kwargs['warmup'] > 0:
        logging.info("[warmup] training")
        for warmup_epoch in range(kwargs['warmup']):
            for i, (inputs, labels) in enumerate(tqdm(train_loader, desc='Train', ascii=True)):
                # adjust learning rate
                warmup_lr = \
                    nas_manager.adjust_lr(kwargs, warmup_epoch, i, len(train_loader))
                xp.train.learning_rate.update(warmup_lr)

                # set model status (train)
                x = inputs
                y = labels

                # train and return predictions, loss, correct
                _, model_out, _, _ = \
                    nas_manager.train(x, y, epoch=warmup_epoch)
                loss = nas_manager.criterion(model_out, y)

                # reset grad zero
                nas_manager.optimizer.zero_grad()

                # update model parameter
                loss.backward()

                # update parameter
                nas_manager.optimizer.step()

        # 保存warmup参数
        nas_manager.supernetwork.search_and_save('./supernetwork/', 'supernetwork_warmup_state')

    logging.info("[search] training and searching")
    nas_manager.supernetwork.search_init(max_generation=kwargs['max_generation'],
                                         population_size=kwargs['population_size'],
                                         folder='./supernetwork/')
    evo_epochs = kwargs['evo_epochs']
    for evo_epoch in range(evo_epochs):
        logging.info("training network parameters in evo_epoch %d"%evo_epoch)
        for epoch in range(kwargs['warmup'], kwargs['epochs']+kwargs['warmup']):
            logging.info("training network parameters for epoch %d(%d)"%(epoch, kwargs['epochs']))

            # write logger
            logger.info('epoch %d in (evo %d)'%(epoch-kwargs['warmup'], evo_epoch))

            # adjust learning rate
            lr = nas_manager.adjust_lr(kwargs, evo_epoch*kwargs['epochs']+epoch, 0, len(train_loader), ['path'])
            xp.train.learning_rate.update(lr)
            logger.info('lr %f in epoch %d'%((float)(lr), epoch))

            # training architecture parameter
            for i, (inputs, labels) in enumerate(tqdm(train_loader, desc='Train', ascii=True)):
                # set model status (train)
                x = inputs
                y = labels

                _, model_out, a, b = \
                    nas_manager.train(x, y, epoch=epoch, index=index)
                loss = nas_manager.criterion(model_out, y)

                # train anchor arch network
                if nas_manager.supernetwork.anchors is not None:
                    anchor_x = x
                    anchor_y = y
                    nas_manager.supernetwork.anchors.run(anchor_x, anchor_y)

                # anchor loss
                anchor_consistent_loss_total = 0.0
                if nas_manager.supernetwork.anchors is not None:
                    b = b.cpu().numpy()
                    for anchor_index in range(nas_manager.supernetwork.anchors.size()):
                        sample_index_in_batch = np.where(b == anchor_index)[0]
                        if sample_index_in_batch.size == 0:
                            continue

                        anchor_consistent_loss = 0.0
                        for k, v in a.items():
                            anchor_node_output = nas_manager.supernetwork.anchors.output(anchor_index, k)

                            sample_a_in_batch = v[sample_index_in_batch, :, :, :]
                            sample_anchor_node_output_in_batch = anchor_node_output[sample_index_in_batch, :, :, :]

                            anchor_consistent_loss += torch.mean((sample_a_in_batch-sample_anchor_node_output_in_batch)**2)

                        anchor_consistent_loss /= len(a)
                        anchor_consistent_loss_total += anchor_consistent_loss

                    anchor_consistent_loss_total /= nas_manager.supernetwork.anchors.size()
                    loss += 0.001 * anchor_consistent_loss_total

                # record model loss
                xp.train.classif_loss.update(loss.item())

                # train and return predictions, loss, correct
                nas_manager.optimizer.zero_grad()

                # update model parameter
                loss.backward()

                # update parameter
                nas_manager.optimizer.step()

                mlogger.update()

            # eval process
            if epoch % 1 == 0:
                # check model accuracy
                accuracy_val = nas_manager.eval(x, y, test_loader, 'test')
                xp.test.accuracy.update(float(accuracy_val))
                logging.info('epoch %d - accuracy %f' % (epoch,xp.test.accuracy.value))

            # save model state
            nas_manager.supernetwork.search_and_plot('./supernetwork/')
            nas_manager.supernetwork.search_and_save('./supernetwork/',
                                                     'supernetwork_state_%d' % (epoch % kwargs['latest_num']))

        logging.info("searching network architecture in evo_epoch %d"%evo_epoch)
        if kwargs['max_generation'] > 0:
            nas_manager.supernetwork.search(max_generation=kwargs['max_generation'],
                                            population_size=kwargs['population_size'],
                                            era=evo_epoch,
                                            folder='./supernetwork/')
            # 重置内部状态
            nas_manager.reset()


if __name__ == '__main__':
    logger.info('Executing main from {}'.format(os.getcwd()))

    # 运行
    main()