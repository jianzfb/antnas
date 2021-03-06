# @Time    : 2020/5/27 19:11
# @Author  : zhangchenming
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


class OutLayer(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan=32, out_chan=2, bias=True):
        super(OutLayer, self).__init__()
        self.conv_1 = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv_2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=bias)

        self.params = {
            'module_list': ['OutLayer'],
            'name_list': ['OutLayer'],
            'OutLayer': {'out_chan': out_chan, 'in_chan': in_chan},
            'out': 'outname',
            'in_chan': in_chan,
            'out_chan': out_chan
        }

    def forward(self, x, sampling=None):
        # B x C x H x W
        x = self.conv_1(x)
        x = torch.nn.functional.interpolate(x,
                                            size=(384,384),
                                            mode='bilinear',
                                            align_corners=True)

        x = self.conv_2(x)

        return x

    def get_flop_cost(self, x):
        return [0] + [0] * (self.state_num - 1)


@antnas_argment
def main(*args, **kwargs):
    if torch.cuda.is_available():
        print('CUDA is OK')

    # start run and set environment
    logger.info('Starting run : {}'.format(kwargs['exp_name']))
    os.environ['CUDA_VISIBLE_DEVICES'] = kwargs['cuda']

    # 获得数据集
    train_loader, val_loader, test_loader, data_properties = \
        get_data(kwargs['dset'], kwargs['bs'], kwargs['path'], kwargs)

    # 创建NAS模型
    nas_manager = Manager(kwargs, data_properties, out_layer=OutLayer(32, 2, True))

    # 构建NAS网络
    nas_manager.build()

    # 构建结构Anchor
    anchors = None
    index = None
    if len(kwargs['anchor_archs']) > 0:
        anchors = Anchors()
        anchors.load(kwargs['anchor_archs'], kwargs['anchor_states'], OutLayer, [int(c) for c in kwargs['cuda'].split(',')])
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
    nas_manager.supernetwork.init(
        shape=(2, data_properties['in_channels'], data_properties['img_dim'], data_properties['img_dim']),
        arc_loss="latency",
        data_loader=test_loader,
        supernetwork_manager=nas_manager)

    nas_manager.supernetwork.draw()

    # initialize parameter
    logging.info('initialize supernetwork parameter')
    if len(kwargs.get('model_path', '')) != 0:
        logging.info('load supernetwork parameter')
        if torch.cuda.is_available():
            nas_manager.supernetwork.to(torch.device("cuda:%d" % int(kwargs['cuda'].split(',')[0])))

        if torch.cuda.is_available():
            nas_manager.supernetwork.load_state_dict(torch.load(kwargs.get('model_path'),
                                                                map_location=torch.device(
                                                                    "cuda:%d" % int(kwargs['cuda'].split(',')[0]))),
                                                     False)
        else:
            nas_manager.supernetwork.load_state_dict(torch.load(kwargs.get('model_path'),
                                                                map_location=torch.device('cpu')), False)
    # set model input
    x = torch.Tensor()
    y = torch.LongTensor()

    if torch.cuda.is_available():
        logger.info('Running with cuda (GPU {})'.format(kwargs['cuda']))
        nas_manager.cuda([int(c) for c in kwargs['cuda'].split(',')])
        x = x.cuda(torch.device("cuda:%d" % int(kwargs['cuda'].split(',')[0])))
        y = y.cuda(torch.device("cuda:%d" % int(kwargs['cuda'].split(',')[0])))
    else:
        logger.warning('Running *WITHOUT* cuda')

    # training and searching iterately
    if kwargs['warmup'] > 0:
        for warmup_epoch in range(kwargs['warmup']):
            for i, (inputs, labels) in enumerate(tqdm(train_loader, desc='Train', ascii=True)):
                # adjust learning rate
                warmup_lr = nas_manager.adjust_lr(kwargs, warmup_epoch, i, len(train_loader), ['path'])
                xp.train.learning_rate.update(warmup_lr)

                # set model status (train)
                x.resize_(inputs.size()).copy_(inputs)
                y.resize_(labels.size()).copy_(labels)

                # train and return predictions, loss, correct
                nas_manager.optimizer.zero_grad()
                loss, model_accuracy, model_sampled_cost, model_pruned_cost = nas_manager.train(x, y,
                                                                                                epoch=warmup_epoch,
                                                                                                warmup=True)

                # update model parameter
                loss.backward()

                # update parameter
                nas_manager.optimizer.step()

    logging.info("training and searching")
    nas_manager.supernetwork.search_init()
    evo_epochs = kwargs['evo_epochs'] if kwargs['iterator_search'] else 1
    for evo_epoch in range(evo_epochs):
        logging.info("training network parameters")
        for epoch in range(kwargs['warmup'], kwargs['epochs'] + kwargs['warmup']):
            logging.info("training network parameters for epoch %d(%d)" % (epoch, kwargs['epochs']))

            # write logger
            logger.info(epoch)

            # train process
            for i, (inputs, labels) in enumerate(tqdm(train_loader, desc='Train', ascii=True)):
                # adjust learning rate
                lr = nas_manager.adjust_lr(kwargs, epoch, i, len(train_loader), ['path'])
                xp.train.learning_rate.update(lr)

                # set model status (train)
                x.resize_(inputs.size()).copy_(inputs)
                y.resize_(labels.size()).copy_(labels)

                # train and return predictions, loss, correct
                nas_manager.optimizer.zero_grad()

                loss, _, a, b = \
                    nas_manager.train(x, y, epoch=epoch, index=index)

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

                            anchor_consistent_loss += torch.mean(
                                (sample_a_in_batch - sample_anchor_node_output_in_batch) ** 2)

                        anchor_consistent_loss /= len(a)
                        anchor_consistent_loss_total += anchor_consistent_loss

                    anchor_consistent_loss_total /= nas_manager.supernetwork.anchors.size()
                    loss += 0.001 * anchor_consistent_loss_total

                # record model loss
                xp.train.classif_loss.update(loss.item())
                logging.info('loss %f'%xp.train.classif_loss.value)

                # update model parameter
                loss.backward()

                # update parameter
                nas_manager.optimizer.step()

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

        logging.info("searching network architecture")
        nas_manager.supernetwork.search(max_generation=1 if kwargs['iterator_search'] else kwargs['evo_epochs'],
                                        population_size=kwargs['population_size'],
                                        epoch=evo_epoch,
                                        folder='./supernetwork/')


if __name__ == '__main__':
    logger.info('Executing main from {}'.format(os.getcwd()))

    # 运行
    main()