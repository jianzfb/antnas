# -*- coding: UTF-8 -*-
# @Time    : 2019-08-19 18:22
# @File    : imagenet_train_and_search.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import logging

import mlogger
from antnas.dataset.datasets import get_data
from antnas.manager import *
from antnas.utils.misc import *
from antnas.networks.Anchors import *
from antnas.utils.drawers.NASDrawer import *
from OutLayerFactory import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def argument_parser():
    parser = argparse.ArgumentParser(description='Budgeted Super Networks')

    # Experience
    parser.add_argument('-exp-name', action='store', default='', type=str, help='Experience Name')
    # Model
    parser.add_argument('-arch', action='store', default='PKImageNetSN', type=str)
    parser.add_argument('-deter_eval', action='store', default=False, type=bool,
                        help='Take blocks with probas >0.5 instead of sampling during evaluation')

    # Training
    parser.add_argument('-path', default='/Users/jian/Downloads/dataset', type=str,
                        help='path for the execution')

    parser.add_argument('-dset', default='ImageNet', type=str, help='Dataset')
    parser.add_argument('-bs', action='store', default=256, type=int, help='Size of each batch')
    parser.add_argument('-epochs', action='store', default=150, type=int,
                        help='Number of training epochs')
    parser.add_argument('-evo_epochs', action='store', default=50, type=int,
                        help='Number of architecture searching epochs')
    parser.add_argument('-warmup', action='store', default=0, type=int,
                        help='warmup epochs before searching architecture')
    parser.add_argument('-iterator_search', action='store', default=False, type=bool,
                        help='is iterator search')
    parser.add_argument('-population_size', action='store', default=50, type=int,
                        help='population size for NSGAII')

    parser.add_argument('-optim', action='store', default='SGD', type=str,
                        help='Optimization method')
    parser.add_argument('-nesterov', action='store', default=False, type=bool,
                        help='Use Nesterov for SGD momentum')
    parser.add_argument('-lr', action='store', default=0.05, type=float, help='Learning rate')
    parser.add_argument('-path_lr', action='store', default=1e-3, type=float, help='path learning rate')
    
    parser.add_argument('--lr_decay', type=str, default='cos',
                        help='mode for learning rate decay')
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='decrease learning rate at these epochs.')

    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--epochs_drop', type=int, default=10, help='for step mode')

    parser.add_argument('-momentum', action='store', default=0.9, type=float,
                        help='momentum used by the optimizer')
    parser.add_argument('-wd', dest='weight_decay', action='store', default=1e-4, type=float,
                        help='weight decay used during optimisation')

    parser.add_argument('-latest_num', action='store', default=1, type=int,help='save the latest number model state')
    
    parser.add_argument('-cuda', action='store', default='0,1,2,3,4,5,6,7', type=str,
                        help='Enables cuda and select device')
    parser.add_argument('-latency', action='store', default='./latency.gpu.855.224_16.32.64.96.112.160_lookuptable.json',type=str,
                        help='latency lookup table')

    parser.add_argument('-draw_env', default='ImageNetPK-1', type=str, help='Visdom drawing environment')

    parser.add_argument('-static_proba', action='store', default=-1, type=restricted_float(0, 1),
                        help='sample a static binary weight with given proba for each stochastic Node.')
    parser.add_argument('-static_arc', dest='', action='store', default=[], type=list, help='fixed architecture')

    parser.add_argument('-np', '--n_parallel', dest='n_parallel', action='store', default=3, type=int,
                        help='Maximum number of module evaluation in parallel')
    parser.add_argument('-ce', '-cost_evaluation', dest='cost_evaluation', action='store',
                        default=['param'],
                        type=restricted_list('comp', 'latency', 'param'))
    parser.add_argument('-co', dest='cost_optimization', action='store', default='latency',
                        type=restricted_str('comp', 'latency', 'param'))

    parser.add_argument('-lambda', dest='lambda', action='store', default=1e-7, type=float,
                        help='Constant balancing the ratio classifier loss/architectural loss')
    parser.add_argument('-oc', dest='objective_cost', action='store', default=60000000000, type=float,
                        help='Maximum allowed cost for architecture')
    parser.add_argument('-max_comp', dest='max_comp', action='store', default=-1, type=float,
                        help='Maximum allowed cost for architecture')
    parser.add_argument('-min_comp', dest='min_comp', action='store', default=-1, type=float,
                        help='Maximum allowed cost for architecture')
    
    parser.add_argument('-max_latency', dest='max_latency', action='store', default=-1, type=float,
                        help='Maximum allowed cost for architecture')
    parser.add_argument('-min_latency', dest='min_latency', action='store', default=-1, type=float,
                        help='Maximum allowed cost for architecture')
    
    parser.add_argument('-max_param', dest='max_param', action='store', default=-1, type=float,
                        help='Maximum allowed cost for architecture')
    parser.add_argument('-min_param', dest='min_param', action='store', default=-1, type=float,
                        help='Maximum allowed cost for architecture')
    
    parser.add_argument('-om', dest='objective_method', action='store', default='max',
                        type=restricted_str('max', 'abs'), help='Method used to compute the cost of an architecture')
    parser.add_argument('-pen', dest='arch_penalty', action='store', default=0, type=float,
                        help='Penalty for inconsistent architecture')
    parser.add_argument('-model_path', dest="model_path", action='store', default="", type=str)

    parser.add_argument('-anchor_archs', dest="anchor_archs", action='store', default=[], type=list)
    parser.add_argument('-anchor_states', dest="anchor_states", action='store', default=[], type=list)
    return parser.parse_known_args()[0]


def main(args, plotter):
    if torch.cuda.is_available():
        print('CUDA is OK')

    # start run and set environment
    logger.info('Starting run : {}'.format(args['exp_name']))
    os.environ['CUDA_VISIBLE_DEVICES'] = args['cuda']

    # 获得数据集
    train_loader, val_loader, test_loader, data_properties = \
        get_data(args['dset'], args['bs'], args['path'], args)

    # 设置visdom plotter
    args.update({'plotter': plotter})

    # 创建NAS模型
    nas_manager = Manager(args, data_properties, out_layer=ImageNetOutLayer(320, 1280, 1000))

    # pk mobilenetv2
    nas_manager.build(blocks_per_stage=[1, 1, 1, 2, 2],
                      cells_per_block=[[2], [2], [3], [4, 4], [3, 3]],
                      channels_per_block=[[16], [24], [32], [64, 96], [160, 320]])

    # 构建结构Anchor
    anchors = None
    if len(args['anchor_archs']) > 0:
        anchors = Anchors()
        anchors.load(args['anchor_archs'], args['anchor_states'], ImageNetOutLayer, [int(c) for c in args['cuda'].split(',')])

    nas_manager.supernetwork.anchors = anchors

    # logger initialize
    xp = mlogger.Container()
    xp.config = mlogger.Config(plotter=plotter)
    xp.epoch = mlogger.metric.Simple()

    xp.train = mlogger.Container()
    xp.train.classif_loss = mlogger.metric.Average(plotter=plotter, plot_title="classif_loss", plot_legend="train")
    xp.train.accuracy = mlogger.metric.Average(plotter=plotter, plot_title="accuracy", plot_legend="train")
    xp.train.rewards = mlogger.metric.Average(plotter=plotter, plot_title="rewards",plot_legend="train")
    xp.train.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend="train")
    xp.train.objective_cost = mlogger.metric.Average(plotter=plotter, plot_title="objective_cost", plot_legend="architecture")
    xp.train.learning_rate = mlogger.metric.Simple(plotter=plotter, plot_title="LR", plot_legend="train")

    xp.test = mlogger.Container()
    xp.test.accuracy = mlogger.metric.Simple(plotter=plotter, plot_title="val_test_accuracy", plot_legend="test")
    xp.test.timer = mlogger.metric.Timer(plotter=plotter, plot_title="Time", plot_legend="test")

    for cost in args['cost_evaluation']:
        xp.train.__setattr__('train_sampled_%s'%cost,
                            mlogger.metric.Average(plotter=plotter,
                                                   plot_title='train_%s'%cost,
                                                   plot_legend="sampled_cost"))
        xp.train.__setattr__('train_pruned_%s'%cost,
                             mlogger.metric.Average(plotter=plotter,
                                                    plot_title='train_%s'%cost,
                                                    plot_legend="pruned_cost"))
    # initialize supernetwork
    logging.info('initialize supernetwork basic info')
    nas_manager.supernetwork.init(shape=(2, data_properties['in_channels'], data_properties['img_dim'], data_properties['img_dim']),
                                  arc_loss="param",
                                  data_loader=test_loader,
                                  supernetwork_manager=nas_manager)

    nas_manager.supernetwork.draw()

    # initialize parameter
    logging.info('initialize supernetwork parameter')
    if len(args.get('model_path', '')) != 0:
        logging.info('load supernetwork parameter')
        if torch.cuda.is_available():
            nas_manager.supernetwork.to(torch.device("cuda:%d" % int(args['cuda'].split(',')[0])))

        if torch.cuda.is_available():
            nas_manager.supernetwork.load_state_dict(torch.load(args.get('model_path'),
                                                                map_location=torch.device("cuda:%d"%int(args['cuda'].split(',')[0]))), False)
        else:
            nas_manager.supernetwork.load_state_dict(torch.load(args.get('model_path'),
                                                                map_location=torch.device('cpu')), False)
    # set model input
    x = torch.Tensor()
    y = torch.LongTensor()

    if torch.cuda.is_available():
        logger.info('Running with cuda (GPU {})'.format(args['cuda']))
        nas_manager.cuda([int(c) for c in args['cuda'].split(',')])
        x = x.cuda(torch.device("cuda:%d"%int(args['cuda'].split(',')[0])))
        y = y.cuda(torch.device("cuda:%d"%int(args['cuda'].split(',')[0])))
    else:
        logger.warning('Running *WITHOUT* cuda')

    # training and searching iterately
    if args['warmup'] > 0:
        for warmup_epoch in range(args['warmup']):
            for i, (inputs, labels) in enumerate(tqdm(train_loader, desc='Train', ascii=True)):
                # adjust learning rate
                warmup_lr = nas_manager.adjust_lr(args, warmup_epoch, i, len(train_loader), logger)
                xp.train.learning_rate.update(warmup_lr)

                # set model status (train)
                x.resize_(inputs.size()).copy_(inputs)
                y.resize_(labels.size()).copy_(labels)

                # train and return predictions, loss, correct
                nas_manager.optimizer.zero_grad()
                loss, model_accuracy, model_sampled_cost, model_pruned_cost = \
                    nas_manager.train(x, y, epoch=warmup_epoch, warmup=True)

                # update model parameter
                loss.backward()

                # update parameter
                nas_manager.optimizer.step()

    logging.info("training and searching")
    nas_manager.supernetwork.search_init()
    evo_epochs = args['evo_epochs'] if args['iterator_search'] else 1
    for evo_epoch in range(evo_epochs):
        logging.info("training network parameters")
        for epoch in range(args['warmup'], args['epochs']+args['warmup']):
            logging.info("training network parameters for epoch %d(%d)"%(epoch, args['epochs']))

            # write logger
            logger.info(epoch)
            
            # training architecture parameter
            for i, (inputs, labels) in enumerate(tqdm(train_loader, desc='Train', ascii=True)):
                # adjust learning rate
                lr = nas_manager.adjust_lr(args, epoch, i, len(train_loader), logger)
                xp.train.learning_rate.update(lr)
                
                # set model status (train)
                x.resize_(inputs.size()).copy_(inputs)
                y.resize_(labels.size()).copy_(labels)

                # train and return predictions, loss, correct
                nas_manager.optimizer.zero_grad()

                loss, model_accuracy, a, b = \
                    nas_manager.train(x, y, epoch=epoch)

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
                # record model accuracy
                xp.train.accuracy.update(model_accuracy * 100 / float(inputs.size(0)))

                # update model parameter
                loss.backward()

                # update parameter
                nas_manager.optimizer.step()

                xp.train.timer.update()
                for metric in xp.train.metrics():
                    metric.log()

                plotter.update_plots()

            # test random sampling architecture accuracy
            if (epoch + 1) % 2 == 0:
                acc_score = nas_manager.eval(x, y, test_loader, 'Test')
                xp.test.accuracy.update(acc_score)
                xp.test.timer.update()
                for metric in xp.test.metrics():
                    metric.log()
                    
                logging.info("accuracy %f on test dataset after epoch %d" % (acc_score, epoch))

            # save model state
            logger.info("plot and save search space")
            nas_manager.supernetwork.search_and_plot('./supernetwork/')
            nas_manager.supernetwork.search_and_save('./supernetwork/',
                                                     'supernetwork_state_%d'%(epoch%args['latest_num']))

        logging.info("searching network architecture")
        nas_manager.supernetwork.search(max_generation=1 if args['iterator_search'] else args['evo_epochs'],
                                        population_size=args['population_size'],
                                        epoch=evo_epoch,
                                        folder='./supernetwork/')


if __name__ == '__main__':
    logger.info('Executing main from {}'.format(os.getcwd()))
    args = vars(argument_parser())

    plotter = mlogger.VisdomPlotter({'env': args['draw_env'],
                                     'server': 'http://localhost',
                                     'port': 8097},
                                    manual_update=True)
    main(args, plotter)
