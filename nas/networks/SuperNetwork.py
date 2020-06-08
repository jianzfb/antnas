from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


from torch.autograd import Variable
import networkx as nx
from nas.component.ParameterCostEvaluator import ParameterCostEvaluator
from nas.component.LatencyCostEvaluator import LatencyCostEvaluator
from nas.component.ComputationalCostEvaluator import ComputationalCostEvaluator
from nas.component.NetworkBlock import *
from nas.component.PathRecorder import PathRecorder
from nas.networks.mutation import *
from nas.networks.crossover import *
from nas.component.NetworkBlock import *
from nas.networks.nsga2 import *
from nas.networks.bayesian import *
from nas.utils.drawers.NASDrawer import *
import copy
from tqdm import tqdm


class ArchitectureModelProblem(Problem):
  def __init__(self, supernetwork_manager, data_loader, arc_loss=['latency']):
      super(ArchitectureModelProblem, self).__init__()
      self.max_objectives = [None, None]
      self.min_objectives = [None, None]
      self.goal = 'MINIMIZE'
      self.supernetwork_manager = supernetwork_manager
      self.arc_loss = arc_loss
      self.data_loader = data_loader

  def generateIndividual(self):
      individual = Individual()
      individual.features = []
      individual.dominates = functools.partial(self.__dominates, individual1=individual)
      individual.objectives = [None, None]
      individual.values = [None, None]
      return individual

  def calculateObjectives(self, individual):
      individual.objectives[0] = self.__f1(individual)
      individual.objectives[1] = self.__f2(individual)
      for i in range(2):
          if self.min_objectives[i] is None or individual.objectives[i] < self.min_objectives[i]:
              self.min_objectives[i] = individual.objectives[i]

          if self.max_objectives[i] is None or individual.objectives[i] > self.max_objectives[i]:
              self.max_objectives[i] = individual.objectives[i]

  def __dominates(self, individual2, individual1):
      if self.goal == 'MAXIMIZE':
          worse_than_other = individual1.objectives[0] >= individual2.objectives[0] and individual1.objectives[1] >= individual2.objectives[1]
          better_than_other = individual1.objectives[0] > individual2.objectives[0] or individual1.objectives[1] > individual2.objectives[1]
          return worse_than_other and better_than_other
      else:
          worse_than_other = individual1.objectives[0] <= individual2.objectives[0] and individual1.objectives[1] <= individual2.objectives[1]
          better_than_other = individual1.objectives[0] < individual2.objectives[0] or individual1.objectives[1] < individual2.objectives[1]
          return worse_than_other and better_than_other

  def __f1(self, arc):
      # logits, loss, accuracy
      total_correct = 0
      total = 0
      # data, label, architecture
      x, y, a = None, None, None
      if torch.cuda.is_available():
        x = torch.Tensor().cuda(torch.device("cuda:%d"%self.supernetwork_manager.cuda_list[0]))
        y = torch.LongTensor().cuda(torch.device("cuda:%d"%self.supernetwork_manager.cuda_list[0]))
        a = torch.Tensor().cuda(torch.device("cuda:%d"%self.supernetwork_manager.cuda_list[0]))
      else:
        x = torch.Tensor()
        y = torch.LongTensor()
        a = torch.Tensor()

      self.supernetwork_manager.parallel.eval()
      for images, labels in tqdm(self.data_loader, desc='Test', ascii=True):
          x.resize_(images.size()).copy_(images)
          y.resize_(labels.size()).copy_(labels)
          batch_size = labels.shape[0]
          a.resize_([batch_size, len(arc.features)]).copy_(torch.as_tensor(np.tile([arc.features], (batch_size, 1))))

          with torch.no_grad():
              _, accuracy, _, _ = self.supernetwork_manager.parallel(x, y, a)

          total_correct += accuracy.sum()
          total += labels.size(0)
          
      accuracy = total_correct.float().item() / total
      return 1.0 - accuracy

  def __f2(self, arc):
      loss = self.supernetwork_manager.supernetwork.arc_loss(arc.features, self.arc_loss[0])
      loss = loss.numpy()[0]
      return loss


class SuperNetwork(nn.Module):
    _INPUT_NODE_FORMAT = 'I_{}_{}'              # 不可学习
    _OUTPUT_NODE_FORMAT = 'O_{}_{}'             # 不可学习
    _AGGREGATION_NODE_FORMAT = 'A_{}_{}'        # 不可学习
    _CELL_NODE_FORMAT = 'CELL_{}_{}'            # 可学习  (多种状态)
    _TRANSFORMATION_FORMAT = 'T_{}_{}-{}_{}'    # 可学习 （激活/不激活）
    _LINK_FORMAT = 'L_{}_{}-{}_{}'              # 不可学习
    _FIXED_NODE_FORMAT = 'FIXED_{}_{}'          # 不可学习

    def __init__(self, *args, **kwargs):
        super(SuperNetwork, self).__init__()

        self.net = None
        self.traversal_order = None
        self.in_node = None
        self.out_node = None

        self.path_recorder = None
        self._cost_evaluators = None

        self._cost_optimization = kwargs['cost_optimization']
        self._arch_penalty = kwargs['arch_penalty']
        self._objective_cost = kwargs['objective_cost']
        self._objective_method = kwargs['objective_method']
        self._arch_lambda = kwargs['lambda']
        self._objective_comp_max = kwargs.get('max_comp', None)
        self._objective_comp_min = kwargs.get('min_comp', None)

        self._objective_latency_max = kwargs.get('max_latency', None)
        self._objective_latency_min = kwargs.get('min_latency', None)
        
        self._objective_param_max = kwargs.get('max_param', None)
        self._objective_param_min = kwargs.get('min_param', None)

        self.population_size = kwargs.get('population', 100)
        self.cost_evaluation = kwargs["cost_evaluation"][0]
        
        cost_evaluators = {
            'comp': ComputationalCostEvaluator,
            'latency': LatencyCostEvaluator,
            'param': ParameterCostEvaluator
        }

        used_ce = {}
        for k in kwargs["cost_evaluation"]:
            used_ce[k] = cost_evaluators[k](model=self,
                                            main_cost=(k == self._cost_optimization),
                                            **kwargs)

        self._cost_evaluators = used_ce

        # global configure
        self.kwargs = kwargs
        self.use_preload_arch = False
        self.problem = None
        self._anchors = None
        self.plotter = kwargs.get('plotter', None)
        self.drawer = NASDrawer(self.plotter)
        self.visdom_window = {}

    def set_graph(self, network, in_node, out_node):
        self.net = network
        if not nx.is_directed_acyclic_graph(self.net):
            raise ValueError('A Super Network must be defined with a directed acyclic graph')

        self.traversal_order = list(nx.topological_sort(self.net))
        self.in_node = in_node
        self.out_node = out_node

        # TODO Allow several input and/or output nodes
        if self.traversal_order[0] != in_node or self.traversal_order[-1] != out_node:
            raise ValueError('Seems like the given graph is broken')
        
        # save graph
        architecture_path = os.path.join("./supernetwork.architecture")
        nx.write_gpickle(self.graph, architecture_path)

        self.path_recorder = PathRecorder(self.net, self.out_node)

    @property
    def anchors(self):
        return self._anchors
    
    @anchors.setter
    def anchors(self, val):
        self._anchors = val

    def init(self, *args, **kwargs):
        # input shape
        shape = kwargs.get('shape')

        # define problem
        arc_loss = kwargs.get('arc_loss', 'latency')
        data_loader = kwargs.get('data_loader', None)
        supernetwork_manager = kwargs.get('supernetwork_manager', None)
        self.problem = ArchitectureModelProblem(supernetwork_manager,
                                                data_loader,
                                                arc_loss=[arc_loss])

        # copy net structure
        graph = copy.deepcopy(self.net)
        x = torch.ones(shape)

        for node in self.traversal_order:
            graph.node[self.in_node]['input'] = [x]
            cur_node = graph.node[node]
            input = self.format_input(cur_node['input'])

            if len(input) == 0:
                raise RuntimeError('Node {} has no inputs'.format(node))

            out = self.blocks[cur_node['module']](input)
            if node == self.out_node:
                break

            # 3.3.step set successor input
            for succ in graph.successors(node):
                if 'input' not in graph.node[succ]:
                    graph.node[succ]['input'] = []
                graph.node[succ]['input'].append(out)

        # 初始化结构信息
        for cost, cost_eval in self.arch_cost_evaluators.items():
            cost_eval.init_costs(self, graph)

        # 获得约束搜索空间的范围 (find max/min arch)
        auto_analyze_comp = False
        if self.cost_evaluation == "comp" and (self.arch_objective_comp_min < 0 or self.arch_objective_comp_max < 0):
            auto_analyze_comp = True

        auto_analyze_latency = False
        if self.cost_evaluation == "latency" and (self.arch_objective_latency_min < 0 or self.arch_objective_latency_max < 0):
            auto_analyze_latency = True
        
        auto_analyze_param = False
        if self.cost_evaluation == "param" and (self.arch_objective_param_min < 0 or self.arch_objective_param_max < 0):
            auto_analyze_param = True
        
        if auto_analyze_comp or auto_analyze_latency or auto_analyze_param:
            try_times = 1000
            while try_times > 0:
                feature = [None for _ in range(len(self.traversal_order))]
                
                sampling = torch.Tensor()
                active = torch.Tensor()
                for node_name in self.traversal_order:
                    cur_node = self.net.node[node_name]
                    if not (node_name.startswith('CELL') or node_name.startswith('T')):
                        # 不可学习，处于永远激活状态
                        feature[cur_node['sampling_param']] = int(1)
                    else:
                        if not self.blocks[cur_node['module']].structure_fixed:
                            feature[cur_node['sampling_param']] = int(np.random.randint(0, 2))
                        else:
                            feature[cur_node['sampling_param']] = int(np.random.randint(0, NetworkBlock.state_num))
                    
                    sampling, active = \
                        self.path_recorder.add_sampling(node_name,
                                                        torch.as_tensor([feature[cur_node['sampling_param']]]).reshape([1,1,1,1]),
                                                        sampling,
                                                        active,
                                                        self.blocks[cur_node['module']].structure_fixed)
    
                sampled_arc, pruned_arc = \
                    self.path_recorder.get_arch(self.out_node, sampling, active)
    
                for cost, cost_eval in self.arch_cost_evaluators.items():
                    _, pruned_cost = \
                        cost_eval.get_costs([sampled_arc, pruned_arc])
    
                    pruned_cost = pruned_cost.item()
                    if cost == "comp" and auto_analyze_comp:
                        if self.arch_objective_comp_min > pruned_cost or self.arch_objective_comp_min < 0:
                            self.arch_objective_comp_min = pruned_cost
                        if self.arch_objective_comp_max < pruned_cost or self.arch_objective_comp_max < 0:
                            self.arch_objective_comp_max = pruned_cost
                    
                    if cost == "latency" or auto_analyze_latency:
                        if self.arch_objective_latency_min > pruned_cost or self.arch_objective_latency_min < 0:
                            self.arch_objective_latency_min = pruned_cost
                        if self.arch_objective_latency_max < pruned_cost or self.arch_objective_latency_max < 0:
                            self.arch_objective_latency_max = pruned_cost
                    
                    if cost == "param" or auto_analyze_param:
                        if self.arch_objective_param_min > pruned_cost or self.arch_objective_param_min < 0:
                            self.arch_objective_param_min = pruned_cost
                        if self.arch_objective_param_max < pruned_cost or self.arch_objective_param_max < 0:
                            self.arch_objective_param_max = pruned_cost
        
                try_times -= 1
        
        if self.cost_evaluation == "comp":
            print("ARCH COMP MIN-%f,MAX-%f"%(self.arch_objective_comp_min,self.arch_objective_comp_max))
        elif self.cost_evaluation == "latency":
            print("ARCH LATENCY MIN-%f,MAX-%f"%(self.arch_objective_latency_min,self.arch_objective_latency_max))
        elif self.cost_evaluation == "param":
            print("ARCH PARAM MIN-%f,MAX-%f"%(self.arch_objective_param_min,self.arch_objective_param_max))
        
        # launch arc sampling thread
        print('launch archtecture sampling thread')
        supernetwork_manager.launchSamplingArcProcess()
        
    def forward(self, *input):
        raise NotImplementedError

    @property
    def input_size(self):
        if not hasattr(self, '_input_size'):
            raise RuntimeError('SuperNetworks should have an `_input_size` attribute.')
        return self._input_size

    @staticmethod
    def format_input(input):
        if (isinstance(input, tuple) or isinstance(input, list)) and len(input) == 1:
            input = input[0]
        return input

    def arch_optimize(self, *args, **kwargs):
        raise NotImplementedError

    def loss(self, predictions, labels):
        raise NotImplementedError

    def accuray(self, predictions, labels):
        raise NotImplementedError

    @property
    def arch_cost_evaluators(self):
        return self._cost_evaluators

    @property
    def arch_cost_optimization(self):
        return self._cost_optimization

    @property
    def arch_penalty(self):
        return self._arch_penalty

    @property
    def arch_objective_cost(self):
        return self._objective_cost

    @property
    def arch_objective_comp_max(self):
        return self._objective_comp_max
    
    @arch_objective_comp_max.setter
    def arch_objective_comp_max(self, val):
        self._objective_comp_max = val

    @property
    def arch_objective_comp_min(self):
        return self._objective_comp_min
    
    @arch_objective_comp_min.setter
    def arch_objective_comp_min(self, val):
        self._objective_comp_min = val
    
    @property
    def arch_objective_latency_max(self):
        return self._objective_latency_max
    
    @arch_objective_latency_max.setter
    def arch_objective_latency_max(self, val):
        self._objective_latency_max = val
    
    @property
    def arch_objective_latency_min(self):
        return self._objective_latency_min
    
    @arch_objective_latency_min.setter
    def arch_objective_latency_min(self, val):
        self._objective_latency_min = val
    
    @property
    def arch_objective_param_max(self):
        return self._objective_param_max
    
    @arch_objective_param_max.setter
    def arch_objective_param_max(self, val):
        self._objective_param_max = val

    @property
    def arch_objective_param_min(self):
        return self._objective_param_min
    
    @arch_objective_param_min.setter
    def arch_objective_param_min(self, val):
        self._objective_param_min = val
    
    @property
    def arch_objective_method(self):
        return self._objective_method

    @property
    def arch_lambda(self):
        return self._arch_lambda

    @property
    def arch_node_index(self):
        return self.path_recorder.node_index

    def search_and_save(self, folder=None, name=None):
        pass

    def search_and_plot(self):
        pass

    def get_path_recorder(self):
        return self.path_recorder

    def get_node_sampling(self, node_name, batch_size, batched_sampling):
        """
        Get a batch of sampling for the given node on the given output.
        Fires a "sampling" event with the node name and the sampling Variable.
        :param node_name: Name of the node to sample
        :param out: Tensor on which the sampling will be applied
        :return: A Variable brodcastable to out's size, with all dimensions equals to one except the first one (batch)
        """
        sampling_dim = [batch_size] + [1] * 3
        node = self.net.node[node_name]
        node_sampling = batched_sampling[:, node['sampling_param']].contiguous().view(sampling_dim)

        return node_sampling

    def load_static_arch(self, arch_path):
        graph = nx.read_gpickle(arch_path)
        travel = list(nx.topological_sort(graph))

        for node_index, node_name in enumerate(travel):
            self.net.node[node_name]['sampled'] = graph.node[node_name]['sampled']

        self.use_preload_arch = True

    def set_static_arch(self, arch_sample):
        for node_index, node in enumerate(self.traversal_order):
            self.net.node[node]['sampled'] = arch_sample[node_index]

        self.use_preload_arch = True

    def sample_arch(self, *args, **kwargs):
        # support Thread reentry
        raise NotImplementedError
    
    def is_satisfied_constraint(self, feature):
        return True
    
    def sample_arch_and_save(self, **kwargs):
        folder = kwargs.get('folder', './supernetwork/')
        arch_suffix = kwargs.get('suffix', "0")
        if not os.path.exists(folder):
            os.makedirs(folder)

        feature = self.sample_arch()
        batched_sampling = torch.Tensor(feature).view(1, len(feature))
        graph = copy.deepcopy(self.net)

        # 1.step prune sampling network
        sampling = torch.Tensor()
        active = torch.Tensor()

        for node in self.traversal_order:
            cur_node = graph.node[node]
            node_sampling = self.get_node_sampling(node, 1, batched_sampling)

            # notify path recorder to add sampling
            sampling, active = self.path_recorder.add_sampling(node,
                                                               node_sampling,
                                                               sampling,
                                                               active,
                                                               self.blocks[cur_node['module']].structure_fixed)

        _, pruned_arch = self.path_recorder.get_arch(self.out_node, sampling, active)

        # 2.step write to graph
        for node in self.traversal_order:
            node_sampling_val = torch.squeeze(pruned_arch[self.path_recorder.node_index[node]]).item()
            graph.node[node]['sampled'] = int(node_sampling_val)

        architecture_path = os.path.join(folder, "anchor_arch_%s.architecture"%arch_suffix)
        nx.write_gpickle(graph, architecture_path)

    def arc_loss(self, arc, name):
        sampling = torch.Tensor()
        active = torch.Tensor()
        for node_name in self.traversal_order:
            cur_node = self.net.node[node_name]
            sampling, active = \
                self.path_recorder.add_sampling(node_name,
                                                torch.as_tensor([arc[cur_node['sampling_param']]]).reshape(
                                                    [1, 1, 1, 1]),
                                                sampling,
                                                active,
                                                self.blocks[cur_node['module']].structure_fixed)

        sampled_arc, pruned_arc = \
            self.path_recorder.get_arch(self.out_node, sampling, active)

        sampled_cost, pruned_cost = \
            self.arch_cost_evaluators[name].get_costs([sampled_arc, pruned_arc])

        return pruned_cost

    def _evolution_callback_func(self, population, generation_index, arc_loss, folder):
        # 1.step draw pareto front image
        print('render pareto front( population size %d )'%len(population))
        # draw pareto front
        plt.figure()
        plt.title('PARETO OPTIMAL for %d GENERATION' % (generation_index))
        x = [individual.objectives[1] for individual in population]
        if arc_loss == 'comp':
            x = [v / 1000000 for v in x]
        elif arc_loss == 'param':
            x = [v / 1000000 for v in x]

        # axis y (objective)
        y = [individual.objectives[0] for individual in population]

        x_min = np.min(x)
        x_max = np.max(x)

        y_min = np.min(y)
        y_max = np.max(y)

        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))

        if arc_loss == 'comp':
            plt.xlabel('MULADD/FLOPS (M - 10e6)')
        elif arc_loss == 'latency':
            plt.xlabel('LATENCY (ms)')
        else:
            plt.xlabel("PARAMETER (M - 10e6)")

        plt.ylabel('ERROR')
        plt.scatter(x=x, y=y, c='r', marker='o')

        for x_p, y_p, individual in zip(x, y, population):
            plt.text(x_p, y_p, '%0.4f'%(1.0-individual.objectives[0]))

        if folder is None:
            folder = './'

        plt.savefig(os.path.join(folder, 'generation_%d_pareto_optimal.png'%generation_index))
        plt.close()

        # 2.step show population architecture info distribution (on visdom)
        if self.plotter is not None:
            title = ""
            if arc_loss == 'comp':
                title = 'MULADD/FLOPS DISTRIBUTION'
            elif arc_loss == 'latency':
                title = "LATENCY(ms) DISTRIBUTION"
            else:
                title = "PARAMETER NUMBER DISTRIBUTION"

            arch_error_list = [individual.objectives[0] for individual in population]
            arch_loss_list = [individual.objectives[1] for individual in population]

            # 2.1.step architecture loss distribution (直方图)
            search_arc_loss_distribution_win = \
                self.plotter.viz.histogram(np.array(arch_loss_list),
                                           opts=dict(
                                               title=title,
                                               numbins=1000,
                                           ),
                                           win=None if "search_arc_loss_distribution" not in self.visdom_window else self.visdom_window['search_arc_loss_distribution'])
            if 'search_arc_loss_distribution' not in self.visdom_window:
                self.visdom_window['search_arc_loss_distribution'] = \
                    search_arc_loss_distribution_win
            
            # 2.2.step pareto front (标签的散点图)
            search_arc_pareto_front_win = \
                self.plotter.viz.scatter(
                    X=np.concatenate([np.array(arch_loss_list).reshape((-1, 1)), np.array(arch_error_list).reshape((-1, 1))], axis=1),
                    Y=np.arange(1, len(population)+1),
                    opts=dict(
                        legend=['ARCH_%d'%i for i in range(1, len(population)+1)],
                        title="PARETO FRONT",
                    ),
                    win=None if "search_arc_pareto_front" not in self.visdom_window else self.visdom_window['search_arc_pareto_front']
                )
            if 'search_arc_pareto_front' not in self.visdom_window:
                self.visdom_window['search_arc_pareto_front'] = search_arc_pareto_front_win

            # 2.3.step architecture information （热点图）
            GENE = []
            for i in range(len(population)):
                feature = [population.population[i].features[self.net.node[node_name]['sampling_param']] for node_name in self.traversal_order]
                GENE.append(feature)

            search_arc_gene_chrome_win = \
                self.plotter.viz.heatmap(
                    np.array(GENE),
                    opts=dict(
                        title="POLUTATION GENE CHROME",
                        columnnames=[node_name for node_name in self.traversal_order],
                        rownames=['ARCH_%d'%index for index in range(1, len(population)+1)],
                        colormap='Electric',
                    ),
                    win=None if "search_arc_gene_chrome" not in self.visdom_window else self.visdom_window['search_arc_gene_chrome']
                )
            if 'search_arc_gene_chrome' not in self.visdom_window:
                self.visdom_window['search_arc_gene_chrome'] = search_arc_gene_chrome_win
            
            # 2.4.step structure sampling visualization
            data = np.array(GENE)
            data = data.astype(np.int32)
            for node_index, node_name in enumerate(self.traversal_order):
                counts = np.bincount(data[:, node_index])
                sampling_val = np.argmax(counts)
                self.net.node[node_name]['sampling_val'] = sampling_val

            self.draw()
    
    def draw(self, net=None):
        net = net if net is not None else self.net
        self.drawer.draw(net)
        
    def hierarchical(self):
        # get network hierarchical
        return []
        
    def search_init(self, *args, **kwargs):
        pass
    
    def search(self, *args, **kwargs):
        max_generation = kwargs.get('max_generation', 100)
        population_size = kwargs.get('population_size', 50)
        crossover_multi_points = kwargs.get('corssover_multi_points', 5)
        mutation_multi_points = kwargs.get('mutation_multi_points', -1)
        folder = kwargs.get('folder', './supernetwork/')

        if not os.path.exists(folder):
            os.makedirs(folder)

        # build nsga2 evolution algorithm
        mutation_control = EvolutionMutation(multi_points=mutation_multi_points,
                                             max_generation=max_generation,
                                             k0=1.0,
                                             k1=1.5,
                                             method='based_matrices',
                                             adaptive=True,
                                             network=self)
        crossover_control = EvolutionCrossover(multi_points=crossover_multi_points,
                                               max_generation=max_generation,
                                               k0=1.0,
                                               k1=0.8,
                                               method='based_matrices',
                                               size=population_size,
                                               network=self)
        evolution = Nsga2(self.problem,
                          mutation_control,
                          crossover_control,
                          num_of_generations=max_generation,
                          callback=functools.partial(self._evolution_callback_func,
                                                     arc_loss=self.problem.arc_loss,
                                                     folder=folder),
                          using_bayesian=True)

        # ###############################
        # # 临时代码，测试结构参数量统计
        # print('step 1')
        # features = self.sample_arch()
        # parameter_num = 0
        # for node in self.traversal_order:
        #     cur_node = self.net.node[node]
        #     sampled_state = features[cur_node['sampling_param']]
        #     cur_node = self.net.node[node]
        #     parameter_num += self.blocks[cur_node['module']].get_param_num(None)[sampled_state]
        #
        # print('step 1 loss %d', parameter_num)
        #
        # print('step 2')
        # loss = self.arc_loss(features, 'param')
        # print('step 2 loss %d', loss)
        #
        # ###############################

        # 1.step init population
        population = Population()
        for individual_index in range(population_size):
            individual = evolution.problem.generateIndividual()
            individual.features = self.sample_arch()
            evolution.problem.calculateObjectives(individual)
            population.population.append(individual)

        # 2.step evolution to find parate front
        explore_position = []
        for node_name in self.traversal_order:
            cur_node = self.net.node[node_name]
            if node_name.startswith('CELL') or node_name.startswith('T'):
                explore_position.append(cur_node['sampling_param'])
                
        # 2.1.step add hierarchical info
        # stage/block/cell
        hierarchical = self.hierarchical()

        elited_population = \
            evolution.evolve(population,
                             explore_position=explore_position,
                             hierarchical=hierarchical,
                             network=self)

        # 3.step save architecture
        for individual in elited_population:
            batched_sampling = torch.Tensor(individual.features).view(1, len(individual.features))

            # 3.1.step prune sampling network
            sampling = torch.Tensor()
            active = torch.Tensor()

            for node in self.traversal_order:
                cur_node = self.net.node[node]
                node_sampling = self.get_node_sampling(node, 1, batched_sampling)

                # notify path recorder to add sampling
                sampling, active = self.path_recorder.add_sampling(node,
                                                                   node_sampling,
                                                                   sampling,
                                                                   active,
                                                                   self.blocks[cur_node['module']].structure_fixed)

            _, pruned_arch = self.path_recorder.get_arch(self.out_node, sampling, active)

            # 3.2.step write to graph
            for node in self.traversal_order:
                node_sampling_val = torch.squeeze(pruned_arch[self.path_recorder.node_index[node]]).item()
                self.net.node[node]['sampled'] = int(node_sampling_val)

            # 3.3.step get architecture parameter number
            parameter_num = 0
            for node in self.traversal_order:
                sampled_state = self.net.node[node]['sampled']
                cur_node = self.net.node[node]
                parameter_num += self.blocks[cur_node['module']].get_param_num(None)[sampled_state]

            # 3.4.step save architecture
            architecture_info = ''
            if self.problem.arc_loss[0] == 'comp':
                architecture_info = 'flops_%d' % int(individual.objectives[1])
            elif self.problem.arc_loss[0] == 'latency':
                architecture_info = 'latency_%0.2f' % individual.objectives[1]
            else:
                architecture_info = 'para_%d' % int(individual.objectives[1])

            architecture_tag = 'accuray_%0.4f_%s_params_%d.architecture' % (1.0-individual.objectives[0],
                                                                            architecture_info,
                                                                            int(parameter_num))
            architecture_path = os.path.join(folder, architecture_tag)
            nx.write_gpickle(self.net, architecture_path)
