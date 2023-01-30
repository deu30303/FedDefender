import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random

from functools import reduce
from collections import OrderedDict
from utils import *
from defence_utils import *
from resnetcifar import *

def get_args():
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('--model', type=str, default='resnet18', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=20, help='number of workers in a distributed cluster')
    parser.add_argument('--n_parties_per_round', type=int, default=10, help='number of workers in a distributed cluster per round')
    parser.add_argument('--alg', type=str, default='defender',help='training strategy: defender/vanilla')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, 
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--temperature', type=float, default=2, help='the temperature parameter for knowledge distillation')
    parser.add_argument('--attacker_type', type=str, default='untargeted_diverse', help='attacker type (either untargeted or untargeted_diverse)')
    parser.add_argument('--attacker_ratio', type=float, default=0.2, help='ratio for number of attackers')
    parser.add_argument('--noise_ratio', type=float, default=0.8, help='noise ratio for label flipping (0 to 1)')
    parser.add_argument('--global_defense', type=str, default='residual',help='communication strategy: average/median/krum/norm/residual')
    
    return parser.parse_args()


def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset == 'cifar10':
        n_classes = 10
    elif args.dataset == 'cifar100':
        n_classes = 100
        
    for net_i in range(n_parties):
        if args.alg == 'vanilla':
            net = ResNet18_cifar(class_num=n_classes)
        elif args.alg == 'defender':
            net = ResNet18_SD_cifar(class_num=n_classes)

        if device == 'cpu':
            net.to(device)
        else:
            net = net.cuda()
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    net.cuda()

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    
    criterion = nn.CrossEntropyLoss().cuda()

    for epoch in range(epochs):
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()
            
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, _ = compute_accuracy(net, test_dataloader, device=device)
    net.to('cpu')

    return train_acc, test_acc

def train_net_defender(net_id, net, global_model, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    net.cuda()
    global_model.cuda()

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()
    cos=torch.nn.CosineSimilarity(dim=-1).cuda()
    kl_criterion = nn.KLDivLoss(reduction="batchmean").cuda()
    
    if args.dataset == 'cifar10':
        class_num = 10
    elif args.dataset == 'cifar100':
        class_num = 100
    
    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            target = target.long()
            optimizer.zero_grad()
            outputs, SD_outputs,  feats = net(x, get_feat=True, SD=True)
            SD_p_output = F.softmax(SD_outputs / args.temperature, dim=1)
            SD_logp = F.log_softmax(SD_outputs / args.temperature, dim=1)
            p_output = F.softmax(outputs / args.temperature,dim=1)
            logp_output = F.log_softmax(outputs / args.temperature,dim=1)
            
            with torch.no_grad():
                logp_global = global_model(x) 
                logp_global = F.softmax(logp_global / args.temperature, dim=1)
                logp_global = logp_global.detach()
            
            alpha = cos(logp_global, F.one_hot(target, num_classes=class_num)).unsqueeze(1)
            targer_g = (1-alpha)*F.one_hot(target, num_classes=class_num) + alpha*logp_global
            loss_gkd = -torch.mean(torch.sum(SD_logp* targer_g, dim=1))
            loss = criterion(outputs, target) + loss_gkd + kl_criterion(logp_output, SD_p_output.detach())
            
            loss.backward(retain_graph=True)
            targets_fast = target.clone()
            randidx = torch.randperm(target.size(0))
            for n in range(int(target.size(0)*0.5)):
                num_neighbor = 10
                idx = randidx[n]
                feat = feats[idx]
                feat.view(1,feat.size(0))
                feat.data = feat.data.expand(target.size(0),feat.size(0))
                dist = torch.sum((feat-feats)**2,dim=1)
                _, neighbor = torch.topk(dist.data,num_neighbor+1,largest=False)
                targets_fast[idx] = target[neighbor[random.randint(1,num_neighbor)]]

            fast_loss = criterion(outputs,targets_fast)
            grads = torch.autograd.grad(fast_loss, net.parameters(), create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)

            for grad in grads:
                if grad == None:
                    continue
                grad = grad.detach()
                grad.requires_grad = False  


            fast_weights = OrderedDict((name, param - args.lr*grad) for ((name, param), grad) in zip(net.named_parameters(), grads) if grad !=None)
            fast_out, SD_fast_out = net(x,fast_weights, SD=True)  

            logp_fast = F.log_softmax(fast_out, dim=1)
            meta_loss = criterion(fast_out, target)
            meta_loss.backward()

            optimizer.step()


        
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, _ = compute_accuracy(net, test_dataloader, device=device)
    net.to('cpu')

    return train_acc, test_acc



def local_train_net(nets, args, net_dataidx_map, attacker_id_list=[], train_dl=None, 
                    test_dl=None, global_model = None, round=None, device="cpu"):
    avg_acc = 0.0
    acc_list = []
    if global_model:
        global_model.cuda()

    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]

        if net_id in attacker_id_list:
            prefix = 'attacker'
            if args.attacker_type == 'untargeted':
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, 
                                                                     attacker_type=args.attacker_type, noise_ratio=args.noise_ratio)
            elif args.attacker_type == 'untargeted_diverse':
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, 
                                                                     attacker_type=args.attacker_type, noise_ratio=args.noise_ratio,
                                                                     perturb_probs=perturb_prob_dict[net_id]) 
        else:
            prefix = 'normal'
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs
        
        if net_id in attacker_id_list:
            trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args, device=device)
        else:
            if args.alg == 'vanilla':
                trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args, device=device)
            elif args.alg == 'defender':
                trainacc, testacc = train_net_defender(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer, args, device=device)
                
        logger.info(prefix + " net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        acc_list.append(testacc)
    avg_acc /= args.n_parties
    
    if global_model:
        global_model.to('cpu')
    return nets




def global_aggregation(nets_this_round, args, fed_avg_freqs, global_w, party_list_this_round):
    if args.global_defense == 'average':
        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if net_id == 0:
                for key in net_para:
                    global_w[key] = net_para[key] / len(nets_this_round)
            else:
                for key in net_para:
                    global_w[key] += net_para[key] / len(nets_this_round)
    elif args.global_defense == 'median':
        key_list = {}
        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if net_id == 0:
                for key in net_para:
                    key_list[key] = [net_para[key].unsqueeze(0)]
            else:
                for key in net_para:
                    key_list[key].append(net_para[key].unsqueeze(0))
        for key in net_para:
            key_value_cat = torch.cat(key_list[key])
            key_value_median, _ = torch.median(key_value_cat, dim=0)
            global_w[key] = key_value_median
            
    elif args.global_defense == 'krum':
        model_weight_list = []
        net_id_list = []
        for net_id, net in enumerate(nets_this_round.values()):
            net_id_list.append(net_id)
            net_para = net.state_dict()
            model_weight = get_weight(net_para).unsqueeze(0)
            model_weight_list.append(model_weight)
        model_weight_cat = torch.cat(model_weight_list, dim=0)
        model_weight_krum, aggregate_idx = get_krum(model_weight_cat)
        model_weight_krum = model_weight_krum.reshape(-1)
        
        aggregate_idx_list = torch.tensor(party_list_this_round)[aggregate_idx].tolist()
        aggregate_idx_list.sort()
        removed_idx = list(set(party_list_this_round) - set(aggregate_idx_list))
        logger.info(">> Removed Network IDX: {}".format(' '.join(map(str, removed_idx))))

        current_idx = 0
        for key in net_para:
            length = len(net_para[key].reshape(-1))
            global_w[key] = model_weight_krum[current_idx:current_idx+length].reshape(net_para[key].shape)
            current_idx +=length 
            
    elif args.global_defense == 'norm':
        model_weight_list = []
        net_id_list = []
        for net_id, net in enumerate(nets_this_round.values()):
            net_id_list.append(net_id)
            net_para = net.state_dict()
            model_weight = get_weight(net_para).unsqueeze(0)
            model_weight_list.append(model_weight)
        model_weight_cat = torch.cat(model_weight_list, dim=0)
        model_weight_norm, aggregate_idx = get_norm(model_weight_cat)
        
        aggregate_idx_list = torch.tensor(party_list_this_round)[aggregate_idx].tolist()
        aggregate_idx_list.sort()
        removed_idx = list(set(party_list_this_round) - set(aggregate_idx_list))
        logger.info(">> Removed Network IDX: {}".format(' '.join(map(str, removed_idx))))

        current_idx = 0
        for key in net_para:
            length = len(net_para[key].reshape(-1))
            global_w[key] = model_weight_norm[current_idx:current_idx+length].reshape(net_para[key].shape)
            current_idx +=length 
            
    elif args.global_defense == 'residual':
        model_weight_list = []
        net_id_list = []
        global_w, reweight = IRLS_median_split_restricted(nets_this_round, 2.0, 0.05)
        logger.info(">> Network Weight: {}".format(' '.join(map(str, reweight.tolist()))))
        
    return global_w


args = get_args()
print(args.global_defense)
mkdirs(args.logdir)
mkdirs(args.modeldir)

device = torch.device(args.device)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

if args.log_file_name is None:
    args.log_file_name = f'log-{args.alg}-{args.dataset}-{args.model}-num-party_{args.n_parties}-beta_{args.beta}-local-epoch_{args.epochs}-type_{args.attacker_type}-attacker-ratio_{args.attacker_ratio}-noise_{args.noise_ratio}_{args.global_defense}'
log_path = args.log_file_name + '.log'
logging.basicConfig(
    filename=os.path.join(args.logdir, log_path),
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.info(device)

seed = args.init_seed
logger.info("#" * 100)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)

logger.info("Partitioning data")
X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

n_party_per_round = min(args.n_parties, args.n_parties_per_round)
party_list = [i for i in range(args.n_parties)]
party_list_rounds = []
n_attacker = int(args.attacker_ratio * args.n_parties)

if n_party_per_round != args.n_parties:
    for i in range(args.comm_round):
        this_round = random.sample(party_list, n_party_per_round)
        this_round.sort()
        party_list_rounds.append(this_round)
else:
    for i in range(args.comm_round):
        party_list_rounds.append(party_list)
        
attacker_id_list = random.sample(party_list, n_attacker)
    
logger.info(">> Attacker Network IDX: {}".format(' '.join(map(str, attacker_id_list))))

    
n_classes = len(np.unique(y_train))

train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                           args.datadir,
                                                                           args.batch_size,
                                                                           32)

print("len train_dl_global:", len(train_ds_global))
train_dl=None
data_size = len(test_ds_global)

logger.info("Initializing nets")
nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cpu')

global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cpu')
global_model = global_models[0]
n_comm_rounds = args.comm_round
    
perturb_prob_dict = {}
if args.attacker_type == 'untargeted_diverse':
    for attacker_id in attacker_id_list:
        perturb_prob_dict[attacker_id] = np.random.dirichlet(np.repeat(0.25, n_classes))
else:
    perturb_prob_dict = None
    
for round in range(n_comm_rounds):
    logger.info("in comm round:" + str(round))
    party_list_this_round = party_list_rounds[round]

    global_w = global_model.state_dict()
    nets_this_round = {k: nets[k] for k in party_list_this_round}
    for net in nets_this_round.values():
        net.load_state_dict(global_w)

    local_train_net(nets_this_round, args, net_dataidx_map, attacker_id_list=attacker_id_list, 
                    train_dl=train_dl, test_dl=test_dl, global_model= global_model, round=round, device=device)
    
    total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
    fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

    global_w = global_aggregation(nets_this_round, args, fed_avg_freqs, global_w, party_list_this_round)
    global_model.load_state_dict(global_w)
    global_model.cuda()
    
    total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
    fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

    global_w = global_aggregation(nets_this_round, args, fed_avg_freqs, global_w, party_list_this_round)
    global_model.load_state_dict(global_w)
    global_model.cuda()
    
    test_acc, _ = compute_accuracy(global_model, test_dl, device=device)

    logger.info('>> Global Model Test accuracy: %f' % test_acc)
    mkdirs(args.modeldir+'fedavg/')
    global_model.to('cpu')

torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')
