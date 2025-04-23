import socket
from time import time
import argparse
import torch
import torch.optim as optim
import networkx as nx
from tqdm import tqdm
from hyperbolicity.delta import compute_hyperbolicity_batch
from hyperbolicity.utils import soft_max, floyd_warshall, soft_max, construct_weighted_matrix, make_batches, create_log_dir
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class ParamError(Exception):
    pass


class NanError(Exception):
    pass


def load_data(dataset_name, base_path):
    # Load the dataset

    if dataset_name == 'celegan':
        c_elegan = 'D_celegan.pkl'
        c_elegan_path = os.path.join(base_path, c_elegan)
        with open(c_elegan_path, 'rb') as f:
            distances = pickle.load(f)

    elif dataset_name == 'phd':
        cs_phd = 'D_csphd.pkl'
        cs_phd_path = os.path.join(base_path, cs_phd)
        with open(cs_phd_path, 'rb') as f:
            distances = pickle.load(f)

    elif dataset_name == 'airport':
        airport = 'airport/airport.p'
        airport_path = os.path.join(base_path, airport)
        with open(airport_path, 'rb') as f:
            aiport_graph = pickle.load(f)
            distances = nx.floyd_warshall_numpy(aiport_graph)

    elif dataset_name == 'cora':
        cora_path = os.path.join(base_path, 'cora')
        cora_dataset = Planetoid(root=cora_path, name='Cora')
        cora_graph = to_networkx(cora_dataset[0], to_undirected=True)
        distances = nx.floyd_warshall_numpy(cora_graph)

    return torch.tensor(distances).to('cuda')


def train_distance_matrix(distances: torch.Tensor,
                          scale_delta: float,
                          scale_soft_max: float,
                          distance_reg: float,
                          num_epochs: int,
                          n_batches: int,
                          batch_size: int,
                          learning_rate: float,
                          verbose: bool):

    num_nodes = distances.shape[0]
    edges = torch.triu_indices(num_nodes, num_nodes, offset=1)
    upper_adjency = torch.triu(distances, diagonal=1).type(torch.float32)
    true_upper = upper_adjency[upper_adjency != 0].to('cuda').detach().clone()
    weights_opt = true_upper.clone().requires_grad_(True)
    
    
    optimizer = optim.AdamW([weights_opt], lr=learning_rate, weight_decay=0)
    losses = []
    deltas = []
    errors = []


    def projection(weight, num_nodes, edges):
        update_dist = construct_weighted_matrix(weight, num_nodes, edges)
        update_dist = floyd_warshall(update_dist)

        return torch.triu(update_dist, diagonal=1)[torch.triu(update_dist, diagonal=1) != 0]

    def loss_fn(w):
        update_dist = construct_weighted_matrix(w, num_nodes, edges)
        M_batch = make_batches(update_dist, size_batches=batch_size, nb_batches=n_batches)
        delta = soft_max(compute_hyperbolicity_batch(M_batch, scale=scale_delta), scale=scale_soft_max)
        err = (distances-update_dist).pow(2).mean()

        return delta + distance_reg*err, delta, err

    with tqdm(range(num_epochs), desc="Training Weights", disable=not verbose) as pbar:
        for epoch in pbar:
            optimizer.zero_grad()
            loss, delta, err = loss_fn(weights_opt)

            pbar.set_description(f"loss = {loss.item():.5f}, delta = {delta:.5f}, error = {err:.5f}")
            if torch.isnan(loss):
                raise NanError('Loss is Nan')
            losses.append(loss.item())
            deltas.append(delta.item())
            errors.append(err.item())
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                weights_opt.data = projection(weights_opt, num_nodes, edges)

    return weights_opt.detach().clone(), losses, deltas, errors


if __name__ == '__main__':

    print("Machine = {}".format(socket.gethostname()))

    # Main parameters
    parser = argparse.ArgumentParser(
        description='Tree fitting experiment')
    parser.add_argument('-r', '--log_dir', nargs='?',
                        type=str, help='the path to the directory where to write to', required=True)
    parser.add_argument('-dp', '--data_path', nargs='?',
                        type=str, help='The path to the data', required=True)
    parser.add_argument('-v', '--verbose', nargs='?',
                        help='to verbose or not to verbose', type=str2bool, default=True)
    parser.add_argument('-rn', '--run_number', nargs='?', type=int,
                        help='The number of the run (for variance)', default=0)
    parser.add_argument('-ds', '--dataset', nargs='?', type=str, help='dataset to choose',
                        choices=['cora', 'phd', 'airport', 'celegan'], required=True)
    parser.add_argument('-lr', '--learning_rate', nargs='?', type=float,
                        help='Learning rates', default=1.0)
    parser.add_argument('-reg', '--distance_reg', nargs='?', type=float,
                        help='Distance regularization', default=1.0)
    parser.add_argument('-ssp', '--scale_sp', nargs='?', type=float,
                        help='Scale shortest path', default=10000.0)
    parser.add_argument('-ssd', '--scale_delta', nargs='?', type=float,
                        help='Scale delta', default=10000.0)
    parser.add_argument('-sssm', '--scale_softmax', nargs='?', type=float,
                        help='Scale soft max', default=10000.0)
    parser.add_argument('-ep', '--epochs', nargs='?', type=int,
                        help='Number of epochs', default=500)
    parser.add_argument('-bs', '--batch_size', nargs='?', type=int,
                        help='Batch size', default=32)

    args = parser.parse_args()

    try:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
    except OSError:
        raise

    name_ = ''
    for arg in vars(args):
        if arg not in ["log_dir", "verbose", "data_path"]:
            name_ += '-{}-{}'.format(arg, str(getattr(args, arg)))

    log_dir = create_log_dir(args, add_name=name_)
    logger = setup_logger('outer_logger', log_dir + '/what_happened_to_us.log')
    logger.info('############  Let us GO   ############')
    logger.info('############ Machine = {} ############ \n'.format(socket.gethostname()))

    results = {}  # dictionnary of results
    results['dataset'] = args.dataset
    results['learning_rate'] = args.learning_rate
    results['distance_reg'] = args.distance_reg
    results['run_number'] = args.run_number
    results['scale_sp'] = args.scale_sp
    results['scale_delta'] = args.scale_delta
    results['scale_softmax'] = args.scale_softmax
    results['epochs'] = args.epochs
    results['batch_size'] = args.batch_size

    # Load data
    logger.info('Doing dataset {}'.format(args.dataset))
    distances = load_data(args.dataset, args.data_path)

    try:
        weights, losses,  deltas, errors = train_distance_matrix(distances,
                                                                 scale_sp=args.scale_sp,
                                                                 scale_delta=args.scale_delta,
                                                                 scale_soft_max=args.scale_softmax,
                                                                 distance_reg=args.distance_reg,
                                                                 num_epochs=args.epochs,
                                                                 batch_size=args.batch_size,
                                                                 learning_rate=args.learning_rate,
                                                                 verbose=args.verbose)
        results['weights'] = weights
        results['loss'] = losses
        results['deltas'] = deltas
        results['errors'] = errors

    except NanError:
        logger.info('!!! Loss is Nan !!!')
        results['weights'] = torch.nan
        results['loss'] = torch.nan
        results['deltas'] = torch.nan
        results['errors'] = torch.nan

    with open(log_dir + '/res.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info('The result dict is saved.')
    logger.info('############ It is over DUDE ############')
