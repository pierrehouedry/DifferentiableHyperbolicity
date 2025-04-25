# %% Script writer
# to use it python script_writer.py
import itertools

name_expe = 'cora'  # name for the experience
name = 'expe_{0}'.format(name_expe)
f = open('./launch_files/' + name + '.sh', "w")
f.write('#!/bin/bash'+'\n')
f.write('export OMP_NUM_THREADS=2'+'\n')

results_path = '../results_expes/' + name + '/'
data_path = "../../datasets/"


f.write('mkdir -p {0}'.format(results_path)+'\n')

datasets = ['cora']

learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
distance_regs = [1e-4, 1e-3, 1e-2, 1e-1, 0.1, 1.0, 10]
scale_delta = [1e-3, 1e-2, 1e-1, 1.0, 10, 100]
epochs = [500]
batch_size = [(8, 1000), (8, 10000), (16, 500), (16, 1000), (32, 50), (32, 100)]
run_numbers = [0]  # for variance
nb_group = 10  # number of jobs in // on the same machine
gpu = True
parallel = False


""" learning_rates = [1e-1]
distance_regs = [10]
scale_delta = [100]
epochs = [2]
batch_size = [(32, 100)]
run_numbers = [0]  # for variance
nb_group = 10  # number of jobs in // on the same machine
gpu = False
parallel = False """

if parallel and gpu:
    raise Exception('You cannot use GPU in parallel (it is too dangerous)')

counter = 0
overallcounter = 0
for ds, lr, reg, ssd, ep, bs, rn in itertools.product(datasets, learning_rates, distance_regs, scale_delta, epochs, batch_size, run_numbers):
    if parallel:
        f.write('python ../launch_distance_hyperbolicity_learning.py  -r {0} -dp {1} -ds {2} -lr {3} -reg {4} -ssd {5}  -ep {6} -bs {7} -nb {8} -rn {9} -gpu {10} &'.format(
            results_path, data_path, ds, lr, reg, ssd, ep, bs[0], bs[1], rn, gpu)+'\n')
    else:
        f.write('python ../launch_distance_hyperbolicity_learning.py  -r {0} -dp {1} -ds {2} -lr {3} -reg {4} -ssd {5}  -ep {6} -bs {7} -nb {8} -rn {9} -gpu {10}'.format(
            results_path, data_path, ds, lr, reg, ssd, ep, bs[0], bs[1], rn, gpu)+'\n')
    counter += 1
    overallcounter += 1
    if counter == nb_group:
        if parallel:
            f.write('wait'+'\n')
        counter = 0
print('NB jobs = {}'.format(overallcounter))
# %%
