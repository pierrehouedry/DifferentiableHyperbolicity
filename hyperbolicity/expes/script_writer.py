# %% Script writer
# to use it python script_writer.py
import itertools

name_expe = 'test_expe'  # name for the experience
name = 'expe_{0}'.format(name_expe)
f = open('./launch_files/' + name + '.sh', "w")
f.write('#!/bin/bash'+'\n')
f.write('export OMP_NUM_THREADS=2'+'\n')

results_path = '../results_expes/' + name + '/'
data_path = "../../datasets/"


f.write('mkdir -p {0}'.format(results_path)+'\n')

datasets = ['celegan']
# learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]
# distance_regs = [1e-4, 1e-3, 1e-2, 1e-1, 0.1, 1.0, 10, 100]
# scale_delta = [1e-3, 1e-2, 1e-1, 1.0, 10, 100]
# scale_softmax = [1e-3, 1e-2, 1e-1, 1.0, 10, 100]
# epochs = [500]
# batch_size = [32, 64, 128]
# n_batches = [50, 100]
# run_numbers = [0]  # for variance
# nb_group = 50  # number of jobs in // on the same machine

learning_rates = [0.5]
distance_regs = [10]
scale_delta = [100]
scale_softmax = [100]
epochs = [500]
batch_size = [32]
n_batches = [5]
run_numbers = [0]  # for variance
nb_group = 1  # number of jobs in // on the same machine

counter = 0
overallcounter = 0
for ds, lr, reg, ssd, ssm, ep, bs, nb, rn in itertools.product(datasets, learning_rates, distance_regs, scale_delta, scale_softmax, epochs, batch_size, n_batches, run_numbers):
    f.write('python ../launch_distance_hyperbolicity_learning.py  -r {0} -dp {1} -ds {2} -lr {3} -reg {4} -ssd {5} -ssm {6} -ep {7} -bs {8} -nb {9} -rn {10} &'.format(
        results_path, data_path, ds, lr, reg, ssd, ssm, ep, bs, nb, rn)+'\n')
    counter += 1
    overallcounter += 1
    if counter == nb_group:
        f.write('wait'+'\n')
        counter = 0
print('NB jobs = {}'.format(overallcounter))
# %%
