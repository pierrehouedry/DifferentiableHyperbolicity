# %%
import os

# dataset_n_nodes = {'celegan': 452}
parallel = False
dataset = 'phd'
result_path = './results_expes/expe_phd/'  # do not forget / in the end

name = 'read_expe_{0}'.format(dataset)
f = open('./launch_files/' + name + '.sh', "w")
f.write('#!/bin/bash'+'\n')
f.write('export OMP_NUM_THREADS=2'+'\n')

# Dirty
modified_path = '../' + result_path.lstrip('./')

counter = 0
overallcounter = 0
nb_group = 10  # number of jobs in // on the same machine
for root, dirs, files in os.walk(result_path):
    for expe_name in dirs:
        if expe_name.startswith("launch_distance_hyperbolicity_learning"):
            folder_path = os.path.join(root, expe_name)
            print(expe_name)
            if parallel:
                f.write(
                    'python ../read_results.py  -r {0} -re {1} -ds {2} &'.format(modified_path, expe_name, dataset)+'\n')
            else:
                f.write(
                    'python ../read_results.py  -r {0} -re {1} -ds {2}'.format(modified_path, expe_name, dataset)+'\n')
            counter += 1
            overallcounter += 1
            if counter == nb_group:
                if parallel:
                    f.write('wait'+'\n')
                counter = 0
print('NB jobs = {}'.format(overallcounter))

# %%
