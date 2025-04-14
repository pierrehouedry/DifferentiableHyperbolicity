# some basic local tests
# %%
from hyperbolicity.delta import compute_hyperbolicity
# %%
import pickle
# %%
path = '../expes/results_expes/expe_test/'
to_read = 'launch_distance_hyperbolicity_learning_2025_04_14_16_01_33-dataset-celegan-learning_rate-0.5-distance_reg-1.0-scale_sp-10000.0-scale_delta-10000.0-scale_softmax-10000.0-epochs-10-batch_size-32'
# %%
file = open(path+to_read+'/res.pickle', 'rb')
object_file = pickle.load(file)
# %%
object_file
# %%
