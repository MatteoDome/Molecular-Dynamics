import numpy
import pickle
import simulate
import post
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#   Values we want to cover in the batch simulations
rho_values = [0.3, 0.45, 0.6, 0.8, 1.0, 1.2]
temp_values = [i for i in numpy.arange(1, 3.5, 0.5)]

#   Parameters of the simulation that don't vary
N= 864
n_iter = 1000
n_iter_init = 250
dt = 0.004
do_plots = 0
size_blocks = 250
num_blocks = int((n_iter-n_iter_init)/size_blocks)

#   Physical quantities we want to plot
compressibility = dict.fromkeys(rho_values)
for rho in compressibility:
    compressibility[rho] = {'T': [], 'y': [], 'yerr': []}

Cv = dict.fromkeys(rho_values)
for rho in Cv:
    Cv[rho] = {'T': [], 'y': [], 'yerr': []}

E = dict.fromkeys(rho_values)
for rho in E:
    E[rho] = {'T': [], 'y': [], 'yerr': []}

#   Apply restrictions in parameters
assert N in [4*m**3 for m in range(1, 10)], 'N needs to be 4*M**3'
assert (n_iter-n_iter_init)%size_blocks == 0 and n_iter-n_iter_init>0, 'Please make n_iter-n_iter_init a multiple of n_iter_init'

for rho in rho_values:
    L = (N/rho)**(1/3)    
    for T in temp_values:
        print("Running simulation for rho=" + str(rho) + "and T=" + str(T))
        simulate.simulate(N, rho, L, T, n_iter, n_iter_init, dt, do_plots, size_blocks)
        results = post.calculate_results(N, rho, L, T, n_iter, n_iter_init, dt, num_blocks)

        #   Dump the data
        pickle.dump(results, open('batch_results/' + str(rho).replace('.', '-') + '_' + str(T).replace('.', '-'), 'wb'))
