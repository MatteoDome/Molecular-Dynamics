# Usage
## Normal mode
This mode allows you to run **one** simulation with specific parameters.
### Setting parameters for simulation
The parameters for the simulation can be set in `params.txt`. They are:

1. `n_iter`: Number of particles
2. `rho`: Density
3. `T`: Temperature
4. `n_iter`: Number of iterations
5. `n_iter_init`: Number of iterations until equilibrium. The number of iterations before the thermostat is used.
6. `dt`: Time step
7.  `do_plots`: Boolean. If set to 1, the correlation function and the diffusion will be outputted to `correlation_function.pdf` and `diffusion.pdf`.
8. `size_blocks`: The size (in iterations) of each block used for data blocking.  **NOTE:** `n_iter - n_iter_init` must be aa multiple of `size_blocks`.

### Running the simulation
1. Set parameters in `params.txt`. 
2. Run `python simulate.py`. The results will be dumped in `sim_data/`.
3. Run `python post.py`.
4. The results should show up in the terminal windows. 

## Batch Mode
This mode allows you to run several simulations in a row in order to plot the energy, heat capacity and pressure as a function of temperature and density. 
### Running the simulation
1. Set parameters and values of density/temperature in `batch.py`.
2. Run `python batch.py`.
3. Results and plots will be dumped in `batch_results/`