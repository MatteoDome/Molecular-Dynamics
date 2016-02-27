import numpy
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def calculate_results(N, rho, L, T, n_iter, n_iter_init, dt, num_blocks):   
    #   Compute energy and error
    E = numpy.loadtxt(open('sim_data/E.dat', 'r'))
    E_split = numpy.array_split(E, num_blocks)
    E_split_av = numpy.array([sample.mean() for sample in E_split])/N
    E_av = E_split_av.mean()
    E_av_error = E_split_av.std()

    #   Compute kinetic energy and error
    K = numpy.loadtxt(open('sim_data/K.dat', 'r'))
    K_split = numpy.array_split(K, num_blocks)
    K_split_av = numpy.array([sample.mean() for sample in K_split])/N
    K_av = K_split_av.mean()
    K_av_error = K_split_av.std()

    #   Compute potential energy
    V = numpy.loadtxt(open('sim_data/V.dat', 'r'))
    V_split = numpy.array_split(V, num_blocks)
    V_split_av = numpy.array([sample.mean() for sample in V_split])/N
    V_av = V_split_av.mean()
    V_av_error = V_split_av.std()
    
    #   Compute heat capacity
    Cv_split = numpy.array([1.5/(1-1.5*N*sample.var()/sample.mean()**2) for sample in K_split])
    Cv_av = Cv_split.mean()
    Cv_av_error = Cv_split.std()

    #   Compute temperature
    T_split = numpy.array_split(K, num_blocks)
    T_split_av = numpy.array([sample.mean()*2/(3*N) for sample in T_split])
    T_av = T_split_av.mean()
    T_av_error = T_split_av.std()

    #   Compute compressibility
    virial = numpy.loadtxt(open('sim_data/virial.dat', 'r'))
    virial_split = numpy.array_split(virial, num_blocks)
    compressibility_split_av = numpy.array([1-sample.mean()/(3*N*T_split_av[index]) for index, sample in enumerate(virial_split)])
    compressibility_av = compressibility_split_av.mean()
    compressibility_av_error = compressibility_split_av.std()
    
    results = {
        'parameters': {'rho': rho, 'T': T},
        'E': {'value': E_av, 'error': E_av_error},
        'K': {'value': K_av, 'error': K_av_error},
        'V': {'value': V_av, 'error': V_av_error},
        'Cv': {'value': Cv_av, 'error': Cv_av_error},
        'T': {'value': T_av, 'error': T_av_error},
        'compressibility': {'value': compressibility_av, 'error': compressibility_av_error}
    }

    return results

def plot_correlation_function(N, L, n_iter, n_iter_init, separations_file):
    separations = numpy.load(separations_file)
    
    # Remove 0's (diagonal elements) and apply square root
    separations = numpy.sqrt(numpy.ravel(separations[numpy.nonzero(separations)]))
    bin_range = 4
    bin_number = 200
    bin_size = bin_range/bin_number

    bins = numpy.arange(0,bin_range,bin_range/bin_number)
    counts, edges = numpy.histogram(separations, bins=bins)

    xvalues = numpy.array(edges[1::])
    yvalues = counts*(size_blocks/(n_iter-n_iter_init))

    for j in range(0,bin_number-1):
        yvalues[j] *= L*L*L/(N*(N-1))/(4*math.pi*(xvalues[j])**2*bin_size)

    with PdfPages('correlation.pdf') as pdf:
        plt.figure()
        plt.plot(xvalues,yvalues)
        plt.xlabel("$\\frac{r}{\\sigma}$")
        plt.ylabel("$g$")
        pdf.savefig() 
        plt.close()

def plot_displacement_sqr(N, n_iter, n_iter_init, dt, dx_file):
    #   We need to have enough iterations in order to fit properly
    assert n_iter-n_iter_init >= 400, 'You need at least 400 iterations between n_iter and n_iter_init'

    dx = numpy.load('sim_data/dx.npy')
    
    #   We only plot from n_iter_init because until then the speeds are being renormalized
    av_x2 = numpy.zeros(shape=(n_iter-n_iter_init,1))

    for i in range(n_iter_init, n_iter): 
        x = numpy.sum(dx[n_iter_init:i],axis=0)
        x2 = numpy.mean(x*x,axis=1)
        av_x2[i-n_iter_init] = numpy.sum(numpy.sum(x2))

    time = numpy.arange(n_iter-n_iter_init)*dt

    #   We only fit after the ballistic part (250 iterations approximately)
    fit, cov = numpy.polyfit(time[250:], av_x2[250:], 1, cov=True)

    with PdfPages('diffusion.pdf') as pdf:
        plt.figure()
        plt.plot(time, av_x2, label='Data')
        plt.plot(time, fit[0]*time + fit[1], label="Fit")
        plt.xlabel("$t$")
        plt.ylabel("$x^2$")
        plt.legend(loc=4)
        
        pdf.savefig() 
        plt.close()

    #   Calculate the erro through the covariance matrix
    diff_coeff = fit[0][0]
    diff_coeff_err = cov[0][0][0]

    return diff_coeff, diff_coeff_err

if __name__ == "__main__": 
    with open('params.txt', 'r') as params_file:
        #   Read simulation parameters
        params = params_file.readlines()
        N = int(params[0].split()[0])
        rho = float(params[1].split()[0])
        T = float(params[2].split()[0])
        L = (N/rho)**(1/3)
        n_iter = int(params[3].split()[0])
        n_iter_init = int(params[4].split()[0])
        dt = float(params[5].split()[0])
        do_plots = int(params[6].split()[0])
        size_blocks = int(params[7].split()[0])
        num_blocks = int((n_iter-n_iter_init)/size_blocks)
        
        results = calculate_results(N, rho, L, T, n_iter, n_iter_init, dt, num_blocks)

        print("------")
        print("ENERGY/N:")
        print(str(results['E']['value']) + ' +- ' + str(results['E']['error']) )

        print("------")
        print("KINETIC ENERGY/N:")
        print(str(results['K']['value']) + ' +- ' + str(results['K']['error']) )

        print("------")
        print("POTENTIAL ENERGY/N:")
        print(str(results['V']['value']) + ' +- ' + str(results['V']['error']) )

        print("------")
        print("HEAT CAPACITY/N:")
        print(str(results['Cv']['value']) + ' +- ' + str(results['Cv']['error']) )

        print("------")
        print("TEMPERATURE:")
        print(str(results['T']['value']) + ' +- ' + str(results['T']['error']) )

        print("------")
        print("COMPRESSIBILITY:")
        print(str(results['compressibility']['value']) + ' +- ' + str(results['compressibility']['error']) )
        
        if do_plots:           
            with open('sim_data/separations.npy', 'rb') as separations_file: 
                plot_correlation_function(N, L, n_iter, n_iter_init, separations_file)

            with open('sim_data/dx.npy') as dx_file:
                diff_coeff, diff_coeff_err = plot_displacement_sqr(N, n_iter, n_iter_init, dt, dx_file)
                print("------")
                print("DIFFUSION COEFFICIENT")
                print(str(diff_coeff) + ' +- ' + str(diff_coeff_err) )