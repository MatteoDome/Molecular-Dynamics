import math
import time
import numpy
import random
import scipy.stats
import matplotlib.pyplot as plt
from numba import jit

def initial_positions(N, L): 
    num_cells = round((N/4)**(1/3))

    #   The matrix that will contain all the position of the particles
    positions = numpy.zeros(shape=(3,N), dtype="float64") 

    #   Primitive cel matrix
    acel = numpy.array([[0,0,0.5,0.5],[0,0.5,0,0.5],[0,0.5,0.5,0]])    

    #   Lattice vectors
    ax = numpy.array([[1,1,1,1],[0,0,0,0],[0,0,0,0]])
    ay = numpy.array([[0,0,0,0],[1,1,1,1],[0,0,0,0]])
    az = numpy.array([[0,0,0,0],[0,0,0,0],[1,1,1,1]])

    offs = 0
    for m in range(num_cells):
        for k in range(num_cells):
            for l in range(num_cells):
                positions[:,offs:offs+4] = acel + l*ax + k*ay + m*az
                offs += 4

    positions = positions*L/num_cells - L/2

    return positions

def initial_velocities(N, T):
    v = numpy.zeros([3, N])

    #   Draw from Maxwell (normal) distribution
    v[0,:] = numpy.random.normal(size=N)
    v[1,:] = numpy.random.normal(size=N)
    v[2,:] = numpy.random.normal(size=N)

    v *= math.sqrt(T)

    #   Set total momentum to zero
    v_mean = numpy.mean(v,1)
    v=v-v_mean[:, None]

    return v

@jit
def compute_forces(N, T, x, L):
    a = numpy.zeros((3,N),dtype=float)
    separations = numpy.zeros(shape=([N, N]))
    V = 0
    virial = 0

    for i in range(0, N):
        for j in range(i+1, N):
            dx = x[0,i] - x[0,j]
            dy = x[1,i] - x[1,j]
            dz = x[2,i] - x[2,j]
            dx = dx - numpy.rint(dx/L)*L
            dy = dy - numpy.rint(dy/L)*L
            dz = dz - numpy.rint(dz/L)*L
            dr2 = dx*dx+dy*dy+dz*dz
            dr2i = 1/dr2

            separations[i,j] = dr2
            separations[j,i] = dr2
           
            # If condition applies the cutoff
            if dr2 < 9.0:
                r6i = dr2i*dr2i*dr2i
 
                Fij = (2*r6i - 1.0 )*r6i*dr2i
                a[0,i] += Fij*dx
                a[1,i] += Fij*dy
                a[2,i] += Fij*dz
                a[0,j] -= Fij*dx
                a[1,j] -= Fij*dy
                a[2,j] -= Fij*dz

                V += r6i*(r6i - 1.0)
                virial = virial-48*r6i*r6i + 24*r6i

    #   Correction term for virial
    virial -= 2*math.pi*N/(3*T*L*L*L)*8*(-2/(2187) + 1/27)

    a = 24.0*a
    V = 4.0*V
    
    return a, V, virial, separations

def simulate(N, rho, L, T, n_iter, n_iter_init, dt, do_plots, size_blocks):
    #   We only store dx and the separations if we want to plot <x^2> and the correlation function
    if do_plots:
        separations_iter = numpy.zeros(shape=([int(n_iter/size_blocks), N, N]))
        dx_iter = numpy.zeros(shape=([n_iter, 3, N]))

    print("Initializing...")

    #   Files where information will be written
    E_file = open('sim_data/E.dat', 'w')
    V_file = open('sim_data/V.dat', 'w')
    K_file = open('sim_data/K.dat', 'w')
    virial_file = open('sim_data/virial.dat', 'w')

    #   Initialize system variables
    x = initial_positions(N, L)
    v = initial_velocities(N, T)
    a = numpy.zeros([3, N])

    #   Main cycle
    for i in range(0, n_iter):
        v += 0.5*a*dt
        dx = v*dt
        x += dx
        x = numpy.mod(x,L)

        #   If user wants to plot <x2> and correlation function store separations and dx
        if do_plots:
            a, V, virial, separations = compute_forces(N, T, x, L)
            
            #   Separations are saved less often, since it's quite memory intensive to store them
            if i%size_blocks==0 and i>=n_iter_init:
                separations_iter[i/size_blocks] = separations
            
            dx_iter[i] = dx
       
        else:
            a, V, virial, _ = compute_forces(N, T, x, L)
        
        v += 0.5*a*dt
        K = 0.5*numpy.sum(v*v)

        #   Compute energy with correction term
        E = V + K + 8*math.pi*(N-1)*(1/(9*19683) - 1/(3*27))

        #   If needed scale velocities
        if i < n_iter_init and i > 10 and i%10 == 0:
            v *= numpy.sqrt((N-1)*(3/2)*T/K)

        #   Write data to file
        elif i>=n_iter_init: 
            E_file.write(str(E) + '\n')
            V_file.write(str(V)+ '\n')
            K_file.write(str(K)+ '\n')
            virial_file.write(str(virial)+ '\n')

        print('Iteration ' + str(i))

    #   Close files
    E_file.close()
    V_file.close()
    K_file.close()
    virial_file.close()

    if do_plots:
        print("Saving data...")
        numpy.save('sim_data/separations', separations_iter)
        numpy.save('sim_data/dx', dx_iter)

if __name__ == "__main__": 
    with open('params.txt', 'r') as params_file:
        print("Reading parameters...")
       
        params = params_file.readlines()
        
        #   Read simulation parameters (we use split so that we can label each one in params.txt)
        N = int(params[0].split()[0])
        rho = float(params[1].split()[0])
        T = float(params[2].split()[0])
        L = (N/rho)**(1/3)
        n_iter = int(params[3].split()[0])
        n_iter_init = int(params[4].split()[0])
        dt = float(params[5].split()[0])
        do_plots = int(params[6].split()[0])
        size_blocks = int(params[7].split()[0])
            
        assert N in [4*m**3 for m in range(1, 10)], 'N needs to be 4*M**3'
        assert (n_iter-n_iter_init)%size_blocks == 0 and n_iter-n_iter_init>0, 'Please make n_iter-n_iter_init a multiple of n_iter_init'

        simulate(N, rho, L, T, n_iter, n_iter_init, dt, do_plots, size_blocks)