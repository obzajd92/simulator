import numpy as np

def simulate_processes(n_iterations=100000, set_length=3):
    x = np.linspace(0, 10, set_length)
    
    # Pre-allocate arrays for tracking over iterations to see convergence/behavior
    # Since n is large, we can sample at intervals or track the running mean to show convergence
    # Let's track the running average of the processes to show convergence to the mean functions.
    
    # We will generate the full simulation efficiently
    # y1_mean = 2*x + 5
    # y2_mean = -1.5*x + 8
    
    # Let's save a few iterations for the dot plots (e.g., first 5 iterations)
    samples_y1 = []
    samples_y2 = []
    
    running_sum_y1 = np.zeros(set_length)
    running_sum_y2 = np.zeros(set_length)
    
    # To show convergence, let's look at a specific point or the norm of the error over iterations
    iterations_to_plot = [1, 10, 100, 1000, 10000, 100000]
    running_means_y1 = {}
    running_means_y2 = {}
    
    for i in range(1, n_iterations + 1):
        n1 = np.random.normal(0, 1, set_length)
        n2 = np.random.normal(0, 1, set_length)
        
        y1 = 2 * x + 5 + n1
        y2 = -1.5 * x + 8 + n2
        
        running_sum_y1 += y1
        running_sum_y2 += y2
        
        if i <= 5:
            samples_y1.append(y1.copy())
            samples_y2.append(y2.copy())
            
        if i in iterations_to_plot:
            running_means_y1[i] = running_sum_y1 / i
            running_means_y2[i] = running_sum_y2 / i

    return x, samples_y1, samples_y2, running_means_y1, running_means_y2

x, samples_y1, samples_y2, rm_y1, rm_y2 = simulate_processes()
print("Simulation complete.")
print("Running mean y1 at 100k:", rm_y1[100000])
print("True mean y1:", 2*x + 5)