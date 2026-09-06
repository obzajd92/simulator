import numpy as np
import matplotlib.pyplot as plt

def run_simulation(n_iterations=100000, set_length=3):
    # 1. Define x values (default length = 3)
    x = np.linspace(1, 5, set_length)
    
    # Target deterministic components
    y1_true = 2 * x + 5
    y2_true = -1.5 * x + 8
    
    # Structures to hold data
    first_few_y1 = []
    first_few_y2 = []
    
    running_sum_y1 = np.zeros(set_length)
    running_sum_y2 = np.zeros(set_length)
    
    # Track error norm to prove convergence over time
    iterations = []
    errors = []

    # 2. Multi-sample simulation loop
    for i in range(1, n_iterations + 1):
        # Mutual exclusive/independent standard normal noise IID
        n1 = np.random.normal(0, 1, set_length)
        n2 = np.random.normal(0, 1, set_length)
        
        y1 = 2 * x + 5 + n1
        y2 = -1.5 * x + 8 + n2
        
        running_sum_y1 += y1
        running_sum_y2 += y2
        
        # Save the first 5 iterations for visualization
        if i <= 5:
            first_few_y1.append(y1)
            first_few_y2.append(y2)
            
        # Log convergence metrics periodically to save memory
        if i == 1 or i % 100 == 0:
            current_mean_y1 = running_sum_y1 / i
            current_mean_y2 = running_sum_y2 / i
            # Mean absolute error from the true line
            mae = (np.mean(np.abs(current_mean_y1 - y1_true)) + 
                   np.mean(np.abs(current_mean_y2 - y2_true))) / 2
            iterations.append(i)
            errors.append(mae)
            
    final_mean_y1 = running_sum_y1 / n_iterations
    final_mean_y2 = running_sum_y2 / n_iterations
    
    # 3. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
#Diagrams

    # Diagram 1: Iterations as Dots (y vs x)
    for idx, (y1_sample, y2_sample) in enumerate(zip(first_few_y1, first_few_y2)):
        axes[0].scatter(x, y1_sample, color='blue', alpha=0.6, label='y1 (Samples)' if idx == 0 else "")
        axes[0].scatter(x, y2_sample, color='orange', alpha=0.6, label='y2 (Samples)' if idx == 0 else "")
    axes[0].plot(x, y1_true, 'b--', label='y1 True Mean')
    axes[0].plot(x, y2_true, 'r--', label='y2 True Mean')
    axes[0].set_title("Process Iterations (First 5 Samples)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()
    axes[0].grid(True)

    # Diagram 2: Phase Space Plot (y2 vs y1)
    for idx, (y1_sample, y2_sample) in enumerate(zip(first_few_y1, first_few_y2)):
        axes[1].scatter(y1_sample, y2_sample, label=f"Iteration {idx+1}")
    axes[1].scatter(y1_true, y2_true, color='black', marker='X', s=100, label='True Mean State')
    axes[1].set_title("Phase Space Diagram ($y_2$ vs $y_1$)")
    axes[1].set_xlabel("y1")
    axes[1].set_ylabel("y2")
    axes[1].legend()
    axes[1].grid(True)

    # Diagram 3: Convergence Checklist
    axes[2].plot(iterations, errors, color='purple', lw=2)
    axes[2].set_xscale('log')
    axes[2].set_title("Simulation Convergence (MAE vs Iterations)")
    axes[2].set_xlabel("Number of Iterations (Log Scale)")
    axes[2].set_ylabel("Mean Absolute Error")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    # Print final verification
    print(f"--- Convergence Evaluation after {n_iterations:,} Iterations ---")
    print(f"Empirical Mean y1: {final_mean_y1.round(4)}")
    print(f"Theoretical Mean y1: {y1_true.round(4)}")
    print(f"Empirical Mean y2: {final_mean_y2.round(4)}")
    print(f"Theoretical Mean y2: {y2_true.round(4)}")

# Run the simulation
run_simulation()