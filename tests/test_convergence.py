import pytest
import accretive_mas
import numpy as np
import time

def test_monte_carlo_convergence():
    # Test how the differential mismatch integral converges as we increase MC resolution
    # We will inject two identical agents at the same position.
    # Because A == B, the integral \iiint \Psi_A \Psi_B^* dV dt evaluates exactly to the norm.
    # The true value of delta(A, B) is essentially 1.0 - (Volume_Integral / T).
    
    substrate = accretive_mas.ContinuousSubstrate()
    
    agent1 = accretive_mas.MicroAgent([0.0, 0.0, 0.0], 1.0, 1.0, 0.0)
    agent2 = accretive_mas.MicroAgent([0.0, 0.0, 0.0], 1.0, 1.0, 0.0)
    
    substrate.inject_agent(agent1)
    substrate.inject_agent(agent2)
    
    resolutions = [100, 1000, 10000, 50000]
    
    print("\n--- Monte Carlo Convergence Test ---")
    previous_variance = None
    
    for res in resolutions:
        substrate.set_mc_resolution(res)
        assert substrate.get_mc_resolution() == res
        
        # We will run the calculation 10 times at each resolution to measure variance (jitter)
        results = []
        start_t = time.time()
        for _ in range(10):
            # calculate_free_energy computes delta(a,b) - dissipation_delta
            energy = substrate.calculate_free_energy()
            results.append(energy)
        end_t = time.time()
        
        std_dev = np.std(results)
        mean_val = np.mean(results)
        
        print(f"Resolution: {res:6d} | Mean Energy: {mean_val:8.4f} | StdDev (Jitter): {std_dev:8.6f} | Time/10 runs: {end_t - start_t:.4f}s")
        
        # Verify that as resolution increases, the standard deviation decreases
        if previous_variance is not None:
            assert std_dev < previous_variance, f"Variance did not decrease! Went from {previous_variance} to {std_dev}"
            
        previous_variance = std_dev
        
    # The final resolution should have a significantly reduced jitter compared to baseline
    assert std_dev < 0.5, f"Monte Carlo integration did not converge sufficiently, StdDev = {std_dev}"
