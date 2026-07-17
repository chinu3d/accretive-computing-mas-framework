import pytest
import accretive_mas
import time
import random

def test_concurrency_stress():
    substrate = accretive_mas.ContinuousSubstrate()
    
    # Inject 100 agents with random parameters
    num_agents = 100
    for i in range(num_agents):
        pos = [random.uniform(-5.0, 5.0) for _ in range(3)]
        amplitude = random.uniform(0.5, 2.0)
        frequency = random.uniform(1.0, 5.0)
        phase = random.uniform(0.0, 6.28)
        
        agent = accretive_mas.MicroAgent(pos, amplitude, frequency, phase)
        substrate.inject_agent(agent)
    
    # Time the free energy calculation (which runs the O(N^2) parallel overlap integrals)
    start_time = time.time()
    energy = substrate.calculate_free_energy()
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\n[Stress Test] Calculated free energy for {num_agents} agents in {duration:.4f} seconds.")
    print(f"[Stress Test] Total Free Energy: {energy}")
    
    # Ensure it computed something valid and didn't crash
    assert isinstance(energy, float)
    
    # Test step_simulation
    start_time_step = time.time()
    substrate.step_simulation(0.01)
    end_time_step = time.time()
    print(f"[Stress Test] Simulation step completed in {end_time_step - start_time_step:.4f} seconds.")
