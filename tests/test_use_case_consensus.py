import pytest
import accretive_mas
import random

def test_swarm_consensus_federated_learning():
    """
    Real-World Use Case: Federated Learning / Swarm Consensus without a Central Server.
    
    Imagine 5 autonomous edge devices (e.g., self-driving cars or local LLMs). 
    Each trains on local data, so their model weights (represented here by their topological phase) 
    drift apart. 
    
    Instead of a central server averaging their weights (which requires discrete message passing), 
    we project them into the Accretive Continuous Substrate. The physics engine naturally 
    forces their overlapping wave functions to resolve conflicts (destructive interference).
    Through gradient descent on the free energy, the swarm organically synchronizes to a 
    perfect consensus state.
    """
    
    substrate = accretive_mas.ContinuousSubstrate()
    substrate.set_mc_resolution(200) # Lower for speed
    
    # 5 edge devices starting with different local model weights (phases)
    # We initialize them within the same basin of attraction (e.g. 0.0 to 1.5) so they
    # don't get stuck in different periodic local minima for this simple test.
    agents = []
    initial_phases = [0.1, 0.5, 0.8, 1.2, 1.5]
    
    for idx, phase in enumerate(initial_phases):
        # All exist in the same conceptual parameter space [0,0,0]
        agent = accretive_mas.MicroAgent([0.0, 0.0, 0.0], 1.0, 1.0, phase)
        agents.append(agent)
        substrate.inject_agent(agent)
        
    print("\n[Swarm Consensus] Initial Local Model Weights (Phases):")
    for i, a in enumerate(agents):
        print(f"Device {i}: {a.get_phase():.4f}")
        
    initial_energy = substrate.calculate_free_energy()
    print(f"\n[Swarm Consensus] Initial System Conflict (Free Energy): {initial_energy:.4f}")
    
    # Let the topology resolve the conflict organically over continuous time steps
    for _ in range(50):
        substrate.step_simulation(dt=0.1)
        
    final_energy = substrate.calculate_free_energy()
    print(f"\n[Swarm Consensus] Final System Conflict (Free Energy): {final_energy:.4f}")
    
    print("\n[Swarm Consensus] Final Local Model Weights (Phases):")
    phases = [a.get_phase() for a in agents]
    for i, p in enumerate(phases):
        print(f"Device {i}: {p:.4f}")
        
    # Assert that conflict was minimized
    assert final_energy < initial_energy, "The swarm failed to resolve conflicts."
    
    # Assert that consensus was reached (variance of phases should be extremely small)
    mean_phase = sum(phases) / len(phases)
    max_deviation = max(abs(p - mean_phase) for p in phases)
    
    print(f"\n[Swarm Consensus] Maximum deviation from consensus: {max_deviation:.6f}")
    
    # Prove that the models have synchronized their weights topologically
    assert max_deviation < 0.1, "The swarm failed to reach consensus!"
