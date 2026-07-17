import pytest
import accretive_mas

def test_micro_agent_creation():
    # Initial position, amplitude, frequency, phase
    agent = accretive_mas.MicroAgent([1.0, 2.0, 3.0], 5.0, 10.0, 0.5)
    
    pos = agent.get_position()
    assert pos.shape == (3, 1)
    assert pos[0] == 1.0
    assert pos[1] == 2.0
    assert pos[2] == 3.0
    
    assert agent.get_amplitude() == 5.0
    assert agent.get_frequency() == 10.0
    assert agent.get_phase() == 0.5

def test_macro_agent_binding():
    macro = accretive_mas.MacroAgent()
    micro1 = accretive_mas.MicroAgent([0.0, 0.0, 0.0], 1.0, 1.0, 0.0)
    micro2 = accretive_mas.MicroAgent([1.0, 1.0, 1.0], 1.0, 1.0, 0.0)
    
    macro.cable_bind(micro1, micro2)
    # traverse_topology doesn't return anything but we ensure it doesn't crash
    macro.traverse_topology()
    
    # We can also evaluate the wave function at a point
    wave = macro.get_wave_function_at(0.5, 0.5, 0.5, 0.0)
    assert isinstance(wave, complex)

def test_continuous_substrate():
    substrate = accretive_mas.ContinuousSubstrate()
    micro1 = accretive_mas.MicroAgent([0.0, 0.0, 0.0], 1.0, 1.0, 0.0)
    micro2 = accretive_mas.MicroAgent([2.0, 2.0, 2.0], 1.0, 2.0, 3.14)
    
    substrate.inject_agent(micro1)
    substrate.inject_agent(micro2)
    
    # Calculate free energy
    energy = substrate.calculate_free_energy()
    assert isinstance(energy, float)
    
    # Step simulation (tests GIL release and async thread execution)
    substrate.step_simulation(0.01)
