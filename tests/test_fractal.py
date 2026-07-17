import pytest
import accretive_mas
import sys
import gc

def test_deep_fractal_nesting():
    # Test recursive depth that might normally trigger issues in non-pointer/non-smart frameworks
    max_depth = 50
    
    # Increase recursion limit slightly just in case python traverse binds heavily, 
    # though C++ does the recursion.
    sys.setrecursionlimit(2000)
    
    root = accretive_mas.MacroAgent()
    current_parent = root
    
    # Build a deep fractal braid (MacroAgent inside MacroAgent)
    for i in range(max_depth):
        child_macro = accretive_mas.MacroAgent()
        
        # Also bind a micro agent at each level for interference
        micro = accretive_mas.MicroAgent([float(i), 0.0, 0.0], 1.0, 1.0, 0.0)
        
        current_parent.cable_bind(child_macro, micro)
        current_parent = child_macro
        
    # At the deepest level, bind a final micro agent
    final_micro = accretive_mas.MicroAgent([100.0, 100.0, 100.0], 5.0, 2.0, 3.14)
    current_parent.cable_bind(final_micro, final_micro)
    
    # 1. Test Traverse Topology
    # C++ handles the recursion natively through shared_ptrs
    root.traverse_topology()
    
    # 2. Test Wave Function Superposition (Deep Recursion)
    # This recursively sums 50+ wave functions down the fractal tree
    wave = root.get_wave_function_at(0.0, 0.0, 0.0, 1.0)
    
    print(f"\n[Fractal Test] Superimposed deep tree wave function: {wave}")
    assert isinstance(wave, complex)
    
    # 3. Memory Clean-up test
    # Ensure deleting the root cascades through the smart pointers without segmentation fault
    del root
    gc.collect()
    
    print("[Fractal Test] Tree de-allocated cleanly.")
