import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add the build directory to the path to import the PyBind11 module
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build'))
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import accretive_mas

def main():
    # 1. Initialize the Substrate and inject MicroAgents
    substrate = accretive_mas.ContinuousSubstrate()
    
    # Inject two agents that will constructively/destructively interfere
    agent1 = accretive_mas.MicroAgent(
        initial_pos=[-2.0, 0.0, 0.0], 
        amplitude=1.5, 
        frequency=2.0, 
        phase=0.0
    )
    agent2 = accretive_mas.MicroAgent(
        initial_pos=[2.0, 0.0, 0.0], 
        amplitude=1.5, 
        frequency=2.0, 
        phase=3.14  # Out of phase
    )
    
    substrate.inject_agent(agent1)
    substrate.inject_agent(agent2)

    # 2. Setup the Visualization Grid (2D slice at Z=0)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Accretive Computing MAS: Continuous Topology (Z=0)")
    ax.set_xlabel("X Space")
    ax.set_ylabel("Y Space")
    ax.set_zlabel("Wave Amplitude")

    # Fixed grid points for visualization
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z_fixed = 0.0

    # Store the surface object so we can remove it during updates
    surface_plot = [None]
    
    time_t = [0.0]
    dt = 0.05

    def update(frame):
        time_t[0] += dt
        t = time_t[0]
        
        # Step the actual C++ simulation physics
        substrate.step_simulation(dt)

        # Calculate the superimposed wave amplitude at each point in the grid
        Z_amplitude = np.zeros_like(X)
        
        # In a real scenario, the substrate would expose the global field evaluation.
        # Here we manually superimpose the known agents for the slice visualization.
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                w1 = agent1.get_wave_function_at(X[i,j], Y[i,j], Z_fixed, t)
                w2 = agent2.get_wave_function_at(X[i,j], Y[i,j], Z_fixed, t)
                
                # Combine their amplitudes (superposition)
                total_wave = w1 + w2
                
                # Render the absolute amplitude (or real part)
                Z_amplitude[i, j] = total_wave.real
        
        if surface_plot[0] is not None:
            surface_plot[0].remove()
        
        # Plot the new 3D surface
        surface_plot[0] = ax.plot_surface(X, Y, Z_amplitude, cmap='viridis', edgecolor='none')
        
        # Keep Z limits fixed to avoid jitter
        ax.set_zlim(-3, 3)
        
        return surface_plot

    # 3. Animate
    # Note: Interval controls refresh rate. blit=False is required for 3D axes.
    anim = FuncAnimation(fig, update, frames=200, interval=50, blit=False)
    
    # Show the interactive window (allows zooming, rotating, panning)
    plt.show()

if __name__ == '__main__':
    main()
