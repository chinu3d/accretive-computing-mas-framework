# Accretive Computing Framework: The Comprehensive Developer's Guide

Welcome to the **Accretive Computing Framework**, the world’s first continuous, physics-driven Multi-Agent System (MAS). This guide is designed for AI architects, researchers, and developers who are looking to push beyond the limitations of rigid, graph-based agent orchestration.

## 1. Introduction to the Framework

### 1.1. The Bottleneck of Traditional Multi-Agent Systems
Over the last few years, frameworks like AutoGen, CrewAI, and LangGraph have popularized the concept of Multi-Agent Systems (MAS). However, they all share a fundamental architectural bottleneck: **Graph-Based Routing**.

In traditional systems, agents are treated as discrete software nodes. Developers must explicitly wire these nodes together using rules, state machines, or directed acyclic graphs (DAGs).
* _If the Researcher Agent finds a source, send it to the Writer Agent._
* _If the Writer Agent finishes a draft, send it to the Editor Agent._

This works for 3 to 5 agents. But what happens when you scale to 500 agents? Or 10,000 agents? The routing logic becomes an unmaintainable \(O(N^2)\) nightmare of conditional statements. Furthermore, it is entirely rigid; agents cannot spontaneously discover new collaborators or react to context outside of their hardcoded pathways.

### 1.2. The Paradigm Shift: Agents as Wave Functions
**Accretive Computing** proposes a radical, bio-inspired paradigm shift. What if we stop treating agents as discrete software components sending JSON payloads, and start treating them as physical wave functions in a continuous topological space? In the real world, human collaboration is organic. If you are at a crowded conference thinking about “Quantum Computing” and you overhear someone nearby talking about “Qubits”, you naturally drift toward them and strike up a conversation. There is no central router telling you to speak to them; the interaction is driven by spatial and semantic proximity. Accretive Computing maps this exact phenomenon into software:

A. **Continuous Topology**: The system is an \(N\)-dimensional spatial simulation (the `ContinuousSubstrate`).
B. **Semantic Projection**: When an LLM generates a thought, that text is embedded into a vector. This vector becomes the agent's physical \((X, Y, Z, \dots)\) coordinate in the substrate.
C. **Physics-Driven Orchestration**: As agents think, they move. When their topological footprints overlap (i.e., their wave functions constructively interfere), the underlying C++ physics engine detects the collision and autonomously forces them to converse. There is no routing logic. You just pour agents into the substrate, and the physics engine orchestrates them organically.

## 2. The Mathematics Behind the Framework

To understand how the engine achieves this without catastrophic computational overhead, we must dive into the underlying physics model.

### 2.1. The Original 3D Wave Equation
In our earliest prototypes, an agent \(A\) was modeled as an oscillating field source in a standard 3D space, position \(\mathbf{p} = (x, y, z)\). The influence of the agent across the space \(\mathbf{r}\) at time \(t\) was defined by a classic wave equation:

\[\Psi_A(\mathbf{r}, t) = A \cdot \exp( - |\mathbf{r} - \mathbf{p}|) \cdot \exp(i(\omega t + \theta_A))\]

* \(A\): Amplitude (the agent’s assertiveness or “volume”).
* \(\mathbf{p}\): Physical coordinate (the semantic topic).
* \(\omega\): Frequency (the thought-cycle rate).
* \(\theta_A\): Semantic Phase Angle (the agent’s stance or opinion on the topic).

While mathematically elegant, standard 3D space is insufficient for the semantic complexity of Large Language Models.

### 2.2. The N-Dimensional Latent Topology
To bridge the physics engine with modern LLMs, we expanded the substrate into \(N\)-dimensional space. For a standard embedding model (e.g., OpenAI `text-embedding-3-small` or Google’s `text-embedding-004`), \(N = 768\) or \(1536\).

An agent’s thought is embedded into a vector \(\mathbf{x} \in \mathbb{R}^N\). This vector becomes its precise coordinate in the topology.

However, integrating a continuous wave field over 768 dimensions using standard Monte Carlo volumetric integration is computationally impossible (_The Curse of Dimensionality_). To solve this, we derived an exact analytical integral for the overlap between two agents using a Gaussian Radial Basis Function (RBF):

\[Overlap(A, B) = A_A A_B \cdot \exp \left( -\frac{1}{2} ||\mathbf{x}_A - \mathbf{x}_B||^2 \right) \cdot \cos(\theta_A - \theta_B)\]

Notice the cosine term. If two agents are talking about the exact same topic (\(\mathbf{x}_A \approx \mathbf{x}_B\)), but they fiercely disagree (their phases \(\theta\) are \(\pi\) radians apart), the cosine term evaluates to \(-1\). This represents destructive interference—a semantic deadlock.

### 2.3. Thermodynamic Free Energy Minimization
The universal goal of the Accretive MAS is to resolve semantic tension and reach a consensus ground state. We define the Hamiltonian (Free Energy \(F\)) of the entire substrate as the sum of all pairwise mismatches, minus a natural dissipation constant \(\delta\):

\[F(\Theta) = \sum_i \sum_{j>i} \left( 1 - Overlap(A_i, A_j) - \delta \right)\]

To step the simulation forward, the C++ calculates the analytical gradient of this Free Energy landscape and applies continuous Gradient Descent to the phase angles:

\[\frac{\partial \theta_i}{\partial t} = - \alpha \frac{\partial F}{\partial \theta_i}\]

As time \(t\) advances, the physics engine physically “drags” the opinions (phases) of the interacting agents toward alignment, simulating the process of debate and compromise.

### 2.4. Achieving O(N log N) Scalability
A naïve calculation of \(F(\Theta)\) requires \(O(N^2)\) pairwise checks, which destroys performance for large swarms. Accretive Computing utilizes a 1-level spatial partitioning algorithm:

I. **Sort**: Agents are dynamically sorted along their dominant Principal Component in \(O(N \log N)\) time.
II. **Cluster**: They are chunked into contiguous blocks of size \(B \approx \sqrt{N}\).
III. **Multipole Expansion**: Within a cluster, exact RBF overlaps are computed. Between distant clusters, the overlap is approximated using the center-of-mass (the centroid). This reduces the global interaction complexity to \(O(N \log N)\), enabling massive agent swarms.

## 3. How to Build and Install the Framework from scratch

Because Accretive Computing relies on extremely high-performance physics calculations, the core engine is written in strict C++20 and exposed to Python via PyBind11.

### 3.1. Prerequisites
You must have a C++ toolchain capable of compiling C++20.
* **macOS:** `xcode-select --install` and `brew install cmake`
* **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install build-essential cmake python3-dev`
* **Windows:** Visual Studio 2022 with C++ workload, or WSL2 (recommended).

You also need **Eigen3** (a C++ template library for linear algebra). The CMake script will attempt to find it locally; if not, it is highly recommended to install it via your package manager (e.g., `brew install eigen`).

### 3.2. Building the C++ Engine
Clone the repository and build the native extension:

```bash
git clone https://github.com/accretive-computing/accretive-mas-framework.git
cd accretive-mas-framework
mkdir build
cd build
cmake ..
make -j4
```

_Note: We use -j4 to utilize 4 cores during compilation. The output will be a shared object file named accretive_mas.cpython-314-darwin.so or similar, depending, typically, on your OS and Python version._

### 3.3. Python Environment Setup
Return to the project root and set up a virtual environment.

```bash
cd ..
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

_Dependencies include google-genai for the LLM integrations, numpy for vector manipulation, and matplotlib / networkx for topological visualization._

**CRITICAL:** You must export the build directory to your PYTHONPATH so Python can discover the compiled C++ module!

```bash
export PYTHONPATH="$(pwd)/build:$PYTHONPATH"
```

### 3.4. API Keys
The framework’s `LatentSemanticAgent` defaults to using Google’s Gemini models (`gemini-2.5-flash`) for extreme speed and context windows.

```bash
export GEMINI_API_KEY="your_api_key_here"
```

## 4. Step-by-Step Guide on How to Use the Framework

This section will guide you through the process of writing an organic MAS script.

### 4.1. Importing and Initialization
Create a file named `main.py`. Start by initializing the C++ substrate and the Python Orchestrator.

```python
import accretive_mas 
from accretive_mas_llm import LatentSemanticAgent, PhysicsOrchestrator 

# Initialize the O(N log N) C++ physics engine substrate
substrate = accretive_mas.ContinuousSubstrate() 

# Initialize the Python Orchestrator. 
# interaction_threshold=0.85 means agents will only converse if their 
# topological wavefunctions overlap by 85% or more. 
orchestrator = PhysicsOrchestrator(substrate, interaction_threshold=0.85)
```

### 4.2. Creating Stateful Agents
The `LatentSemanticAgent` is a powerful wrapper. Under the hood, it maintains a persistent `genai.chats` session (so it never forgets conversation history) and holds a C++ `LatentMicroAgent` (so it exists in the physics engine).

```python
# Create the agents 
agent_alpha = LatentSemanticAgent( 
    name="Alpha", 
    system_prompt="You are a brilliant software architect focusing on backend scale.", 
    embedding_dim=768 
) 

agent_beta = LatentSemanticAgent( 
    name="Beta", 
    system_prompt="You are a frontend UI/UX expert focusing on user conversion.", 
    embedding_dim=768 
) 

# Register them with the Orchestrator (which injects them into the C++ Substrate) 
orchestrator.register(agent_alpha) 
orchestrator.register(agent_beta)
```

### 4.3. Binding Tools
Agents can interact with the outside world via Python functions.

```python
def check_server_load(region: str) -> str: 
    """Returns the CPU load for a given server region.""" 
    return f"Load in {region} is currently at 94%." 
    
# Bind the tool to the agent 
agent_alpha.bind_tool(check_server_load)
```

When an agent successfully executes a tool, the framework automatically applies a “Semantic Phase Spike” (\(\theta += 0.5\)) in the physics engine, representing the influx of new, unassimilated data into the system.

### 4.4. Embedding Thoughts
To move an agent in the topological space, you embed its “thoughts”. _(Note: In the current open-source prototype, `embed_thought` uses a stable deterministic hash projection for speed, but in production, this should be wired to an embedding model)_.

```python
# Move Alpha to the “Backend Scale” region of the topology 
agent_alpha.embed_thought("How do we handle 10 million concurrent websockets?") 

# Move Beta to the “Frontend” region of the topology 
agent_beta.embed_thought("The button padding needs to be increased by 2px.")
```

### 4.5. The Physics Tick Loop
In a traditional MAS, this is where you would write `alpha.send(beta)`. We do not do that here. Instead, we advance time.

```python
import time 

print("Starting Simulation…") 
while True: 
    # Advance the continuous simulation by dt = 0.1 
    # The orchestrator will check overlaps. If Alpha and Beta drift into the 
    # same topological space, it will autonomously trigger a debate! 
    orchestrator.tick(dt=0.1) 
    time.sleep(1)
```

## 5. Examples, Tests, and Results

To truly grasp the power of the framework, let’s examine two concrete implementations included in the repository.

### 5.1. The Supply Chain Disruption (examples/supply_chain.py)
This script models a logistics company.
* `SalesBot` monitors sales data.
* `LogisticsBot` monitors truck routes.

**Scenario A: Routine Operations**
`SalesBot` embeds a thought about “Midwest sales”. `LogisticsBot` embeds a thought about “Truck Maintenance”. When `orchestrator.tick()` runs, the C++ engine computes the distance between these two vectors. Because the concepts are mathematically orthogonal, the resulting Overlap is \(\approx 0.36\). Since this is below the `0.8` threshold, the physics engine ignores the interaction. The agents remain independent and silent, saving massive amounts of API token costs.

**Scenario B: The Convergence Event**
A hurricane approaches Florida. `SalesBot` embeds: _“Urgent: Massive demand spike in Florida due to Hurricane”_. `LogisticsBot` embeds: _“Alert: Florida ports closing due to severe weather”_.

When `orchestrator.tick()` runs, the physics engine detects a massive topological collision (Overlap \(\approx 1.00\)). The Orchestrator immediately intercepts this physical event and bridges it to the LLM layer:
1. It prompts `SalesBot`: _“You have collided in thought-space with LogisticsBot. State your perspective”_.
2. `SalesBot` generates a response, which is routed into `LogisticsBot`’s asynchronous inbox.
3. `LogisticsBot` reads its inbox, processes the context, and responds with a rerouting plan.
4. The physics engine then applies a mathematical recoil (\(\theta_A += 0.1\), \(\theta_B -= 0.1\)) to push them slightly apart, preventing an infinite conversational loop.

This entire collaboration happened without a single explicit routing command from the developer!

### 5.2. The Substrate Dashboard (tests/test_latent_topology.py)
Because it is impossible for human developers to visualize a 768-dimensional space, the framework includes a `SubstrateDashboard`.

In this test, after agents embed their thoughts and interact, we call:

```python
dashboard.render()
```

This extracts the exact pairwise overlap weights from the C++ `ContinuousSubstrate`, pipes them into `networkx`, and uses a spring-layout algorithm to project the \(N\)-dimensional semantic deadlock down into a 2D `matplotlib` graph (`topology.png`). Developers can visually see which agents are tightly clustered (collaborating) and which agents are drifting in isolation.

## 6. Conclusion and Implications

The Accretive Computing framework represents a robust and highly scalable alternative to the static MAS routing graph. By treating agents as physical entities rather than software scripts, we have unlocked an architecture with profound implications for the future of AI and computing in general.

### 6.1. Infinite Scalability
When routing logic is handled by a spatial physics engine rather than conditional if-statements, the cost of adding agents drops to near zero. A developer can spin up 10,000 independent agents monitoring 10,000 different datastreams. They will only incur LLM inference costs when their wave functions actually collide in the latent space. The \(O(N \log N)\) C++ backend easily handles the topological tracking of these massive swarms.

### 6.2. Organic Swarm Intelligence
Natural organizations do not operate on rigid Directed Acyclic Graphs. They operate on proximity, shared context, and spontaneous collaboration. Accretive Computing allows AI swarms to behave identically. Agents can form temporary sub-committees simply by thinking about the same problem simultaneously, resolving the issue, and then naturally drifting apart in the topology.

### 6.3. Future Roadmap
The `v0.1.1` release is just the foundation. Our roadmap for `v1.0.0` includes:
* **GPU Acceleration**: Moving the `ContinuousSubstrate` from Eigen3 CPU compute to CUDA, allowing for real-time simulation of millions of agents.
* **Multi-Modal Topology**: Projecting Image Embeddings (e.g., CLIP vectors) into the same substrate, allowing vision agents and text agents to physically collide over shared concepts.
* **Dynamic Dimensional Reduction**: Using real-time UMAP within the C++ engine to further optimize the clustering algorithms.

Welcome to the future of Multi-Agent Systems. Happy coding!
