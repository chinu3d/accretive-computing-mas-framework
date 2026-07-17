import os
from google import genai
from google.genai import types
from . import _core as accretive_mas

class SemanticAgent:
    def __init__(self, name: str, system_prompt: str, initial_phase: float = 0.0):
        self.name = name
        self.system_prompt = system_prompt
        
        # The underlying physical representation in the topology
        self.micro_agent = accretive_mas.MicroAgent([0.0, 0.0, 0.0], 1.0, 1.0, initial_phase)
        
        # Initialize the C++ unordered_map payload
        self.micro_agent.update_semantic_payload("name", name)
        self.micro_agent.update_semantic_payload("system_prompt", system_prompt)
        
        # Initialize LLM Client (Expects GEMINI_API_KEY in environment)
        self.client = genai.Client()
        self.tools = []
        self.tool_functions = {}
        
    def get_agent(self):
        """Returns the underlying C++ MicroAgent to be injected into the Substrate."""
        return self.micro_agent
        
    def bind_tool(self, func):
        """Binds a Python function as a tool for the LLM."""
        self.tool_functions[func.__name__] = func
        self.tools.append(func)
        
    def receive_message(self, sender_name: str, message: str):
        """Buffers an incoming message from another agent."""
        if not hasattr(self, 'inbox'):
            self.inbox = []
        self.inbox.append(f"[{sender_name}]: {message}")

    def send_message(self, recipient: 'SemanticAgent', message: str):
        """Sends a message to another agent."""
        recipient.receive_message(self.name, message)

    def process_llm_step(self, prompt: str) -> str:
        """Runs the LLM, executes tools, and dynamically updates the physics topology."""
        if not hasattr(self, 'inbox'):
            self.inbox = []
            
        config_kwargs = {}
        if self.system_prompt:
            config_kwargs["system_instruction"] = self.system_prompt
        if self.tools:
            config_kwargs["tools"] = self.tools
            
        # Process inbox if any
        if self.inbox:
            inbox_text = "\n".join(self.inbox)
            prompt = f"New messages received from other agents:\n{inbox_text}\n\nYour task/prompt: {prompt}"
            self.inbox.clear()

        # Initialize stateful chat if not exists
        if not hasattr(self, 'chat') or self.chat is None:
            self.chat = self.client.chats.create(
                model='gemini-2.5-flash',
                config=types.GenerateContentConfig(**config_kwargs)
            )

        try:
            # Send message to the stateful chat
            response = self.chat.send_message(prompt)
            
            # Handle tool calls manually
            if response.function_calls:
                function_responses = []
                for function_call in response.function_calls:
                    name = function_call.name
                    args = function_call.args
                    if name in self.tool_functions:
                        tool_result = self.tool_functions[name](**args)
                        
                        function_responses.append(
                            types.Part.from_function_response(
                                name=name,
                                response={"result": str(tool_result)}
                            )
                        )
                        
                        # [Physics-Semantic Link]
                        # Successful tool execution causes a "semantic spike" in the agent's phase
                        current_phase = self.micro_agent.get_phase()
                        self.micro_agent.set_phase(current_phase + 0.5)
                        
                # Send the function responses back to the model to get the final text response
                response = self.chat.send_message(function_responses)
                
            output_text = response.text if response.text else "No text response."
                
            # Store the semantic interaction natively in the C++ vector field
            self.micro_agent.update_semantic_payload("last_prompt", prompt)
            self.micro_agent.update_semantic_payload("last_response", output_text)
            
            return output_text
            
        except Exception as e:
            error_msg = f"LLM Error: {str(e)}"
            self.micro_agent.update_semantic_payload("last_error", error_msg)
            return error_msg


class LatentSemanticAgent(SemanticAgent):
    def __init__(self, name: str, system_prompt: str, embedding_dim: int = 768, initial_phase: float = 0.0):
        self.name = name
        self.system_prompt = system_prompt
        
        # Start at origin in latent space
        import numpy as np
        initial_latent = np.zeros(embedding_dim)
        
        # Use LatentMicroAgent instead of MicroAgent
        self.micro_agent = accretive_mas.LatentMicroAgent(initial_latent, 1.0, 1.0, initial_phase)
        
        self.micro_agent.update_semantic_payload("name", name)
        self.micro_agent.update_semantic_payload("system_prompt", system_prompt)
        
        self.client = genai.Client()
        self.tools = []
        self.tool_functions = {}

    def seek_consensus_with(self, other_agent: 'LatentSemanticAgent', strength: float = 1.0):
        """DSL: Forces this agent to align its phase with another agent to minimize mismatch in the physics engine."""
        # Simple heuristic: move our phase closer to their phase
        my_phase = self.micro_agent.get_phase()
        their_phase = other_agent.micro_agent.get_phase()
        self.micro_agent.set_phase(my_phase + strength * (their_phase - my_phase))
        
    def embed_thought(self, text: str):
        """Takes an LLM thought and projects it down into the topological Latent Space."""
        import numpy as np
        # In a real app we'd call text-embedding-004 here
        # For prototype, we generate a stable random projection based on the text hash
        np.random.seed(abs(hash(text)) % (2**32))
        dim = len(self.micro_agent.get_position())
        embedding = np.random.randn(dim)
        embedding = embedding / np.linalg.norm(embedding)
        self.micro_agent.set_position(embedding)

import matplotlib.pyplot as plt
import networkx as nx

class SubstrateDashboard:
    def __init__(self, substrate):
        self.substrate = substrate
        self.agents = []
        
    def register(self, agent: SemanticAgent):
        self.agents.append(agent)
        
    def render(self):
        """Draws a real-time topology map of the agents' latent space."""
        G = nx.Graph()
        
        for agent in self.agents:
            name = agent.micro_agent.get_semantic_payload().get("name", "Unknown")
            G.add_node(name)
            
        # Draw edges based on semantic deadlock / overlap
        for i, a1 in enumerate(self.agents):
            for j, a2 in enumerate(self.agents):
                if i < j:
                    overlap = a1.micro_agent.calculate_overlap_with(a2.micro_agent)
                    if overlap > 0.1:
                        G.add_edge(
                            a1.micro_agent.get_semantic_payload().get("name"),
                            a2.micro_agent.get_semantic_payload().get("name"),
                            weight=overlap
                        )
                        
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, weight='weight')
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=2000, font_size=10, font_weight='bold')
        
        plt.title("Accretive Computing Latent Topology")
        plt.savefig("topology.png")
        plt.close()

class PhysicsOrchestrator:
    """The killer feature: Physics-driven LLM orchestration."""
    def __init__(self, substrate, interaction_threshold: float = 0.5):
        self.substrate = substrate
        self.agents = []
        self.interaction_threshold = interaction_threshold
        
    def register(self, agent: SemanticAgent):
        self.agents.append(agent)
        self.substrate.inject_agent(agent.micro_agent)
        
    def tick(self, dt: float = 0.1):
        """Advances the continuous physics simulation by dt, then checks for interactions."""
        # 1. Physics Step (Gradient Descent)
        self.substrate.step_simulation(dt)
        
        # 2. Topology-driven Interaction
        for i, a1 in enumerate(self.agents):
            for j, a2 in enumerate(self.agents):
                if i < j:
                    overlap = a1.micro_agent.calculate_overlap_with(a2.micro_agent)
                    
                    if overlap > self.interaction_threshold:
                        print(f"\n[Physics Orchestrator] ⚡ Toplogy Collision Detected: {a1.name} & {a2.name} (Overlap: {overlap:.2f})")
                        
                        # The physics engine forces them to communicate
                        prompt_a1 = f"You have collided in latent thought-space with {a2.name}. Based on your role, briefly state your perspective or share an insight."
                        print(f"{a1.name} is thinking...")
                        response_a1 = a1.process_llm_step(prompt_a1)
                        print(f"[{a1.name}]: {response_a1}")
                        
                        # Route message to a2
                        a1.send_message(a2, response_a1)
                        
                        prompt_a2 = f"Respond to {a1.name} who just encroached on your topological space with their previous message."
                        print(f"{a2.name} is thinking...")
                        response_a2 = a2.process_llm_step(prompt_a2)
                        print(f"[{a2.name}]: {response_a2}")
                        
                        # Route back to a1 for their inbox
                        a2.send_message(a1, response_a2)
                        
                        # To prevent infinite looping on the same threshold, we can push them apart slightly
                        # In a true physics engine, they'd naturally drift or attract. We manually perturb.
                        a1.micro_agent.set_phase(a1.micro_agent.get_phase() + 0.1)
                        a2.micro_agent.set_phase(a2.micro_agent.get_phase() - 0.1)

