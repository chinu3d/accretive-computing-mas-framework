import os
import sys

# Ensure the core module is found (assuming we run from project root)
sys.path.append(os.path.join(os.getcwd(), "build"))

import accretive_mas
from accretive_mas import LatentSemanticAgent, PhysicsOrchestrator

def main():
    # Note: Requires a valid GEMINI_API_KEY environment variable. 
    # Using a mock string here for demonstration of the API structure.
    os.environ.setdefault("GEMINI_API_KEY", "mock_key")
    
    # 1. Initialize the Substrate and the Physics Orchestrator
    substrate = accretive_mas.ContinuousSubstrate()
    # We set a threshold of 0.8 for topology-driven interaction. 
    # High threshold means they only interact when their latent thoughts heavily overlap.
    orchestrator = PhysicsOrchestrator(substrate, interaction_threshold=0.8)
    
    # 2. Define the Agents
    sales_agent = LatentSemanticAgent(
        name="SalesWeatherBot", 
        system_prompt="You monitor incoming sales data and weather disruptions. Your job is to alert the system of critical supply chain risks.",
        embedding_dim=768
    )
    
    logistics_agent = LatentSemanticAgent(
        name="LogisticsBot", 
        system_prompt="You control shipping routes and inventory orders. You adjust routes to avoid delays and reorder stock when demand spikes.",
        embedding_dim=768
    )
    
    orchestrator.register(sales_agent)
    orchestrator.register(logistics_agent)
    
    # =====================================================================
    # TRADITIONAL MAS APPROACH vs. ACCRETIVE COMPUTING APPROACH
    # =====================================================================
    # In a traditional framework like AutoGen or LangGraph, a developer would have 
    # to explicitly hardcode: `sales_agent.send_message(logistics_agent, "Storm coming")`
    #
    # In Accretive Computing, agents simply "think" (embed their current context into 
    # the N-dimensional space). The physics engine handles the routing!
    # =====================================================================
    
    print("--- Scenario 1: Unrelated Contexts (No Interaction) ---")
    # Agents are focused on totally different things.
    sales_agent.embed_thought("Sales are stable. Clear skies in the midwest.")
    logistics_agent.embed_thought("Standard maintenance on delivery truck fleet #4.")
    
    # Tick the physics engine
    print("Physics Tick...")
    orchestrator.tick(dt=0.1)
    # The latent distance is far apart. Overlap < 0.8. No conversation is triggered!
    
    
    print("\n--- Scenario 2: Converging Contexts (Physics triggers Interaction) ---")
    # A sudden event occurs. Both agents ingest data that pushes their semantic
    # wavefunctions into the same region of the latent topology.
    sales_agent.embed_thought("URGENT: Hurricane approaching Florida port. Massive spike in demand for emergency supplies.")
    logistics_agent.embed_thought("Alert: Port of Florida showing severe weather warnings, possible inbound delays.")
    
    # Tick the physics engine
    print("Physics Tick...")
    # Because both agents' latent vectors now occupy the same conceptual space (Emergency/Florida), 
    # the physics engine's Overlap calculation exceeds the 0.8 threshold.
    # The Orchestrator will automatically trigger a conversation between them, 
    # forcing them to exchange context and resolve the semantic tension!
    
    try:
        # In a real environment with a valid API key, this will print an organic debate/resolution.
        orchestrator.tick(dt=0.1)
    except Exception as e:
        print(f"LLM Interaction skipped due to mock API key: {e}")

if __name__ == "__main__":
    main()
