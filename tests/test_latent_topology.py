import pytest
import accretive_mas
from accretive_mas import LatentSemanticAgent, SubstrateDashboard
import numpy as np
import os

def test_latent_topology():
    os.environ["GEMINI_API_KEY"] = "mock_api_key_for_testing"
    
    # 1. Create Substrate and Dashboard
    substrate = accretive_mas.ContinuousSubstrate()
    dashboard = SubstrateDashboard(substrate)
    
    # 2. Create Latent Agents
    agent1 = LatentSemanticAgent("FinanceBot", "Expert in finance", embedding_dim=768)
    agent2 = LatentSemanticAgent("LegalBot", "Expert in law", embedding_dim=768)
    
    # 3. Embed thoughts (pushes them into 768-D space)
    agent1.embed_thought("What is the quarterly revenue?")
    agent2.embed_thought("Reviewing the NDA contract terms.")
    
    # Check that they are in 768-D space
    pos1 = agent1.micro_agent.get_position()
    assert len(pos1) == 768
    
    # 4. Inject into Substrate
    substrate.inject_agent(agent1.micro_agent)
    substrate.inject_agent(agent2.micro_agent)
    dashboard.register(agent1)
    dashboard.register(agent2)
    
    # 5. O(N log N) Free energy compute
    energy = substrate.calculate_free_energy()
    print(f"\nSystem Free Energy: {energy}")
    
    # 6. DSL: Seek Consensus
    # FinanceBot tries to align with LegalBot
    agent1.seek_consensus_with(agent2, strength=0.5)
    
    # 7. Render Visualization
    dashboard.render()
    assert os.path.exists("topology.png")
