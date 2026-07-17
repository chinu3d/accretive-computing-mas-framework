import pytest
import os
import accretive_mas
from accretive_mas import LatentSemanticAgent, PhysicsOrchestrator

def setup_orchestrator():
    os.environ["GEMINI_API_KEY"] = "mock_api_key_for_testing"
    substrate = accretive_mas.ContinuousSubstrate()
    # High threshold means they must be very close in latent space
    orchestrator = PhysicsOrchestrator(substrate, interaction_threshold=0.8)
    
    sales_agent = LatentSemanticAgent("SalesBot", "Monitors sales", embedding_dim=768)
    log_agent = LatentSemanticAgent("LogisticsBot", "Monitors logistics", embedding_dim=768)
    
    orchestrator.register(sales_agent)
    orchestrator.register(log_agent)
    
    return orchestrator, sales_agent, log_agent

def test_scenario_a_unrelated_contexts():
    """Scenario A: Agents are focused on totally different things. Overlap should be low."""
    orchestrator, sales_agent, log_agent = setup_orchestrator()
    
    # Embed unrelated thoughts. Our mock `embed_thought` uses a hash to generate a stable random vector.
    # Different strings will be orthogonal in 768-D space (distance ~ sqrt(2), overlap ~ 0.36)
    sales_agent.embed_thought("Sales are stable in the midwest.")
    log_agent.embed_thought("Standard maintenance on delivery truck fleet #4.")
    
    a1_called = False
    
    def mock_process_llm_step(prompt):
        nonlocal a1_called
        a1_called = True
        return "Response"
        
    sales_agent.process_llm_step = mock_process_llm_step
    
    orchestrator.tick(dt=0.1)
    
    # Because overlap < 0.8, the orchestrator should NOT trigger an interaction
    assert not a1_called, "Orchestrator falsely triggered an interaction for unrelated contexts!"

def test_scenario_b_converging_contexts():
    """Scenario B: Agents embed similar contexts (Florida Hurricane). Overlap should be high."""
    orchestrator, sales_agent, log_agent = setup_orchestrator()
    
    # We use identical thoughts so the mock embedding places them at the exact same coordinate.
    # Overlap will be 1.0 (since distance is 0).
    critical_thought = "Hurricane approaching Florida port. Massive disruption expected."
    sales_agent.embed_thought(critical_thought)
    log_agent.embed_thought(critical_thought)
    
    a1_called = False
    a2_called = False
    
    def mock_process_a1(prompt):
        nonlocal a1_called
        a1_called = True
        return "Sales alerting spike."
        
    def mock_process_a2(prompt):
        nonlocal a2_called
        a2_called = True
        return "Logistics rerouting."
        
    sales_agent.process_llm_step = mock_process_a1
    log_agent.process_llm_step = mock_process_a2
    
    orchestrator.tick(dt=0.1)
    
    # Because overlap is 1.0 (which is > 0.8), the orchestrator MUST trigger an interaction
    assert a1_called, "Orchestrator failed to trigger Agent 1 when contexts converged!"
    assert a2_called, "Orchestrator failed to trigger Agent 2 when contexts converged!"
    
    # Ensure they actually sent messages to each other's inboxes
    assert "Sales alerting spike" in log_agent.inbox[0]
