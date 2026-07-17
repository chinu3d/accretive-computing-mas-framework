import pytest
import os
import accretive_mas
from accretive_mas import LatentSemanticAgent, PhysicsOrchestrator

def test_physics_driven_orchestrator():
    os.environ["GEMINI_API_KEY"] = "mock_api_key_for_testing"
    
    substrate = accretive_mas.ContinuousSubstrate()
    orchestrator = PhysicsOrchestrator(substrate, interaction_threshold=0.0)
    
    agent1 = LatentSemanticAgent("FinanceBot", "Expert in finance", embedding_dim=768)
    agent2 = LatentSemanticAgent("LegalBot", "Expert in law", embedding_dim=768)
    
    # We embed identical thoughts so they are perfectly aligned in latent space
    # (Distance = 0 -> Max Overlap)
    thought = "We need to discuss the upcoming merger."
    agent1.embed_thought(thought)
    agent2.embed_thought(thought)
    
    orchestrator.register(agent1)
    orchestrator.register(agent2)
    
    # Tick the physics engine. Because they are at identical coordinates, overlap is ~1.0
    # Our threshold is 0.0, so this MUST trigger a topology collision and conversation!
    
    # In a real setup, genai.Client() would hit the network. Since we have a mock API key,
    # the LLM call will actually fail unless we mock the client.
    # For this test, we can mock the `process_llm_step` just to prove the physics routes the message.
    
    a1_called = False
    a2_called = False
    
    def mock_process_llm_step_a1(prompt):
        nonlocal a1_called
        a1_called = True
        return "Finance perspective: Let's do it."
        
    def mock_process_llm_step_a2(prompt):
        nonlocal a2_called
        a2_called = True
        return "Legal perspective: Ensure compliance."
        
    agent1.process_llm_step = mock_process_llm_step_a1
    agent2.process_llm_step = mock_process_llm_step_a2
    
    orchestrator.tick(dt=0.1)
    
    assert a1_called, "Physics orchestrator failed to trigger Agent 1"
    assert a2_called, "Physics orchestrator failed to trigger Agent 2"
    
    # Verify inter-agent message queues populated
    assert "Finance perspective" in agent2.inbox[0]
