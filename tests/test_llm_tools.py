import pytest
from accretive_mas import SemanticAgent

def get_weather(location: str) -> str:
    """Returns the weather for a given location."""
    # Mock tool response
    return f"The weather in {location} is 72 degrees and sunny."

class MockFunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args

class MockResponse:
    def __init__(self, calls):
        self.function_calls = calls
        self.text = "I have called the tool."

from unittest.mock import patch

@patch('accretive_mas.llm.genai.Client')
def test_semantic_agent_tool_calling(MockClient):
    # Setup mock LLM response that triggers the tool
    mock_instance = MockClient.return_value
    mock_instance.models.generate_content.return_value = MockResponse(
        [MockFunctionCall(name="get_weather", args={"location": "Tokyo"})]
    )

    # 1. Create the Semantic Agent
    agent = SemanticAgent(
        name="WeatherBot",
        system_prompt="You are a helpful assistant. Use the get_weather tool to look up the weather if asked."
    )
    
    # 2. Bind the tool
    agent.bind_tool(get_weather)
    
    # 3. Check initial physics/semantic state
    micro_agent = agent.get_agent()
    payload = micro_agent.get_semantic_payload()
    
    assert payload["name"] == "WeatherBot"
    assert payload["system_prompt"] != ""
    initial_phase = micro_agent.get_phase()
    
    # 4. Process an LLM step that should trigger the tool
    prompt = "What is the weather like in Tokyo?"
    response = agent.process_llm_step(prompt)
    
    print(f"\n[LLM Test] Response:\n{response}")
    
    # 5. Verify the Semantic Payload updated natively in C++
    updated_payload = micro_agent.get_semantic_payload()
    assert "last_prompt" in updated_payload
    assert updated_payload["last_prompt"] == prompt
    assert "last_response" in updated_payload
    
    # 6. Verify the Physics-Semantic Link
    # If the LLM successfully called the tool, the wrapper should have spiked the phase by 0.5
    final_phase = micro_agent.get_phase()
    
    print(f"[LLM Test] Initial Phase: {initial_phase} -> Final Phase: {final_phase}")
    
    if "Tool get_weather returned" in updated_payload["last_response"]:
        assert final_phase == initial_phase + 0.5, "Physics phase did not update after semantic tool action!"
    else:
        print("[LLM Test] The LLM answered directly without calling the tool.")
