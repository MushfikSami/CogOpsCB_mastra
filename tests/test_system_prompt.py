"""test_system_prompt.py — Phase 1: system prompt content tests."""
from cogops.prompts.system import get_graph_prompt


class TestSystemPrompt:
    def test_returns_non_empty(self):
        prompt = get_graph_prompt("TestAgent", "A test agent.", "[]")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_replaces_agent_name(self):
        prompt = get_graph_prompt("TestAgent", "A test agent.", "[]")
        assert "TestAgent" in prompt

    def test_replaces_agent_story(self):
        prompt = get_graph_prompt("TestAgent", "This is my story.", "[]")
        assert "This is my story" in prompt

    def test_replaces_tools_description(self):
        tools_desc = '{"function": {"name": "test_tool"}}'
        prompt = get_graph_prompt("A", "B", tools_desc)
        assert 'test_tool' in prompt

    def test_contains_bengali_language_rules(self):
        prompt = get_graph_prompt("A", "B", "[]")
        assert "সেবা" in prompt or "প্রমিত বাংলা" in prompt

    def test_contains_safety_tiers(self):
        prompt = get_graph_prompt("A", "B", "[]")
        assert "TIER 1" in prompt
        assert "TIER 2" in prompt
        assert "TIER 3" in prompt

    def test_contains_reasoning_framework(self):
        prompt = get_graph_prompt("A", "B", "[]")
        # 5-phase: Analyze, Disambiguate, Plan, Act, Synthesize
        for word in ["Analyze", "Disambiguate", "Plan", "Act", "Synthesize"]:
            assert word in prompt, f"Missing reasoning stage: {word}"

    def test_contains_tool_doctrine(self):
        prompt = get_graph_prompt("A", "B", "[]")
        assert "Tool Usage" in prompt or "TOOL" in prompt

    def test_contains_zero_hallucination(self):
        prompt = get_graph_prompt("A", "B", "[]")
        assert "Zero Hallucination" in prompt or "NEVER" in prompt

    def test_contains_neutrality(self):
        prompt = get_graph_prompt("A", "B", "[]")
        assert "Strict Neutrality" in prompt or "NEUTRALITY" in prompt

    def test_contains_official_persona(self):
        prompt = get_graph_prompt("A", "B", "[]")
        assert "Official Persona" in prompt or "government" in prompt.lower()
