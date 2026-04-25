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
        assert "chitchat" in prompt
        assert "identity" in prompt
        assert "safety_deflect" in prompt

    def test_contains_reasoning_framework(self):
        prompt = get_graph_prompt("A", "B", "[]")
        # 5-phase: Intent, Classification, Follow-up check, Plan, Synthesize
        for word in ["Intent", "Classification", "Plan", "Synthesize"]:
            assert word in prompt, f"Missing reasoning stage: {word}"

    def test_contains_tool_doctrine(self):
        prompt = get_graph_prompt("A", "B", "[]")
        assert "tool" in prompt.lower() and "Tool selection" in prompt

    def test_contains_zero_hallucination(self):
        prompt = get_graph_prompt("A", "B", "[]")
        assert "Never" in prompt or "NEVER" in prompt

    def test_contains_neutrality(self):
        prompt = get_graph_prompt("A", "B", "[]")
        assert "strict neutrality" in prompt.lower() or "neutrality" in prompt.lower()

    def test_contains_official_persona(self):
        prompt = get_graph_prompt("A", "B", "[]")
        assert "government" in prompt.lower()

    def test_no_wikipedia_references(self):
        """Verify Wikipedia tools have been removed from the prompt."""
        prompt = get_graph_prompt("A", "B", "[]")
        assert "wikipedia" not in prompt.lower()
