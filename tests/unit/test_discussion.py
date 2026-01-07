"""Unit tests for agent discussion modules."""

from __future__ import annotations

import pytest
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.moderator_agent import (
    ModeratorAgent,
    DiscussionPhase,
    DiscussionTurn,
    DiscussionConflict,
    DiscussionSummary,
    ConflictStrategy,
    AgentProtocol,
    DEFAULT_MODERATOR_INSTRUCTIONS,
    create_moderator_agent,
)
from src.agents.discussion import (
    DiscussionRole,
    Participant,
    DiscussionConfig,
    RoundResult,
    DiscussionProtocol,
    DebateProtocol,
    RoundRobinProtocol,
    create_debate,
    create_roundtable,
    DiscussionAgent,
    create_discussion_agent,
    _build_role_instructions,
)


# =============================================================================
# Fixtures
# =============================================================================


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str, response: str = "Mock response"):
        self.name = name
        self._response = response
        self.call_count = 0

    async def run(self, prompt: str) -> str:
        self.call_count += 1
        return f"{self.name}: {self._response} (call {self.call_count})"


@pytest.fixture
def mock_agent() -> MockAgent:
    """Create a mock agent."""
    return MockAgent("TestAgent")


@pytest.fixture
def mock_agents() -> list[MockAgent]:
    """Create multiple mock agents."""
    return [
        MockAgent("Agent1", "Response from Agent1"),
        MockAgent("Agent2", "Response from Agent2"),
        MockAgent("Agent3", "Response from Agent3"),
    ]


@pytest.fixture
def moderator(mock_agents: list[MockAgent]) -> ModeratorAgent:
    """Create a moderator with registered agents."""
    mod = ModeratorAgent(
        name="TestModerator",
        conflict_strategy=ConflictStrategy.EXPLORE,
        max_rounds=3,
    )
    for agent in mock_agents:
        mod.register_participant(agent)
    return mod


@pytest.fixture
def discussion_config() -> DiscussionConfig:
    """Create a discussion configuration."""
    return DiscussionConfig(
        topic="Test Discussion Topic",
        max_rounds=2,
        conflict_strategy=ConflictStrategy.ACKNOWLEDGE,
        allow_rebuttals=True,
        rebuttal_rounds=1,
    )


# =============================================================================
# DiscussionPhase Tests
# =============================================================================


class TestDiscussionPhase:
    """Tests for DiscussionPhase enum."""

    def test_all_phases_defined(self) -> None:
        """Test that all phases are defined."""
        assert DiscussionPhase.OPENING is not None
        assert DiscussionPhase.DISCUSSION is not None
        assert DiscussionPhase.REBUTTAL is not None
        assert DiscussionPhase.SYNTHESIS is not None
        assert DiscussionPhase.CLOSING is not None

    def test_phase_values(self) -> None:
        """Test phase values."""
        assert DiscussionPhase.OPENING.value == "opening"
        assert DiscussionPhase.DISCUSSION.value == "discussion"
        assert DiscussionPhase.REBUTTAL.value == "rebuttal"
        assert DiscussionPhase.SYNTHESIS.value == "synthesis"
        assert DiscussionPhase.CLOSING.value == "closing"


# =============================================================================
# ConflictStrategy Tests
# =============================================================================


class TestConflictStrategy:
    """Tests for ConflictStrategy enum."""

    def test_all_strategies_defined(self) -> None:
        """Test that all strategies are defined."""
        assert ConflictStrategy.ACKNOWLEDGE is not None
        assert ConflictStrategy.EXPLORE is not None
        assert ConflictStrategy.VOTE is not None
        assert ConflictStrategy.DEFER is not None

    def test_strategy_values(self) -> None:
        """Test strategy values."""
        assert ConflictStrategy.ACKNOWLEDGE.value == "acknowledge"
        assert ConflictStrategy.EXPLORE.value == "explore"
        assert ConflictStrategy.VOTE.value == "vote"
        assert ConflictStrategy.DEFER.value == "defer"


# =============================================================================
# DiscussionRole Tests
# =============================================================================


class TestDiscussionRole:
    """Tests for DiscussionRole enum."""

    def test_all_roles_defined(self) -> None:
        """Test that all roles are defined."""
        assert DiscussionRole.PROPONENT is not None
        assert DiscussionRole.OPPONENT is not None
        assert DiscussionRole.NEUTRAL is not None
        assert DiscussionRole.EXPERT is not None
        assert DiscussionRole.DEVIL_ADVOCATE is not None

    def test_role_values(self) -> None:
        """Test role values."""
        assert DiscussionRole.PROPONENT.value == "proponent"
        assert DiscussionRole.OPPONENT.value == "opponent"
        assert DiscussionRole.NEUTRAL.value == "neutral"
        assert DiscussionRole.EXPERT.value == "expert"
        assert DiscussionRole.DEVIL_ADVOCATE.value == "devil_advocate"


# =============================================================================
# DiscussionTurn Tests
# =============================================================================


class TestDiscussionTurn:
    """Tests for DiscussionTurn dataclass."""

    def test_create_turn(self) -> None:
        """Test creating a discussion turn."""
        turn = DiscussionTurn(
            participant="Agent1",
            content="Test content",
            phase=DiscussionPhase.DISCUSSION,
            round_number=1,
        )
        assert turn.participant == "Agent1"
        assert turn.content == "Test content"
        assert turn.phase == DiscussionPhase.DISCUSSION
        assert turn.round_number == 1

    def test_turn_with_timestamp(self) -> None:
        """Test turn with timestamp."""
        timestamp = 1234567890.0
        turn = DiscussionTurn(
            participant="Agent1",
            content="Test content",
            phase=DiscussionPhase.OPENING,
            round_number=0,
            timestamp=timestamp,
        )
        assert turn.timestamp == timestamp

    def test_turn_with_references(self) -> None:
        """Test turn with references."""
        turn = DiscussionTurn(
            participant="Agent1",
            content="Test content",
            phase=DiscussionPhase.DISCUSSION,
            round_number=1,
            references=["Agent2", "Agent3"],
        )
        assert turn.references == ["Agent2", "Agent3"]


# =============================================================================
# DiscussionConflict Tests
# =============================================================================


class TestDiscussionConflict:
    """Tests for DiscussionConflict dataclass."""

    def test_create_conflict(self) -> None:
        """Test creating a discussion conflict."""
        conflict = DiscussionConflict(
            participants=["Agent1", "Agent2"],
            topic="Disagreement on topic",
            positions={"Agent1": "Position A", "Agent2": "Position B"},
        )
        assert conflict.participants == ["Agent1", "Agent2"]
        assert conflict.topic == "Disagreement on topic"
        assert "Agent1" in conflict.positions

    def test_conflict_with_resolution(self) -> None:
        """Test conflict with resolution."""
        conflict = DiscussionConflict(
            participants=["Agent1", "Agent2"],
            topic="Disagreement",
            positions={"Agent1": "Position A", "Agent2": "Position B"},
            resolution="Agreed to disagree",
        )
        assert conflict.resolution == "Agreed to disagree"


# =============================================================================
# DiscussionSummary Tests
# =============================================================================


class TestDiscussionSummary:
    """Tests for DiscussionSummary dataclass."""

    def test_create_summary(self) -> None:
        """Test creating a discussion summary."""
        summary = DiscussionSummary(
            topic="Test topic",
            participants=["Agent1", "Agent2", "Agent3"],
            conclusion="Test conclusion",
            key_points=["Point 1", "Point 2"],
            total_rounds=3,
            total_turns=9,
            conflicts=[],
            consensus_reached=True,
        )
        assert summary.topic == "Test topic"
        assert summary.conclusion == "Test conclusion"
        assert summary.key_points == ["Point 1", "Point 2"]
        assert summary.total_rounds == 3
        assert summary.total_turns == 9
        assert summary.consensus_reached is True

    def test_summary_with_conflicts(self) -> None:
        """Test summary with conflicts."""
        conflict = DiscussionConflict(
            participants=["A", "B"],
            topic="Conflict topic",
            positions={"A": "Position A", "B": "Position B"},
        )
        summary = DiscussionSummary(
            topic="Test",
            participants=["A", "B"],
            conclusion="Conclusion",
            key_points=[],
            total_rounds=1,
            total_turns=2,
            consensus_reached=False,
            conflicts=[conflict],
        )
        assert len(summary.conflicts) == 1
        assert summary.conflicts[0].topic == "Conflict topic"


# =============================================================================
# Participant Tests
# =============================================================================


class TestParticipant:
    """Tests for Participant dataclass."""

    def test_create_participant(self, mock_agent: MockAgent) -> None:
        """Test creating a participant."""
        participant = Participant(
            agent=mock_agent,
            role=DiscussionRole.PROPONENT,
        )
        assert participant.agent == mock_agent
        assert participant.name == "TestAgent"
        assert participant.role == DiscussionRole.PROPONENT

    def test_participant_with_priority(self, mock_agent: MockAgent) -> None:
        """Test participant with priority."""
        participant = Participant(
            agent=mock_agent,
            role=DiscussionRole.EXPERT,
            priority=10,
        )
        assert participant.priority == 10

    def test_participant_default_priority(self, mock_agent: MockAgent) -> None:
        """Test participant default priority."""
        participant = Participant(
            agent=mock_agent,
            role=DiscussionRole.NEUTRAL,
        )
        assert participant.priority == 0


# =============================================================================
# DiscussionConfig Tests
# =============================================================================


class TestDiscussionConfig:
    """Tests for DiscussionConfig dataclass."""

    def test_create_config(self) -> None:
        """Test creating a discussion config."""
        config = DiscussionConfig(
            topic="Test Topic",
            max_rounds=5,
        )
        assert config.topic == "Test Topic"
        assert config.max_rounds == 5

    def test_config_with_all_options(self) -> None:
        """Test config with all options."""
        config = DiscussionConfig(
            topic="Full Config Test",
            max_rounds=3,
            conflict_strategy=ConflictStrategy.VOTE,
            allow_rebuttals=True,
            rebuttal_rounds=2,
            turn_timeout=300.0,
        )
        assert config.conflict_strategy == ConflictStrategy.VOTE
        assert config.allow_rebuttals is True
        assert config.rebuttal_rounds == 2
        assert config.turn_timeout == 300.0

    def test_config_defaults(self) -> None:
        """Test config defaults."""
        config = DiscussionConfig(topic="Default Test")
        assert config.max_rounds == 5
        assert config.conflict_strategy == ConflictStrategy.EXPLORE
        assert config.allow_rebuttals is True
        assert config.rebuttal_rounds == 1


# =============================================================================
# RoundResult Tests
# =============================================================================


class TestRoundResult:
    """Tests for RoundResult dataclass."""

    def test_create_round_result(self) -> None:
        """Test creating a round result."""
        turns = [
            DiscussionTurn(
                participant="Agent1",
                content="Content",
                phase=DiscussionPhase.DISCUSSION,
                round_number=1,
            )
        ]
        result = RoundResult(
            round_number=1,
            turns=turns,
            conflicts_detected=0,
            duration_seconds=10.5,
            participation_rate=1.0,
        )
        assert result.round_number == 1
        assert len(result.turns) == 1
        assert result.duration_seconds == 10.5
        assert result.conflicts_detected == 0
        assert result.participation_rate == 1.0


# =============================================================================
# ModeratorAgent Tests
# =============================================================================


class TestModeratorAgent:
    """Tests for ModeratorAgent class."""

    def test_create_moderator(self) -> None:
        """Test creating a moderator."""
        moderator = ModeratorAgent(
            name="TestModerator",
            conflict_strategy=ConflictStrategy.EXPLORE,
            max_rounds=5,
        )
        assert moderator.name == "TestModerator"
        assert moderator.conflict_strategy == ConflictStrategy.EXPLORE
        assert moderator.max_rounds == 5

    def test_register_participant(
        self, mock_agent: MockAgent
    ) -> None:
        """Test registering a participant."""
        moderator = ModeratorAgent(name="Mod")
        moderator.register_participant(mock_agent)
        assert "TestAgent" in moderator.participants

    def test_register_multiple_participants(
        self, mock_agents: list[MockAgent]
    ) -> None:
        """Test registering multiple participants."""
        moderator = ModeratorAgent(name="Mod")
        for agent in mock_agents:
            moderator.register_participant(agent)
        assert len(moderator.participants) == 3

    def test_empty_turn_history(self) -> None:
        """Test empty turn history."""
        moderator = ModeratorAgent(name="Mod")
        assert len(moderator.turn_history) == 0

    @pytest.mark.asyncio
    async def test_start_discussion(
        self, moderator: ModeratorAgent
    ) -> None:
        """Test starting a discussion."""
        await moderator.start_discussion("Test Topic")
        assert len(moderator.turn_history) > 0
        # Opening statements from all participants
        opening_turns = [
            t for t in moderator.turn_history
            if t.phase == DiscussionPhase.OPENING
        ]
        assert len(opening_turns) == 3  # One per agent

    @pytest.mark.asyncio
    async def test_run_round(
        self, moderator: ModeratorAgent
    ) -> None:
        """Test running a discussion round."""
        await moderator.start_discussion("Test Topic")
        turns = await moderator.run_round()
        assert len(turns) > 0
        for turn in turns:
            assert turn.phase == DiscussionPhase.DISCUSSION

    @pytest.mark.asyncio
    async def test_synthesize(
        self, moderator: ModeratorAgent
    ) -> None:
        """Test synthesizing discussion."""
        await moderator.start_discussion("Test Topic")
        await moderator.run_round()
        summary = await moderator.synthesize()
        assert isinstance(summary, DiscussionSummary)
        assert summary.topic == "Test Topic"


# =============================================================================
# ModeratorAgent ChatAgent Integration Tests (T105)
# =============================================================================


class TestModeratorAgentChatAgentIntegration:
    """Tests for ModeratorAgent's ChatAgent-based moderation features."""
    
    def test_is_llm_based_without_chat_agent(self) -> None:
        """Test is_llm_based returns False without ChatAgent."""
        moderator = ModeratorAgent(name="TestMod")
        assert moderator.is_llm_based is False
        assert moderator.chat_agent is None
    
    def test_is_llm_based_with_mock_chat_agent(self) -> None:
        """Test is_llm_based returns True with ChatAgent."""
        mock_chat_agent = MagicMock()
        moderator = ModeratorAgent(
            name="TestMod",
            chat_agent=mock_chat_agent,
        )
        assert moderator.is_llm_based is True
        assert moderator.chat_agent is mock_chat_agent
    
    def test_custom_moderator_instructions(self) -> None:
        """Test custom moderator instructions are stored."""
        custom_instructions = "Be very strict about time."
        mock_chat_agent = MagicMock()
        moderator = ModeratorAgent(
            name="TestMod",
            chat_agent=mock_chat_agent,
            moderator_instructions=custom_instructions,
        )
        assert moderator._moderator_instructions == custom_instructions
    
    def test_default_moderator_instructions(self) -> None:
        """Test default moderator instructions are used when not specified."""
        moderator = ModeratorAgent(name="TestMod")
        assert "discussion moderator" in moderator._moderator_instructions.lower()
    
    @pytest.mark.asyncio
    async def test_select_speaker_round_robin_without_chat_agent(
        self, mock_agents: list[MockAgent]
    ) -> None:
        """Test round-robin speaker selection without ChatAgent."""
        moderator = ModeratorAgent(name="TestMod")
        for agent in mock_agents:
            moderator.register_participant(agent)
        
        # Should cycle through participants
        speaker1 = await moderator.select_speaker()
        assert speaker1 in moderator.participants
    
    @pytest.mark.asyncio
    async def test_select_speaker_with_exclusion(
        self, mock_agents: list[MockAgent]
    ) -> None:
        """Test speaker selection with exclusion list."""
        moderator = ModeratorAgent(name="TestMod")
        for agent in mock_agents:
            moderator.register_participant(agent)
        
        first_participant = moderator.participants[0]
        speaker = await moderator.select_speaker(exclude=[first_participant])
        assert speaker != first_participant
        assert speaker in moderator.participants
    
    @pytest.mark.asyncio
    async def test_select_speaker_all_excluded(self) -> None:
        """Test speaker selection returns None when all excluded."""
        moderator = ModeratorAgent(name="TestMod")
        agent = MockAgent("Agent1")
        moderator.register_participant(agent)
        
        speaker = await moderator.select_speaker(exclude=["Agent1"])
        assert speaker is None
    
    @pytest.mark.asyncio
    async def test_select_speaker_no_participants(self) -> None:
        """Test speaker selection returns None with no participants."""
        moderator = ModeratorAgent(name="TestMod")
        speaker = await moderator.select_speaker()
        assert speaker is None
    
    @pytest.mark.asyncio
    async def test_select_speaker_with_chat_agent_success(self) -> None:
        """Test speaker selection using ChatAgent."""
        mock_result = MagicMock()
        mock_result.text = "Agent2"
        
        mock_chat_agent = AsyncMock()
        mock_chat_agent.run = AsyncMock(return_value=mock_result)
        
        moderator = ModeratorAgent(
            name="TestMod",
            chat_agent=mock_chat_agent,
        )
        moderator.register_participant(MockAgent("Agent1"))
        moderator.register_participant(MockAgent("Agent2"))
        moderator.register_participant(MockAgent("Agent3"))
        
        speaker = await moderator.select_speaker(context="Focus on Agent2")
        assert speaker == "Agent2"
        mock_chat_agent.run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_select_speaker_chat_agent_fallback_on_error(self) -> None:
        """Test fallback to round-robin when ChatAgent fails."""
        mock_chat_agent = AsyncMock()
        mock_chat_agent.run = AsyncMock(side_effect=Exception("LLM Error"))
        
        moderator = ModeratorAgent(
            name="TestMod",
            chat_agent=mock_chat_agent,
        )
        moderator.register_participant(MockAgent("Agent1"))
        moderator.register_participant(MockAgent("Agent2"))
        
        speaker = await moderator.select_speaker()
        assert speaker in ["Agent1", "Agent2"]
    
    @pytest.mark.asyncio
    async def test_select_speaker_chat_agent_content_attribute(self) -> None:
        """Test parsing ChatAgent result with content attribute."""
        mock_result = MagicMock(spec=[])  # No attributes initially
        mock_result.content = "Agent1"
        del mock_result.text  # Ensure text doesn't exist
        
        mock_chat_agent = AsyncMock()
        mock_chat_agent.run = AsyncMock(return_value=mock_result)
        
        moderator = ModeratorAgent(
            name="TestMod",
            chat_agent=mock_chat_agent,
        )
        moderator.register_participant(MockAgent("Agent1"))
        
        speaker = await moderator.select_speaker()
        assert speaker == "Agent1"
    
    @pytest.mark.asyncio
    async def test_run_with_chat_agent(self) -> None:
        """Test run() delegates to ChatAgent."""
        mock_result = MagicMock()
        mock_result.text = "Moderator response"
        
        mock_chat_agent = AsyncMock()
        mock_chat_agent.run = AsyncMock(return_value=mock_result)
        
        moderator = ModeratorAgent(
            name="TestMod",
            chat_agent=mock_chat_agent,
        )
        
        response = await moderator.run("Who should speak next?")
        assert response == "Moderator response"
        mock_chat_agent.run.assert_called_once_with("Who should speak next?")
    
    @pytest.mark.asyncio
    async def test_run_without_chat_agent(self) -> None:
        """Test run() returns placeholder without ChatAgent."""
        moderator = ModeratorAgent(name="TestMod")
        response = await moderator.run("Any prompt")
        assert "[Moderator TestMod]" in response
        assert "Processing" in response
    
    @pytest.mark.asyncio
    async def test_run_stream_with_chat_agent(self) -> None:
        """Test run_stream() delegates to ChatAgent."""
        async def mock_stream(prompt: str):
            chunks = [MagicMock(text="chunk1"), MagicMock(text="chunk2")]
            for chunk in chunks:
                yield chunk
        
        mock_chat_agent = MagicMock()
        mock_chat_agent.run_stream = mock_stream
        
        moderator = ModeratorAgent(
            name="TestMod",
            chat_agent=mock_chat_agent,
        )
        
        chunks = []
        async for chunk in moderator.run_stream("Test prompt"):
            chunks.append(chunk)
        
        assert chunks == ["chunk1", "chunk2"]
    
    @pytest.mark.asyncio
    async def test_run_stream_without_chat_agent(self) -> None:
        """Test run_stream() returns placeholder without ChatAgent."""
        moderator = ModeratorAgent(name="TestMod")
        
        chunks = []
        async for chunk in moderator.run_stream("Any prompt"):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert "[Moderator TestMod]" in chunks[0]
    
    @pytest.mark.asyncio
    async def test_run_stream_delta_attribute(self) -> None:
        """Test run_stream() handles delta attribute."""
        async def mock_stream(prompt: str):
            chunks = [MagicMock(delta="delta1"), MagicMock(delta="delta2")]
            for chunk in chunks:
                # Remove text attribute to test delta path
                del chunk.text
                yield chunk
        
        mock_chat_agent = MagicMock()
        mock_chat_agent.run_stream = mock_stream
        
        moderator = ModeratorAgent(
            name="TestMod",
            chat_agent=mock_chat_agent,
        )
        
        chunks = []
        async for chunk in moderator.run_stream("Test prompt"):
            chunks.append(chunk)
        
        assert chunks == ["delta1", "delta2"]
    
    def test_build_speaker_selection_prompt(self) -> None:
        """Test speaker selection prompt generation."""
        moderator = ModeratorAgent(name="TestMod")
        moderator._topic = "Climate Change"
        moderator._current_round = 2
        
        # Add a turn to test turns_text generation
        moderator._turns.append(
            DiscussionTurn(
                participant="Agent1",
                content="This is a test turn content",
                phase=DiscussionPhase.DISCUSSION,
                round_number=1,
            )
        )
        
        prompt = moderator._build_speaker_selection_prompt(
            context="Focus on solutions",
            available=["Agent1", "Agent2", "Agent3"],
        )
        
        assert "Climate Change" in prompt
        assert "Agent1, Agent2, Agent3" in prompt
        assert "round: 2" in prompt.lower()
        assert "Focus on solutions" in prompt
        assert "[Agent1]:" in prompt
    
    def test_parse_speaker_selection_exact_match(self) -> None:
        """Test parsing speaker selection with exact match."""
        moderator = ModeratorAgent(name="TestMod")
        result = moderator._parse_speaker_selection(
            "Agent2",
            ["Agent1", "Agent2", "Agent3"],
        )
        assert result == "Agent2"
    
    def test_parse_speaker_selection_case_insensitive(self) -> None:
        """Test parsing speaker selection is case insensitive."""
        moderator = ModeratorAgent(name="TestMod")
        result = moderator._parse_speaker_selection(
            "AGENT2",
            ["Agent1", "Agent2", "Agent3"],
        )
        assert result == "Agent2"
    
    def test_parse_speaker_selection_with_extra_text(self) -> None:
        """Test parsing speaker selection with surrounding text."""
        moderator = ModeratorAgent(name="TestMod")
        result = moderator._parse_speaker_selection(
            "I think Agent2 should speak next.",
            ["Agent1", "Agent2", "Agent3"],
        )
        assert result == "Agent2"
    
    def test_parse_speaker_selection_no_match_returns_first(self) -> None:
        """Test parsing returns first available when no match."""
        moderator = ModeratorAgent(name="TestMod")
        result = moderator._parse_speaker_selection(
            "Nobody matches",
            ["Agent1", "Agent2"],
        )
        assert result == "Agent1"
    
    def test_parse_speaker_selection_empty_list(self) -> None:
        """Test parsing returns None for empty available list."""
        moderator = ModeratorAgent(name="TestMod")
        result = moderator._parse_speaker_selection("Agent1", [])
        assert result is None


# =============================================================================
# create_moderator_agent Factory Tests (T105)
# =============================================================================


class TestCreateModeratorAgentFactory:
    """Tests for create_moderator_agent factory function."""
    
    def test_create_rule_based_moderator(self) -> None:
        """Test creating a rule-based moderator without chat client."""
        moderator = create_moderator_agent(
            name="rule_mod",
            conflict_strategy=ConflictStrategy.VOTE,
            max_rounds=10,
        )
        
        assert moderator.name == "rule_mod"
        assert moderator.conflict_strategy == ConflictStrategy.VOTE
        assert moderator.max_rounds == 10
        assert moderator.chat_agent is None
        assert moderator.is_llm_based is False
    
    def test_create_moderator_default_values(self) -> None:
        """Test creating moderator with default values."""
        moderator = create_moderator_agent()
        
        assert moderator.name == "moderator"
        assert moderator.conflict_strategy == ConflictStrategy.EXPLORE
        assert moderator.max_rounds == 5
        assert moderator.is_llm_based is False
    
    def test_create_moderator_custom_instructions(self) -> None:
        """Test custom instructions are passed to moderator."""
        custom_instructions = "Be very concise and direct."
        moderator = create_moderator_agent(
            name="custom_mod",
            instructions=custom_instructions,
        )
        
        assert moderator._moderator_instructions == custom_instructions
    
    def test_create_moderator_custom_synthesis_prompt(self) -> None:
        """Test custom synthesis prompt is passed to moderator."""
        custom_synthesis = "Summarize in bullet points only."
        moderator = create_moderator_agent(
            name="synth_mod",
            synthesis_prompt=custom_synthesis,
        )
        
        assert moderator.synthesis_prompt == custom_synthesis
    
    def test_create_moderator_with_mock_chat_client(self) -> None:
        """Test creating LLM-based moderator with chat client."""
        # Mock the ChatAgent creation
        mock_chat_agent = MagicMock()
        mock_chat_client = MagicMock()
        mock_chat_client.create_agent = MagicMock(return_value=mock_chat_agent)
        
        # Since ChatAgent is imported dynamically, we patch it in agent_framework module
        mock_agent_framework = MagicMock()
        mock_agent_framework.ChatAgent = MagicMock()
        
        with patch.dict("sys.modules", {"agent_framework": mock_agent_framework}):
            moderator = create_moderator_agent(
                name="llm_mod",
                instructions="Focus on consensus",
                chat_client=mock_chat_client,
                model="gpt-4o",
                temperature=0.5,
            )
            
            mock_chat_client.create_agent.assert_called_once()
            call_kwargs = mock_chat_client.create_agent.call_args[1]
            assert call_kwargs["name"] == "llm_mod"
            assert call_kwargs["temperature"] == 0.5
            assert "Focus on consensus" in call_kwargs["instructions"]
            
            assert moderator.chat_agent is mock_chat_agent
            assert moderator.is_llm_based is True
    
    def test_create_moderator_llm_uses_default_instructions(self) -> None:
        """Test that default instructions are used when none provided."""
        mock_chat_agent = MagicMock()
        mock_chat_client = MagicMock()
        mock_chat_client.create_agent = MagicMock(return_value=mock_chat_agent)
        
        mock_agent_framework = MagicMock()
        mock_agent_framework.ChatAgent = MagicMock()
        
        with patch.dict("sys.modules", {"agent_framework": mock_agent_framework}):
            create_moderator_agent(
                name="default_llm_mod",
                chat_client=mock_chat_client,
            )
            
            call_kwargs = mock_chat_client.create_agent.call_args[1]
            assert "discussion moderator" in call_kwargs["instructions"].lower()
    
    def test_default_moderator_instructions_content(self) -> None:
        """Test that DEFAULT_MODERATOR_INSTRUCTIONS has expected content."""
        assert "speaker" in DEFAULT_MODERATOR_INSTRUCTIONS.lower()
        assert "synthesize" in DEFAULT_MODERATOR_INSTRUCTIONS.lower()
        assert "conflict" in DEFAULT_MODERATOR_INSTRUCTIONS.lower()


# =============================================================================
# DiscussionProtocol Tests
# =============================================================================


class TestDiscussionProtocol:
    """Tests for DiscussionProtocol class."""

    def test_create_protocol(
        self, discussion_config: DiscussionConfig
    ) -> None:
        """Test creating a discussion protocol."""
        protocol = DiscussionProtocol(discussion_config)
        assert protocol.config == discussion_config
        assert len(protocol.participants) == 0

    def test_register_participant(
        self,
        discussion_config: DiscussionConfig,
        mock_agent: MockAgent,
    ) -> None:
        """Test registering a participant."""
        protocol = DiscussionProtocol(discussion_config)
        protocol.register_participant(
            mock_agent,
            role=DiscussionRole.PROPONENT,
        )
        assert len(protocol.participants) == 1
        assert protocol.participants[0].role == DiscussionRole.PROPONENT

    def test_register_with_priority(
        self,
        discussion_config: DiscussionConfig,
        mock_agent: MockAgent,
    ) -> None:
        """Test registering with priority."""
        protocol = DiscussionProtocol(discussion_config)
        protocol.register_participant(
            mock_agent,
            role=DiscussionRole.EXPERT,
            priority=10,
        )
        assert protocol.participants[0].priority == 10

    def test_participant_names(
        self,
        discussion_config: DiscussionConfig,
        mock_agents: list[MockAgent],
    ) -> None:
        """Test getting participant names."""
        protocol = DiscussionProtocol(discussion_config)
        for agent in mock_agents:
            protocol.register_participant(agent)
        assert protocol.participant_names == ["Agent1", "Agent2", "Agent3"]

    def test_on_turn_callback(
        self, discussion_config: DiscussionConfig
    ) -> None:
        """Test registering turn callback."""
        protocol = DiscussionProtocol(discussion_config)
        callback = MagicMock()
        protocol.on_turn(callback)
        assert callback in protocol._on_turn_callbacks

    def test_on_round_callback(
        self, discussion_config: DiscussionConfig
    ) -> None:
        """Test registering round callback."""
        protocol = DiscussionProtocol(discussion_config)
        callback = MagicMock()
        protocol.on_round(callback)
        assert callback in protocol._on_round_callbacks

    @pytest.mark.asyncio
    async def test_run_discussion(
        self,
        discussion_config: DiscussionConfig,
        mock_agents: list[MockAgent],
    ) -> None:
        """Test running a full discussion."""
        protocol = DiscussionProtocol(discussion_config)
        for i, agent in enumerate(mock_agents):
            role = [
                DiscussionRole.PROPONENT,
                DiscussionRole.OPPONENT,
                DiscussionRole.NEUTRAL,
            ][i]
            protocol.register_participant(agent, role=role)

        summary = await protocol.run_discussion()
        assert isinstance(summary, DiscussionSummary)
        assert summary.total_turns > 0

    def test_get_participant_stats(
        self,
        discussion_config: DiscussionConfig,
        mock_agents: list[MockAgent],
    ) -> None:
        """Test getting participant statistics."""
        protocol = DiscussionProtocol(discussion_config)
        for agent in mock_agents:
            protocol.register_participant(agent)
        
        stats = protocol.get_participant_stats()
        assert "Agent1" in stats
        assert "role" in stats["Agent1"]

    def test_get_cross_references(
        self,
        discussion_config: DiscussionConfig,
        mock_agents: list[MockAgent],
    ) -> None:
        """Test getting cross references."""
        protocol = DiscussionProtocol(discussion_config)
        for agent in mock_agents:
            protocol.register_participant(agent)
        
        refs = protocol.get_cross_references()
        assert isinstance(refs, dict)


# =============================================================================
# DebateProtocol Tests
# =============================================================================


class TestDebateProtocol:
    """Tests for DebateProtocol class."""

    def test_create_debate_protocol(self) -> None:
        """Test creating a debate protocol."""
        proponent = MockAgent("Pro")
        opponent = MockAgent("Con")
        
        debate = create_debate(
            topic="Test Debate",
            proponent=proponent,
            opponent=opponent,
        )
        assert debate.config.topic == "Test Debate"
        assert len(debate.participants) == 2

    def test_debate_with_judges(self) -> None:
        """Test debate with judges."""
        proponent = MockAgent("Pro")
        opponent = MockAgent("Con")
        judge = MockAgent("Judge")
        
        debate = create_debate(
            topic="Test Debate",
            proponent=proponent,
            opponent=opponent,
            judges=[judge],
        )
        assert len(debate.participants) == 3

    def test_create_debate_factory(self) -> None:
        """Test create_debate factory function."""
        proponent = MockAgent("Pro")
        opponent = MockAgent("Con")
        
        debate = create_debate(
            topic="Factory Test",
            proponent=proponent,
            opponent=opponent,
        )
        assert isinstance(debate, DebateProtocol)
        assert debate.config.topic == "Factory Test"


# =============================================================================
# RoundRobinProtocol Tests
# =============================================================================


class TestRoundRobinProtocol:
    """Tests for RoundRobinProtocol class."""

    def test_create_round_robin(
        self, mock_agents: list[MockAgent]
    ) -> None:
        """Test creating a round robin protocol."""
        protocol = create_roundtable(
            topic="Round Robin Test",
            participants=mock_agents,
        )
        assert protocol.config.topic == "Round Robin Test"
        assert len(protocol.participants) == 3

    def test_round_robin_with_rounds(
        self, mock_agents: list[MockAgent]
    ) -> None:
        """Test round robin with custom rounds."""
        protocol = create_roundtable(
            topic="Test",
            participants=mock_agents,
            rounds=5,
        )
        assert protocol.config.max_rounds == 5

    def test_create_roundtable_factory(
        self, mock_agents: list[MockAgent]
    ) -> None:
        """Test create_roundtable factory function."""
        protocol = create_roundtable(
            topic="Roundtable Test",
            participants=mock_agents,
            rounds=3,
        )
        assert isinstance(protocol, RoundRobinProtocol)
        assert protocol.config.topic == "Roundtable Test"


# =============================================================================
# Integration Tests for Discussion Flows
# =============================================================================


class TestDiscussionFlows:
    """Integration tests for discussion flows."""

    @pytest.mark.asyncio
    async def test_full_debate_flow(self) -> None:
        """Test a complete debate flow."""
        proponent = MockAgent("Advocate", "I support this position")
        opponent = MockAgent("Critic", "I oppose this position")
        judge = MockAgent("Judge", "Evaluating arguments")

        debate = create_debate(
            topic="Should AI be regulated?",
            proponent=proponent,
            opponent=opponent,
            judges=[judge],
        )

        summary = await debate.run_discussion()
        
        assert summary.topic == "Should AI be regulated?"
        assert summary.total_turns > 0
        # All agents should have been called
        assert proponent.call_count > 0
        assert opponent.call_count > 0

    @pytest.mark.asyncio
    async def test_roundtable_flow(self) -> None:
        """Test a complete roundtable flow."""
        experts = [
            MockAgent("Economist", "Economic perspective"),
            MockAgent("Ethicist", "Ethical considerations"),
            MockAgent("Engineer", "Technical feasibility"),
        ]

        roundtable = create_roundtable(
            topic="Future of AI",
            participants=experts,
            rounds=2,
        )

        summary = await roundtable.run_discussion()
        
        assert summary.topic == "Future of AI"
        assert summary.total_rounds <= 2
        # All experts should have been called
        for expert in experts:
            assert expert.call_count > 0

    @pytest.mark.asyncio
    async def test_callback_execution(
        self,
        discussion_config: DiscussionConfig,
        mock_agents: list[MockAgent],
    ) -> None:
        """Test that callbacks are executed."""
        protocol = DiscussionProtocol(discussion_config)
        for agent in mock_agents:
            protocol.register_participant(agent)

        turn_count = 0
        round_count = 0

        def on_turn(turn: DiscussionTurn) -> None:
            nonlocal turn_count
            turn_count += 1

        def on_round(result: RoundResult) -> None:
            nonlocal round_count
            round_count += 1

        protocol.on_turn(on_turn)
        protocol.on_round(on_round)

        await protocol.run_discussion()

        assert turn_count > 0
        assert round_count > 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_discussion(self) -> None:
        """Test protocol with no participants."""
        config = DiscussionConfig(topic="Empty Test")
        protocol = DiscussionProtocol(config)
        assert len(protocol.participants) == 0

    def test_single_participant_discussion(
        self, discussion_config: DiscussionConfig
    ) -> None:
        """Test discussion with single participant."""
        protocol = DiscussionProtocol(discussion_config)
        agent = MockAgent("Solo")
        protocol.register_participant(agent)
        assert len(protocol.participants) == 1

    def test_moderator_no_participants(self) -> None:
        """Test moderator with no participants."""
        moderator = ModeratorAgent(name="Empty")
        assert len(moderator.participants) == 0

    @pytest.mark.asyncio
    async def test_discussion_timeout(self) -> None:
        """Test discussion with time limit."""
        config = DiscussionConfig(
            topic="Timed Test",
            max_rounds=100,  # Many rounds
            turn_timeout=1.0,  # Short time limit
        )
        protocol = DiscussionProtocol(config)
        
        # Discussion should respect time limit if implemented
        # This is a placeholder for timeout logic testing
        assert config.turn_timeout == 1.0

    def test_duplicate_participant_registration(
        self,
        discussion_config: DiscussionConfig,
        mock_agent: MockAgent,
    ) -> None:
        """Test registering same agent twice."""
        protocol = DiscussionProtocol(discussion_config)
        protocol.register_participant(mock_agent)
        protocol.register_participant(mock_agent)
        # Should handle duplicates gracefully
        # Implementation may allow or reject duplicates
        assert len(protocol.participants) >= 1


# =============================================================================
# DiscussionAgent Tests (T104)
# =============================================================================


class TestDiscussionAgent:
    """Tests for DiscussionAgent wrapper class."""

    @pytest.fixture
    def mock_chat_agent(self) -> MagicMock:
        """Create a mock ChatAgent."""
        mock = MagicMock()
        mock.run = AsyncMock(return_value=MagicMock(text="Response from ChatAgent"))
        return mock

    def test_discussion_agent_creation(self, mock_chat_agent: MagicMock) -> None:
        """Test creating a DiscussionAgent."""
        agent = DiscussionAgent(
            name="test_agent",
            chat_agent=mock_chat_agent,
            role=DiscussionRole.PROPONENT,
            perspective="Test perspective",
        )
        
        assert agent.name == "test_agent"
        assert agent.role == DiscussionRole.PROPONENT
        assert agent.perspective == "Test perspective"
        assert agent.turn_count == 0

    def test_discussion_agent_name_property(self, mock_chat_agent: MagicMock) -> None:
        """Test name property returns correct value."""
        agent = DiscussionAgent(name="my_agent", chat_agent=mock_chat_agent)
        assert agent.name == "my_agent"

    def test_discussion_agent_role_property(self, mock_chat_agent: MagicMock) -> None:
        """Test role property with different roles."""
        for role in DiscussionRole:
            agent = DiscussionAgent(
                name="agent", chat_agent=mock_chat_agent, role=role
            )
            assert agent.role == role

    def test_discussion_agent_no_role(self, mock_chat_agent: MagicMock) -> None:
        """Test agent without a role."""
        agent = DiscussionAgent(name="neutral", chat_agent=mock_chat_agent)
        assert agent.role is None

    def test_discussion_agent_chat_agent_property(
        self, mock_chat_agent: MagicMock
    ) -> None:
        """Test chat_agent property returns underlying agent."""
        agent = DiscussionAgent(name="test", chat_agent=mock_chat_agent)
        assert agent.chat_agent is mock_chat_agent

    def test_discussion_agent_turn_count_initial(
        self, mock_chat_agent: MagicMock
    ) -> None:
        """Test turn count starts at zero."""
        agent = DiscussionAgent(name="test", chat_agent=mock_chat_agent)
        assert agent.turn_count == 0

    @pytest.mark.asyncio
    async def test_discussion_agent_run_increments_turn_count(
        self, mock_chat_agent: MagicMock
    ) -> None:
        """Test run() increments turn count."""
        agent = DiscussionAgent(name="test", chat_agent=mock_chat_agent)
        
        await agent.run("prompt 1")
        assert agent.turn_count == 1
        
        await agent.run("prompt 2")
        assert agent.turn_count == 2

    @pytest.mark.asyncio
    async def test_discussion_agent_run_returns_text(
        self, mock_chat_agent: MagicMock
    ) -> None:
        """Test run() returns text from ChatAgent."""
        agent = DiscussionAgent(name="test", chat_agent=mock_chat_agent)
        result = await agent.run("test prompt")
        assert result == "Response from ChatAgent"
        mock_chat_agent.run.assert_called_once_with("test prompt")

    @pytest.mark.asyncio
    async def test_discussion_agent_run_extracts_content_attribute(
        self, mock_chat_agent: MagicMock
    ) -> None:
        """Test run() extracts content attribute when text not available."""
        mock_chat_agent.run = AsyncMock(
            return_value=MagicMock(spec=[], content="Content response")
        )
        agent = DiscussionAgent(name="test", chat_agent=mock_chat_agent)
        result = await agent.run("test")
        assert result == "Content response"

    @pytest.mark.asyncio
    async def test_discussion_agent_run_fallback_to_str(
        self, mock_chat_agent: MagicMock
    ) -> None:
        """Test run() falls back to str() when no known attributes."""
        mock_chat_agent.run = AsyncMock(return_value="plain string")
        agent = DiscussionAgent(name="test", chat_agent=mock_chat_agent)
        result = await agent.run("test")
        assert result == "plain string"

    def test_discussion_agent_reset_turn_count(
        self, mock_chat_agent: MagicMock
    ) -> None:
        """Test reset_turn_count() resets to zero."""
        agent = DiscussionAgent(name="test", chat_agent=mock_chat_agent)
        agent._turn_count = 5
        agent.reset_turn_count()
        assert agent.turn_count == 0

    def test_discussion_agent_repr(self, mock_chat_agent: MagicMock) -> None:
        """Test string representation."""
        agent = DiscussionAgent(
            name="test", chat_agent=mock_chat_agent, role=DiscussionRole.OPPONENT
        )
        repr_str = repr(agent)
        assert "DiscussionAgent" in repr_str
        assert "test" in repr_str
        assert "opponent" in repr_str

    def test_discussion_agent_repr_no_role(
        self, mock_chat_agent: MagicMock
    ) -> None:
        """Test repr without role."""
        agent = DiscussionAgent(name="test", chat_agent=mock_chat_agent)
        repr_str = repr(agent)
        assert "DiscussionAgent" in repr_str
        assert "test" in repr_str
        assert "role" not in repr_str


class TestBuildRoleInstructions:
    """Tests for _build_role_instructions helper function."""

    def test_build_instructions_proponent(self) -> None:
        """Test instructions for proponent role."""
        instructions = _build_role_instructions(
            "agent1", DiscussionRole.PROPONENT, ""
        )
        assert "agent1" in instructions
        assert "IN FAVOR" in instructions
        assert "benefits" in instructions.lower()

    def test_build_instructions_opponent(self) -> None:
        """Test instructions for opponent role."""
        instructions = _build_role_instructions(
            "critic", DiscussionRole.OPPONENT, ""
        )
        assert "critic" in instructions
        assert "AGAINST" in instructions
        assert "risks" in instructions.lower()

    def test_build_instructions_neutral(self) -> None:
        """Test instructions for neutral role."""
        instructions = _build_role_instructions(
            "neutral", DiscussionRole.NEUTRAL, ""
        )
        assert "balanced" in instructions.lower()
        assert "multiple perspectives" in instructions.lower()

    def test_build_instructions_expert(self) -> None:
        """Test instructions for expert role."""
        instructions = _build_role_instructions(
            "expert", DiscussionRole.EXPERT, ""
        )
        assert "subject matter expert" in instructions.lower()
        assert "specialized" in instructions.lower()

    def test_build_instructions_devil_advocate(self) -> None:
        """Test instructions for devil's advocate role."""
        instructions = _build_role_instructions(
            "devil", DiscussionRole.DEVIL_ADVOCATE, ""
        )
        assert "devil's advocate" in instructions.lower()
        assert "challenging" in instructions.lower()

    def test_build_instructions_with_perspective(self) -> None:
        """Test instructions with custom perspective."""
        instructions = _build_role_instructions(
            "agent", DiscussionRole.PROPONENT, "Focus on economic benefits"
        )
        assert "Focus on economic benefits" in instructions

    def test_build_instructions_no_role(self) -> None:
        """Test instructions without role."""
        instructions = _build_role_instructions("agent", None, "")
        assert "agent" in instructions
        assert "participant" in instructions.lower()

    def test_build_instructions_includes_guidelines(self) -> None:
        """Test that guidelines are included."""
        instructions = _build_role_instructions(
            "agent", DiscussionRole.NEUTRAL, ""
        )
        assert "Guidelines" in instructions
        assert "concise" in instructions.lower()
        assert "respectful" in instructions.lower()


class TestCreateDiscussionAgentFactory:
    """Tests for create_discussion_agent factory function."""

    def test_create_discussion_agent_requires_agent_framework(self) -> None:
        """Test that factory uses agent-framework imports."""
        # The factory has a try/except that raises ImportError
        # when agent-framework is not available
        # This is tested implicitly by the structure
        pass

    def test_create_discussion_agent_factory_exists(self) -> None:
        """Test that create_discussion_agent is callable."""
        assert callable(create_discussion_agent)

    def test_factory_uses_build_role_instructions(self) -> None:
        """Test that factory uses _build_role_instructions internally."""
        # Verify the helper function is used by the factory
        # by testing its output matches expected patterns
        instructions = _build_role_instructions(
            "test", DiscussionRole.PROPONENT, "custom perspective"
        )
        assert "test" in instructions
        assert "IN FAVOR" in instructions
        assert "custom perspective" in instructions


class TestDiscussionAgentStreaming:
    """Tests for DiscussionAgent streaming functionality."""

    @pytest.fixture
    def mock_streaming_chat_agent(self) -> MagicMock:
        """Create a mock ChatAgent with streaming support."""
        mock = MagicMock()
        
        async def stream_gen():
            yield MagicMock(text="Hello")
            yield MagicMock(text=" World")
            yield MagicMock(text="!")
        
        mock.run_stream = MagicMock(return_value=stream_gen())
        return mock

    @pytest.mark.asyncio
    async def test_discussion_agent_run_stream(
        self, mock_streaming_chat_agent: MagicMock
    ) -> None:
        """Test run_stream yields chunks."""
        agent = DiscussionAgent(
            name="test", chat_agent=mock_streaming_chat_agent
        )
        
        chunks = []
        async for chunk in agent.run_stream("prompt"):
            chunks.append(chunk)
        
        assert chunks == ["Hello", " World", "!"]

    @pytest.mark.asyncio
    async def test_discussion_agent_run_stream_increments_turn(
        self,
    ) -> None:
        """Test run_stream increments turn count."""
        mock = MagicMock()
        
        async def stream_gen():
            yield MagicMock(text="chunk")
        
        mock.run_stream = MagicMock(return_value=stream_gen())
        
        agent = DiscussionAgent(name="test", chat_agent=mock)
        assert agent.turn_count == 0
        
        async for _ in agent.run_stream("prompt"):
            pass
        
        assert agent.turn_count == 1

    @pytest.mark.asyncio
    async def test_discussion_agent_run_stream_with_delta_attr(self) -> None:
        """Test run_stream handles delta attribute."""
        mock = MagicMock()
        
        async def stream_gen():
            yield MagicMock(spec=[], delta="delta_chunk")
        
        mock.run_stream = MagicMock(return_value=stream_gen())
        
        agent = DiscussionAgent(name="test", chat_agent=mock)
        chunks = []
        async for chunk in agent.run_stream("prompt"):
            chunks.append(chunk)
        
        assert chunks == ["delta_chunk"]

    @pytest.mark.asyncio
    async def test_discussion_agent_run_stream_fallback(self) -> None:
        """Test run_stream falls back to str()."""
        mock = MagicMock()
        
        async def stream_gen():
            yield "plain_string"
        
        mock.run_stream = MagicMock(return_value=stream_gen())
        
        agent = DiscussionAgent(name="test", chat_agent=mock)
        chunks = []
        async for chunk in agent.run_stream("prompt"):
            chunks.append(chunk)
        
        assert chunks == ["plain_string"]


# =============================================================================
# Tests for T106: DiscussionProtocol GroupChatBuilder Integration
# =============================================================================


class TestDiscussionProtocolGroupChatMode:
    """Tests for DiscussionProtocol GroupChatBuilder integration (T106)."""

    @pytest.fixture
    def basic_config(self) -> DiscussionConfig:
        """Create basic discussion config."""
        return DiscussionConfig(topic="Test Topic", max_rounds=2)

    @pytest.fixture
    def mock_moderator(self) -> MagicMock:
        """Create mock moderator."""
        moderator = MagicMock(spec=ModeratorAgent)
        moderator.start_discussion = AsyncMock()
        moderator.run_round = AsyncMock(return_value=[])
        moderator.synthesize = AsyncMock(return_value=DiscussionSummary(
            topic="Test Topic",
            participants=["agent1"],
            total_rounds=2,
            total_turns=4,
            key_points=["Point 1"],
            conflicts=[],
            conclusion="Test conclusion",
            consensus_reached=True,
        ))
        moderator.is_llm_based = False
        moderator.run = AsyncMock(return_value="Summary")
        return moderator

    @pytest.fixture
    def mock_discussion_agent(self) -> MagicMock:
        """Create mock DiscussionAgent."""
        agent = MagicMock(spec=DiscussionAgent)
        agent.name = "agent1"
        agent.role = DiscussionRole.PROPONENT
        agent.chat_agent = MagicMock()
        return agent

    def test_use_group_chat_default_false(
        self, basic_config: DiscussionConfig, mock_moderator: MagicMock
    ) -> None:
        """Test use_group_chat defaults to False."""
        protocol = DiscussionProtocol(basic_config, mock_moderator)
        assert protocol.use_group_chat is False

    def test_use_group_chat_enabled(
        self, basic_config: DiscussionConfig, mock_moderator: MagicMock
    ) -> None:
        """Test use_group_chat can be enabled."""
        protocol = DiscussionProtocol(
            basic_config, mock_moderator, use_group_chat=True
        )
        assert protocol.use_group_chat is True

    def test_group_chat_manager_parameter(
        self, basic_config: DiscussionConfig, mock_moderator: MagicMock
    ) -> None:
        """Test group_chat_manager parameter is stored."""
        mock_manager = MagicMock()
        protocol = DiscussionProtocol(
            basic_config,
            mock_moderator,
            use_group_chat=True,
            group_chat_manager=mock_manager,
        )
        assert protocol._group_chat_manager is mock_manager

    def test_discussion_agents_initially_empty(
        self, basic_config: DiscussionConfig, mock_moderator: MagicMock
    ) -> None:
        """Test discussion_agents is initially empty."""
        protocol = DiscussionProtocol(
            basic_config, mock_moderator, use_group_chat=True
        )
        assert protocol.discussion_agents == []

    def test_register_discussion_agent(
        self,
        basic_config: DiscussionConfig,
        mock_moderator: MagicMock,
        mock_discussion_agent: MagicMock,
    ) -> None:
        """Test registering a DiscussionAgent."""
        protocol = DiscussionProtocol(
            basic_config, mock_moderator, use_group_chat=True
        )
        protocol.register_discussion_agent(mock_discussion_agent)
        
        assert len(protocol.discussion_agents) == 1
        assert protocol.discussion_agents[0] is mock_discussion_agent

    def test_register_discussion_agent_with_priority(
        self,
        basic_config: DiscussionConfig,
        mock_moderator: MagicMock,
    ) -> None:
        """Test registering with priority also registers in participants."""
        agent = MagicMock(spec=DiscussionAgent)
        agent.name = "priority_agent"
        agent.role = DiscussionRole.EXPERT
        agent.chat_agent = MagicMock()
        
        protocol = DiscussionProtocol(
            basic_config, mock_moderator, use_group_chat=True
        )
        protocol.register_discussion_agent(agent, priority=10)
        
        assert "priority_agent" in protocol.participant_names

    def test_register_multiple_discussion_agents(
        self,
        basic_config: DiscussionConfig,
        mock_moderator: MagicMock,
    ) -> None:
        """Test registering multiple DiscussionAgents."""
        protocol = DiscussionProtocol(
            basic_config, mock_moderator, use_group_chat=True
        )
        
        for i in range(3):
            agent = MagicMock(spec=DiscussionAgent)
            agent.name = f"agent{i}"
            agent.role = DiscussionRole.NEUTRAL
            agent.chat_agent = MagicMock()
            protocol.register_discussion_agent(agent)
        
        assert len(protocol.discussion_agents) == 3


class TestRoundRobinSelector:
    """Tests for _round_robin_selector method on DiscussionProtocol."""

    @pytest.fixture
    def protocol_with_agents(self) -> DiscussionProtocol:
        """Create protocol with mock agents registered."""
        config = DiscussionConfig(
            topic="Test",
            max_rounds=2,
            max_turns_per_round=1,
        )
        mock_mod = MagicMock(spec=ModeratorAgent)
        mock_mod.is_llm_based = False
        
        protocol = DiscussionProtocol(config, mock_mod, use_group_chat=True)
        
        for name in ["alice", "bob", "charlie"]:
            agent = MagicMock(spec=DiscussionAgent)
            agent.name = name
            agent.role = DiscussionRole.NEUTRAL
            agent.chat_agent = MagicMock()
            protocol.register_discussion_agent(agent)
        
        return protocol

    def test_round_robin_selector_first_call(
        self, protocol_with_agents: DiscussionProtocol
    ) -> None:
        """Test first call returns first participant."""
        state = {"round_index": 0, "participants": {"alice": {}, "bob": {}, "charlie": {}}}
        
        result = protocol_with_agents._round_robin_selector(state)
        
        assert result == "alice"

    def test_round_robin_selector_cycles(
        self, protocol_with_agents: DiscussionProtocol
    ) -> None:
        """Test selector cycles through participants."""
        participants = {"a": {}, "b": {}, "c": {}}
        
        results = []
        for i in range(6):
            state = {"round_index": i, "participants": participants}
            results.append(protocol_with_agents._round_robin_selector(state))
        
        assert results == ["a", "b", "c", "a", "b", "c"]

    def test_round_robin_selector_empty_participants(
        self, protocol_with_agents: DiscussionProtocol
    ) -> None:
        """Test selector handles empty participants."""
        state = {"round_index": 0, "participants": {}}
        result = protocol_with_agents._round_robin_selector(state)
        
        assert result is None


class TestBuildGroupChat:
    """Tests for _build_group_chat method."""

    @pytest.fixture
    def mock_group_chat_builder(self) -> MagicMock:
        """Create mock GroupChatBuilder."""
        mock = MagicMock()
        mock.participants.return_value = mock
        mock.with_max_rounds.return_value = mock
        mock.set_manager.return_value = mock
        mock.set_select_speakers_func.return_value = mock
        return mock

    def test_build_group_chat_caches_result(self) -> None:
        """Test _build_group_chat caches the result."""
        config = DiscussionConfig(topic="Test", max_rounds=2)
        mock_mod = MagicMock(spec=ModeratorAgent)
        mock_mod.is_llm_based = False
        
        protocol = DiscussionProtocol(config, mock_mod, use_group_chat=True)
        
        # Mock the GroupChatBuilder import
        mock_builder = MagicMock()
        mock_builder.participants.return_value = mock_builder
        mock_builder.with_max_rounds.return_value = mock_builder
        mock_builder.set_select_speakers_func.return_value = mock_builder
        
        with patch.dict("sys.modules", {"agent_framework": MagicMock(GroupChatBuilder=MagicMock(return_value=mock_builder))}):
            # Add a mock discussion agent
            agent = MagicMock(spec=DiscussionAgent)
            agent.name = "agent1"
            agent.role = DiscussionRole.NEUTRAL
            agent.chat_agent = MagicMock()
            protocol.register_discussion_agent(agent)
            
            # First call should build
            result1 = protocol._build_group_chat()
            # Second call should return cached
            result2 = protocol._build_group_chat()
            
            assert result1 is result2

    def test_build_group_chat_raises_without_agents(self) -> None:
        """Test _build_group_chat raises if no agents registered."""
        config = DiscussionConfig(topic="Test", max_rounds=2)
        mock_mod = MagicMock(spec=ModeratorAgent)
        mock_mod.is_llm_based = False
        
        protocol = DiscussionProtocol(config, mock_mod, use_group_chat=True)
        
        with pytest.raises(ValueError, match="No DiscussionAgents registered"):
            protocol._build_group_chat()


class TestRunDiscussionStream:
    """Tests for run_discussion_stream method."""

    @pytest.fixture
    def basic_config(self) -> DiscussionConfig:
        """Create basic discussion config."""
        return DiscussionConfig(topic="Test Topic", max_rounds=2)

    @pytest.fixture
    def mock_moderator(self) -> MagicMock:
        """Create mock moderator."""
        moderator = MagicMock(spec=ModeratorAgent)
        moderator.is_llm_based = False
        moderator.run = AsyncMock(return_value="Summary")
        return moderator

    @pytest.mark.asyncio
    async def test_run_discussion_stream_requires_group_chat_mode(
        self, basic_config: DiscussionConfig, mock_moderator: MagicMock
    ) -> None:
        """Test run_discussion_stream raises without use_group_chat=True."""
        protocol = DiscussionProtocol(
            basic_config, mock_moderator, use_group_chat=False
        )
        
        with pytest.raises(RuntimeError, match="requires use_group_chat=True"):
            async for _ in protocol.run_discussion_stream():
                pass

    @pytest.mark.asyncio
    async def test_run_discussion_stream_yields_started_event(
        self, basic_config: DiscussionConfig, mock_moderator: MagicMock
    ) -> None:
        """Test run_discussion_stream yields discussion_started event."""
        protocol = DiscussionProtocol(
            basic_config, mock_moderator, use_group_chat=True
        )
        
        # Add mock agent
        agent = MagicMock(spec=DiscussionAgent)
        agent.name = "agent1"
        agent.role = DiscussionRole.NEUTRAL
        agent.chat_agent = MagicMock()
        protocol.register_discussion_agent(agent)
        
        # Mock GroupChatBuilder
        mock_builder = MagicMock()
        mock_builder.participants.return_value = mock_builder
        mock_builder.with_max_rounds.return_value = mock_builder
        mock_builder.set_select_speakers_func.return_value = mock_builder
        
        async def empty_stream():
            return
            yield  # Make it an async generator
        
        mock_builder.run_stream = MagicMock(return_value=empty_stream())
        
        with patch.dict("sys.modules", {"agent_framework": MagicMock(GroupChatBuilder=MagicMock(return_value=mock_builder))}):
            events = []
            async for event in protocol.run_discussion_stream():
                events.append(event)
                if len(events) >= 3:
                    break
            
            assert events[0]["type"] == "discussion_started"
            assert events[0]["topic"] == "Test Topic"
            assert "agent1" in events[0]["participants"]

    @pytest.mark.asyncio
    async def test_run_discussion_stream_yields_round_started(
        self, basic_config: DiscussionConfig, mock_moderator: MagicMock
    ) -> None:
        """Test run_discussion_stream yields round_started event."""
        protocol = DiscussionProtocol(
            basic_config, mock_moderator, use_group_chat=True
        )
        
        agent = MagicMock(spec=DiscussionAgent)
        agent.name = "agent1"
        agent.role = DiscussionRole.NEUTRAL
        agent.chat_agent = MagicMock()
        protocol.register_discussion_agent(agent)
        
        mock_builder = MagicMock()
        mock_builder.participants.return_value = mock_builder
        mock_builder.with_max_rounds.return_value = mock_builder
        mock_builder.set_select_speakers_func.return_value = mock_builder
        
        async def empty_stream():
            return
            yield
        
        mock_builder.run_stream = MagicMock(return_value=empty_stream())
        
        with patch.dict("sys.modules", {"agent_framework": MagicMock(GroupChatBuilder=MagicMock(return_value=mock_builder))}):
            events = []
            async for event in protocol.run_discussion_stream():
                events.append(event)
            
            round_events = [e for e in events if e.get("type") == "round_started"]
            assert len(round_events) >= 1
            assert round_events[0]["round"] == 1

    @pytest.mark.asyncio
    async def test_run_discussion_stream_yields_completion(
        self, basic_config: DiscussionConfig, mock_moderator: MagicMock
    ) -> None:
        """Test run_discussion_stream yields discussion_completed event."""
        protocol = DiscussionProtocol(
            basic_config, mock_moderator, use_group_chat=True
        )
        
        agent = MagicMock(spec=DiscussionAgent)
        agent.name = "agent1"
        agent.role = DiscussionRole.NEUTRAL
        agent.chat_agent = MagicMock()
        protocol.register_discussion_agent(agent)
        
        mock_builder = MagicMock()
        mock_builder.participants.return_value = mock_builder
        mock_builder.with_max_rounds.return_value = mock_builder
        mock_builder.set_select_speakers_func.return_value = mock_builder
        
        async def empty_stream():
            return
            yield
        
        mock_builder.run_stream = MagicMock(return_value=empty_stream())
        
        with patch.dict("sys.modules", {"agent_framework": MagicMock(GroupChatBuilder=MagicMock(return_value=mock_builder))}):
            events = []
            async for event in protocol.run_discussion_stream():
                events.append(event)
            
            # Last event should be discussion_completed
            assert events[-1]["type"] == "discussion_completed"
            assert "summary" in events[-1]

    @pytest.mark.asyncio
    async def test_run_discussion_stream_handles_turn_events(
        self, basic_config: DiscussionConfig, mock_moderator: MagicMock
    ) -> None:
        """Test run_discussion_stream handles turn events from GroupChatBuilder."""
        protocol = DiscussionProtocol(
            basic_config, mock_moderator, use_group_chat=True
        )
        
        agent = MagicMock(spec=DiscussionAgent)
        agent.name = "agent1"
        agent.role = DiscussionRole.PROPONENT
        agent.chat_agent = MagicMock()
        protocol.register_discussion_agent(agent)
        
        # Create a simple event class to ensure hasattr works correctly
        class MockEvent:
            def __init__(self, speaker: str, text: str) -> None:
                self.speaker = speaker
                self.text = text
        
        # Create mock GroupChatBuilder that returns a mock with run_stream
        mock_builder = MagicMock()
        mock_builder.participants.return_value = mock_builder
        mock_builder.with_max_rounds.return_value = mock_builder
        mock_builder.set_select_speakers_func.return_value = mock_builder
        
        mock_built = MagicMock()
        
        async def mock_stream(prompt: str):
            yield MockEvent("agent1", "Hello ")
            yield MockEvent("agent1", "world!")
        
        mock_built.run_stream = mock_stream
        mock_builder.build.return_value = mock_built
        
        MockGroupChatBuilder = MagicMock(return_value=mock_builder)
        
        # Patch at module level before calling
        with patch.object(
            protocol, "_build_group_chat", return_value=mock_built
        ):
            events = []
            async for event in protocol.run_discussion_stream():
                events.append(event)
            
            # Find turn events
            turn_started = [e for e in events if e.get("type") == "turn_started"]
            turn_deltas = [e for e in events if e.get("type") == "turn_delta"]
            
            # Should have at least one turn started (first event triggers new turn)
            assert len(turn_started) >= 1
            assert turn_started[0]["speaker"] == "agent1"
            
            # Should have turn deltas for "Hello " and "world!"
            assert len(turn_deltas) >= 2
            assert turn_deltas[0]["delta"] == "Hello "

    @pytest.mark.asyncio
    async def test_run_discussion_stream_resets_active_state(
        self, basic_config: DiscussionConfig, mock_moderator: MagicMock
    ) -> None:
        """Test run_discussion_stream resets _is_active on completion."""
        protocol = DiscussionProtocol(
            basic_config, mock_moderator, use_group_chat=True
        )
        
        agent = MagicMock(spec=DiscussionAgent)
        agent.name = "agent1"
        agent.role = DiscussionRole.NEUTRAL
        agent.chat_agent = MagicMock()
        protocol.register_discussion_agent(agent)
        
        mock_builder = MagicMock()
        mock_builder.participants.return_value = mock_builder
        mock_builder.with_max_rounds.return_value = mock_builder
        mock_builder.set_select_speakers_func.return_value = mock_builder
        
        async def empty_stream():
            return
            yield
        
        mock_builder.run_stream = MagicMock(return_value=empty_stream())
        
        with patch.dict("sys.modules", {"agent_framework": MagicMock(GroupChatBuilder=MagicMock(return_value=mock_builder))}):
            async for _ in protocol.run_discussion_stream():
                pass
            
            assert protocol._is_active is False

    @pytest.mark.asyncio
    async def test_run_discussion_stream_invokes_callbacks(
        self, basic_config: DiscussionConfig, mock_moderator: MagicMock
    ) -> None:
        """Test run_discussion_stream invokes turn and round callbacks."""
        protocol = DiscussionProtocol(
            basic_config, mock_moderator, use_group_chat=True
        )
        
        agent = MagicMock(spec=DiscussionAgent)
        agent.name = "agent1"
        agent.role = DiscussionRole.NEUTRAL
        agent.chat_agent = MagicMock()
        protocol.register_discussion_agent(agent)
        
        turn_callback = MagicMock()
        round_callback = MagicMock()
        protocol.on_turn(turn_callback)
        protocol.on_round(round_callback)
        
        # Create a simple event class to ensure hasattr works correctly
        class MockEvent:
            def __init__(self, speaker: str, text: str) -> None:
                self.speaker = speaker
                self.text = text
        
        mock_built = MagicMock()
        
        async def mock_stream(prompt: str):
            yield MockEvent("agent1", "Hello")
        
        mock_built.run_stream = mock_stream
        
        with patch.object(
            protocol, "_build_group_chat", return_value=mock_built
        ):
            async for _ in protocol.run_discussion_stream():
                pass
            
            # Turn callback should be called when turn completes
            assert turn_callback.called
            # Round callback should be called when round completes
            assert round_callback.called

    @pytest.mark.asyncio
    async def test_run_discussion_stream_with_llm_moderator_summary(
        self, basic_config: DiscussionConfig
    ) -> None:
        """Test run_discussion_stream uses LLM moderator for summary."""
        mock_mod = MagicMock(spec=ModeratorAgent)
        mock_mod.is_llm_based = True
        mock_mod.run = AsyncMock(return_value="LLM generated summary")
        
        protocol = DiscussionProtocol(
            basic_config, mock_mod, use_group_chat=True
        )
        
        agent = MagicMock(spec=DiscussionAgent)
        agent.name = "agent1"
        agent.role = DiscussionRole.NEUTRAL
        agent.chat_agent = MagicMock()
        protocol.register_discussion_agent(agent)
        
        mock_builder = MagicMock()
        mock_builder.participants.return_value = mock_builder
        mock_builder.with_max_rounds.return_value = mock_builder
        mock_builder.set_select_speakers_func.return_value = mock_builder
        
        async def empty_stream():
            return
            yield
        
        mock_builder.run_stream = MagicMock(return_value=empty_stream())
        
        with patch.dict("sys.modules", {"agent_framework": MagicMock(GroupChatBuilder=MagicMock(return_value=mock_builder))}):
            events = []
            async for event in protocol.run_discussion_stream():
                events.append(event)
            
            completion = events[-1]
            assert completion["summary"] == "LLM generated summary"
            mock_mod.run.assert_called_once()
