"""Unit tests for agent discussion modules."""

from __future__ import annotations

import pytest
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from src.agents.moderator_agent import (
    ModeratorAgent,
    DiscussionPhase,
    DiscussionTurn,
    DiscussionConflict,
    DiscussionSummary,
    ConflictStrategy,
    AgentProtocol,
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
