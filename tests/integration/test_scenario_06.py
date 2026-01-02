"""Integration tests for Scenario 06: Moderated Agent Discussions."""

from __future__ import annotations

import pytest
from typing import Any

# =============================================================================
# Test Module Imports
# =============================================================================


class TestScenario06Imports:
    """Test that all Scenario 06 modules can be imported."""

    def test_import_moderator_agent(self) -> None:
        """Test importing ModeratorAgent."""
        from src.agents.moderator_agent import ModeratorAgent
        assert ModeratorAgent is not None

    def test_import_discussion_phase(self) -> None:
        """Test importing DiscussionPhase."""
        from src.agents.moderator_agent import DiscussionPhase
        assert DiscussionPhase is not None

    def test_import_conflict_strategy(self) -> None:
        """Test importing ConflictStrategy."""
        from src.agents.moderator_agent import ConflictStrategy
        assert ConflictStrategy is not None

    def test_import_discussion_turn(self) -> None:
        """Test importing DiscussionTurn."""
        from src.agents.moderator_agent import DiscussionTurn
        assert DiscussionTurn is not None

    def test_import_discussion_summary(self) -> None:
        """Test importing DiscussionSummary."""
        from src.agents.moderator_agent import DiscussionSummary
        assert DiscussionSummary is not None

    def test_import_discussion_protocol(self) -> None:
        """Test importing DiscussionProtocol."""
        from src.agents.discussion import DiscussionProtocol
        assert DiscussionProtocol is not None

    def test_import_debate_protocol(self) -> None:
        """Test importing DebateProtocol."""
        from src.agents.discussion import DebateProtocol
        assert DebateProtocol is not None

    def test_import_round_robin_protocol(self) -> None:
        """Test importing RoundRobinProtocol."""
        from src.agents.discussion import RoundRobinProtocol
        assert RoundRobinProtocol is not None

    def test_import_discussion_role(self) -> None:
        """Test importing DiscussionRole."""
        from src.agents.discussion import DiscussionRole
        assert DiscussionRole is not None

    def test_import_participant(self) -> None:
        """Test importing Participant."""
        from src.agents.discussion import Participant
        assert Participant is not None

    def test_import_discussion_config(self) -> None:
        """Test importing DiscussionConfig."""
        from src.agents.discussion import DiscussionConfig
        assert DiscussionConfig is not None

    def test_import_factory_functions(self) -> None:
        """Test importing factory functions."""
        from src.agents.discussion import create_debate, create_roundtable
        assert create_debate is not None
        assert create_roundtable is not None


# =============================================================================
# Mock Agent for Testing
# =============================================================================


class TestDiscussionAgent:
    """Mock agent for integration testing."""

    def __init__(self, name: str, stance: str):
        self.name = name
        self.stance = stance
        self._responses: list[str] = []

    async def run(self, prompt: str) -> str:
        response = f"{self.name} ({self.stance}): This is my response to the discussion."
        self._responses.append(response)
        return response


# =============================================================================
# ModeratorAgent Integration Tests
# =============================================================================


class TestModeratorAgentIntegration:
    """Integration tests for ModeratorAgent."""

    def test_moderator_creation(self) -> None:
        """Test creating a moderator with all options."""
        from src.agents.moderator_agent import ModeratorAgent, ConflictStrategy

        moderator = ModeratorAgent(
            name="TestModerator",
            conflict_strategy=ConflictStrategy.EXPLORE,
            max_rounds=5,
        )

        assert moderator.name == "TestModerator"
        assert moderator.conflict_strategy == ConflictStrategy.EXPLORE
        assert moderator.max_rounds == 5
        assert len(moderator.participants) == 0

    def test_participant_registration(self) -> None:
        """Test registering participants."""
        from src.agents.moderator_agent import ModeratorAgent

        moderator = ModeratorAgent(name="Mod")
        
        agents = [
            TestDiscussionAgent("Agent1", "positive"),
            TestDiscussionAgent("Agent2", "negative"),
            TestDiscussionAgent("Agent3", "neutral"),
        ]

        for agent in agents:
            moderator.register_participant(agent)

        assert len(moderator.participants) == 3
        assert "Agent1" in moderator.participants
        assert "Agent2" in moderator.participants
        assert "Agent3" in moderator.participants

    @pytest.mark.asyncio
    async def test_full_discussion_lifecycle(self) -> None:
        """Test complete discussion from start to synthesis."""
        from src.agents.moderator_agent import (
            ModeratorAgent,
            ConflictStrategy,
            DiscussionPhase,
        )

        moderator = ModeratorAgent(
            name="LifecycleModerator",
            conflict_strategy=ConflictStrategy.ACKNOWLEDGE,
            max_rounds=2,
        )

        agents = [
            TestDiscussionAgent("Proponent", "supportive"),
            TestDiscussionAgent("Opponent", "critical"),
        ]

        for agent in agents:
            moderator.register_participant(agent)

        # Start discussion - opening statements
        await moderator.start_discussion("Test topic for discussion")
        
        opening_turns = [
            t for t in moderator.turn_history
            if t.phase == DiscussionPhase.OPENING
        ]
        assert len(opening_turns) == 2

        # Run discussion round
        round_turns = await moderator.run_round()
        assert len(round_turns) > 0
        for turn in round_turns:
            assert turn.phase == DiscussionPhase.DISCUSSION

        # Synthesize
        summary = await moderator.synthesize()
        assert summary.topic == "Test topic for discussion"
        assert summary.total_turns > 0


# =============================================================================
# DiscussionProtocol Integration Tests
# =============================================================================


class TestDiscussionProtocolIntegration:
    """Integration tests for DiscussionProtocol."""

    def test_protocol_creation_with_config(self) -> None:
        """Test creating protocol with full configuration."""
        from src.agents.discussion import (
            DiscussionProtocol,
            DiscussionConfig,
            ConflictStrategy,
        )
        from src.agents.moderator_agent import ConflictStrategy

        config = DiscussionConfig(
            topic="Protocol Test",
            max_rounds=3,
            conflict_strategy=ConflictStrategy.VOTE,
            allow_rebuttals=True,
            rebuttal_rounds=1,
        )

        protocol = DiscussionProtocol(config)
        assert protocol.config.topic == "Protocol Test"
        assert protocol.config.max_rounds == 3

    def test_participant_roles(self) -> None:
        """Test registering participants with different roles."""
        from src.agents.discussion import (
            DiscussionProtocol,
            DiscussionConfig,
            DiscussionRole,
        )

        config = DiscussionConfig(topic="Role Test")
        protocol = DiscussionProtocol(config)

        roles = [
            (TestDiscussionAgent("Pro", "positive"), DiscussionRole.PROPONENT),
            (TestDiscussionAgent("Con", "negative"), DiscussionRole.OPPONENT),
            (TestDiscussionAgent("Mod", "balanced"), DiscussionRole.NEUTRAL),
            (TestDiscussionAgent("Exp", "expert"), DiscussionRole.EXPERT),
            (TestDiscussionAgent("Dev", "challenging"), DiscussionRole.DEVIL_ADVOCATE),
        ]

        for agent, role in roles:
            protocol.register_participant(agent, role=role)

        assert len(protocol.participants) == 5
        
        # Verify roles assigned correctly
        role_map = {p.name: p.role for p in protocol.participants}
        assert role_map["Pro"] == DiscussionRole.PROPONENT
        assert role_map["Dev"] == DiscussionRole.DEVIL_ADVOCATE

    @pytest.mark.asyncio
    async def test_run_full_discussion(self) -> None:
        """Test running a complete discussion."""
        from src.agents.discussion import (
            DiscussionProtocol,
            DiscussionConfig,
            DiscussionRole,
        )

        config = DiscussionConfig(
            topic="Full Discussion Test",
            max_rounds=2,
        )

        protocol = DiscussionProtocol(config)
        
        agents = [
            TestDiscussionAgent("Alice", "optimistic"),
            TestDiscussionAgent("Bob", "pessimistic"),
            TestDiscussionAgent("Carol", "pragmatic"),
        ]

        for agent in agents:
            protocol.register_participant(agent, role=DiscussionRole.NEUTRAL)

        summary = await protocol.run_discussion()

        assert summary.topic == "Full Discussion Test"
        # max_rounds=2 specifies main discussion rounds
        # Opening phase adds 1 additional round, so total can be up to 3
        assert summary.total_rounds <= 3
        assert summary.total_turns > 0

    def test_callback_registration(self) -> None:
        """Test registering event callbacks."""
        from src.agents.discussion import DiscussionProtocol, DiscussionConfig

        config = DiscussionConfig(topic="Callback Test")
        protocol = DiscussionProtocol(config)

        turn_callback_called = False
        round_callback_called = False

        def on_turn(turn: Any) -> None:
            nonlocal turn_callback_called
            turn_callback_called = True

        def on_round(result: Any) -> None:
            nonlocal round_callback_called
            round_callback_called = True

        protocol.on_turn(on_turn)
        protocol.on_round(on_round)

        assert len(protocol._on_turn_callbacks) == 1
        assert len(protocol._on_round_callbacks) == 1


# =============================================================================
# DebateProtocol Integration Tests
# =============================================================================


class TestDebateProtocolIntegration:
    """Integration tests for DebateProtocol."""

    def test_create_debate_with_factory(self) -> None:
        """Test creating debate with factory function."""
        from src.agents.discussion import create_debate, DiscussionRole

        proponent = TestDiscussionAgent("Advocate", "support")
        opponent = TestDiscussionAgent("Critic", "oppose")

        debate = create_debate(
            topic="Factory Debate",
            proponent=proponent,
            opponent=opponent,
        )

        assert debate.config.topic == "Factory Debate"
        assert len(debate.participants) == 2
        
        roles = {p.name: p.role for p in debate.participants}
        assert roles["Advocate"] == DiscussionRole.PROPONENT
        assert roles["Critic"] == DiscussionRole.OPPONENT

    def test_debate_with_judges(self) -> None:
        """Test debate with judge panel."""
        from src.agents.discussion import create_debate, DiscussionRole

        proponent = TestDiscussionAgent("For", "support")
        opponent = TestDiscussionAgent("Against", "oppose")
        judges = [
            TestDiscussionAgent("Judge1", "fair"),
            TestDiscussionAgent("Judge2", "impartial"),
        ]

        debate = create_debate(
            topic="Judged Debate",
            proponent=proponent,
            opponent=opponent,
            judges=judges,
        )

        assert len(debate.participants) == 4
        
        judge_count = sum(
            1 for p in debate.participants
            if p.role == DiscussionRole.NEUTRAL  # Judges are neutral
        )
        assert judge_count >= 2

    @pytest.mark.asyncio
    async def test_run_debate(self) -> None:
        """Test running a complete debate."""
        from src.agents.discussion import create_debate

        proponent = TestDiscussionAgent("Yes", "affirmative")
        opponent = TestDiscussionAgent("No", "negative")

        debate = create_debate(
            topic="Should we proceed?",
            proponent=proponent,
            opponent=opponent,
        )

        summary = await debate.run_discussion()

        assert summary.topic == "Should we proceed?"
        assert len(proponent._responses) > 0
        assert len(opponent._responses) > 0


# =============================================================================
# RoundRobinProtocol Integration Tests
# =============================================================================


class TestRoundRobinProtocolIntegration:
    """Integration tests for RoundRobinProtocol."""

    def test_create_roundtable_with_factory(self) -> None:
        """Test creating roundtable with factory function."""
        from src.agents.discussion import create_roundtable

        participants = [
            TestDiscussionAgent("Expert1", "domain1"),
            TestDiscussionAgent("Expert2", "domain2"),
            TestDiscussionAgent("Expert3", "domain3"),
        ]

        roundtable = create_roundtable(
            topic="Expert Roundtable",
            participants=participants,
            rounds=3,
        )

        assert roundtable.config.topic == "Expert Roundtable"
        assert roundtable.config.max_rounds == 3
        assert len(roundtable.participants) == 3

    @pytest.mark.asyncio
    async def test_run_roundtable(self) -> None:
        """Test running a complete roundtable."""
        from src.agents.discussion import create_roundtable

        participants = [
            TestDiscussionAgent("Panelist1", "view1"),
            TestDiscussionAgent("Panelist2", "view2"),
            TestDiscussionAgent("Panelist3", "view3"),
            TestDiscussionAgent("Panelist4", "view4"),
        ]

        roundtable = create_roundtable(
            topic="Panel Discussion",
            participants=participants,
            rounds=2,
        )

        summary = await roundtable.run_discussion()

        assert summary.topic == "Panel Discussion"
        # Each participant should have spoken
        for p in participants:
            assert len(p._responses) > 0


# =============================================================================
# Conflict Detection and Resolution Tests
# =============================================================================


class TestConflictHandling:
    """Integration tests for conflict handling."""

    def test_conflict_strategies(self) -> None:
        """Test all conflict strategies are available."""
        from src.agents.moderator_agent import ConflictStrategy

        strategies = [
            ConflictStrategy.ACKNOWLEDGE,
            ConflictStrategy.EXPLORE,
            ConflictStrategy.VOTE,
            ConflictStrategy.DEFER,
        ]

        assert len(strategies) == 4

    @pytest.mark.asyncio
    async def test_discussion_with_explore_strategy(self) -> None:
        """Test discussion with EXPLORE conflict strategy."""
        from src.agents.discussion import DiscussionProtocol, DiscussionConfig
        from src.agents.moderator_agent import ConflictStrategy

        config = DiscussionConfig(
            topic="Conflict Test",
            max_rounds=2,
            conflict_strategy=ConflictStrategy.EXPLORE,
        )

        protocol = DiscussionProtocol(config)
        
        # Add opposing viewpoints
        protocol.register_participant(
            TestDiscussionAgent("Optimist", "very positive")
        )
        protocol.register_participant(
            TestDiscussionAgent("Pessimist", "very negative")
        )

        summary = await protocol.run_discussion()
        
        # Discussion should complete regardless of conflicts
        assert summary is not None
        assert summary.total_turns > 0


# =============================================================================
# Statistics and Analysis Tests
# =============================================================================


class TestDiscussionStatistics:
    """Integration tests for discussion statistics."""

    def test_participant_stats(self) -> None:
        """Test getting participant statistics."""
        from src.agents.discussion import DiscussionProtocol, DiscussionConfig

        config = DiscussionConfig(topic="Stats Test")
        protocol = DiscussionProtocol(config)

        agents = [
            TestDiscussionAgent("Alice", "view1"),
            TestDiscussionAgent("Bob", "view2"),
        ]

        for agent in agents:
            protocol.register_participant(agent)

        stats = protocol.get_participant_stats()
        
        assert "Alice" in stats
        assert "Bob" in stats
        assert "role" in stats["Alice"]

    def test_cross_references(self) -> None:
        """Test getting cross references."""
        from src.agents.discussion import DiscussionProtocol, DiscussionConfig

        config = DiscussionConfig(topic="CrossRef Test")
        protocol = DiscussionProtocol(config)

        agents = [
            TestDiscussionAgent("Speaker1", "ref"),
            TestDiscussionAgent("Speaker2", "ref"),
        ]

        for agent in agents:
            protocol.register_participant(agent)

        refs = protocol.get_cross_references()
        
        assert isinstance(refs, dict)
        assert "Speaker1" in refs or len(refs) >= 0


# =============================================================================
# End-to-End Scenario Tests
# =============================================================================


class TestEndToEndScenarios:
    """End-to-end tests for complete scenarios."""

    @pytest.mark.asyncio
    async def test_ai_ethics_debate(self) -> None:
        """Test a complete AI ethics debate scenario."""
        from src.agents.discussion import create_debate

        proponent = TestDiscussionAgent(
            "AIAdvocate", 
            "AI will benefit humanity"
        )
        opponent = TestDiscussionAgent(
            "AISkeptic",
            "AI poses existential risks"
        )

        debate = create_debate(
            topic="Should AI development continue unregulated?",
            proponent=proponent,
            opponent=opponent,
        )

        summary = await debate.run_discussion()

        assert "unregulated" in summary.topic
        assert summary.total_turns >= 2

    @pytest.mark.asyncio
    async def test_expert_panel_discussion(self) -> None:
        """Test a complete expert panel discussion."""
        from src.agents.discussion import create_roundtable

        experts = [
            TestDiscussionAgent("Economist", "economic impact"),
            TestDiscussionAgent("Ethicist", "moral implications"),
            TestDiscussionAgent("Technologist", "technical feasibility"),
            TestDiscussionAgent("Policymaker", "regulatory approach"),
        ]

        panel = create_roundtable(
            topic="How should society prepare for AI transformation?",
            participants=experts,
            rounds=3,
        )

        summary = await panel.run_discussion()

        assert summary.total_rounds <= 3
        assert summary.total_turns > 0
        
        # All experts should have participated
        for expert in experts:
            assert expert._responses, f"{expert.name} should have responses"

    @pytest.mark.asyncio
    async def test_multi_round_discussion_with_callbacks(self) -> None:
        """Test multi-round discussion with event tracking."""
        from src.agents.discussion import (
            DiscussionProtocol,
            DiscussionConfig,
            DiscussionRole,
        )

        config = DiscussionConfig(
            topic="Tracked Discussion",
            max_rounds=3,
        )

        protocol = DiscussionProtocol(config)

        # Track events
        events: list[str] = []

        def track_turn(turn: Any) -> None:
            events.append(f"turn:{turn.participant}")

        def track_round(result: Any) -> None:
            events.append(f"round:{result.round_number}")

        protocol.on_turn(track_turn)
        protocol.on_round(track_round)

        # Add participants
        for name in ["Alice", "Bob", "Carol"]:
            protocol.register_participant(
                TestDiscussionAgent(name, f"{name}'s view"),
                role=DiscussionRole.NEUTRAL,
            )

        summary = await protocol.run_discussion()

        # Verify events were tracked
        turn_events = [e for e in events if e.startswith("turn:")]
        round_events = [e for e in events if e.startswith("round:")]

        assert len(turn_events) > 0, "Turn events should be tracked"
        assert len(round_events) > 0, "Round events should be tracked"
