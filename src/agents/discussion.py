"""
Discussion protocol for multi-agent debates.

Provides structured protocols for managing multi-agent
discussions with participant registration, round management,
and cross-reference detection.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol
from enum import Enum
import asyncio
import time

from opentelemetry import trace

from src.agents.moderator_agent import (
    ModeratorAgent,
    DiscussionPhase,
    DiscussionTurn,
    DiscussionSummary,
    ConflictStrategy,
    AgentProtocol,
)

tracer = trace.get_tracer(__name__)


class DiscussionRole(str, Enum):
    """Roles a participant can have in a discussion."""
    
    PROPONENT = "proponent"  # Argues for the topic
    OPPONENT = "opponent"  # Argues against
    NEUTRAL = "neutral"  # Balanced perspective
    EXPERT = "expert"  # Subject matter expert
    DEVIL_ADVOCATE = "devil_advocate"  # Challenges all positions


@dataclass
class Participant:
    """A participant in a discussion."""
    
    agent: AgentProtocol
    role: DiscussionRole = DiscussionRole.NEUTRAL
    priority: int = 0  # Higher speaks earlier
    max_turns_per_round: int = 1
    turns_taken: int = 0
    
    @property
    def name(self) -> str:
        """Get participant name from agent."""
        return self.agent.name


@dataclass
class DiscussionConfig:
    """Configuration for a discussion."""
    
    topic: str
    max_rounds: int = 5
    max_turns_per_round: int = 10
    turn_timeout: float = 30.0  # seconds
    conflict_strategy: ConflictStrategy = ConflictStrategy.EXPLORE
    require_references: bool = False
    allow_rebuttals: bool = True
    rebuttal_rounds: int = 1


@dataclass
class RoundResult:
    """Result of a single discussion round."""
    
    round_number: int
    turns: list[DiscussionTurn]
    conflicts_detected: int
    duration_seconds: float
    participation_rate: float


class DiscussionProtocol:
    """Protocol for managing structured discussions."""
    
    def __init__(
        self,
        config: DiscussionConfig,
        moderator: Optional[ModeratorAgent] = None,
    ) -> None:
        """Initialize the discussion protocol."""
        self.config = config
        # Calculate total rounds needed: main rounds + rebuttal rounds
        total_rounds = config.max_rounds + (
            config.rebuttal_rounds if config.allow_rebuttals else 0
        )
        self.moderator = moderator or ModeratorAgent(
            conflict_strategy=config.conflict_strategy,
            max_rounds=total_rounds,
        )
        
        self._participants: dict[str, Participant] = {}
        self._round_results: list[RoundResult] = []
        self._is_active = False
        self._on_turn_callbacks: list[Callable[[DiscussionTurn], None]] = []
        self._on_round_callbacks: list[Callable[[RoundResult], None]] = []
    
    def register_participant(
        self,
        agent: AgentProtocol,
        role: DiscussionRole = DiscussionRole.NEUTRAL,
        priority: int = 0,
    ) -> None:
        """Register an agent as a discussion participant."""
        participant = Participant(
            agent=agent,
            role=role,
            priority=priority,
        )
        self._participants[agent.name] = participant
        self.moderator.register_participant(agent, role.value)
    
    def unregister_participant(self, name: str) -> None:
        """Remove a participant from the discussion."""
        self._participants.pop(name, None)
        self.moderator.unregister_participant(name)
    
    @property
    def participants(self) -> list[Participant]:
        """Get all registered participants."""
        return list(self._participants.values())
    
    @property
    def participant_names(self) -> list[str]:
        """Get all participant names."""
        return list(self._participants.keys())
    
    def on_turn(
        self,
        callback: Callable[[DiscussionTurn], None],
    ) -> None:
        """Register callback for turn events."""
        self._on_turn_callbacks.append(callback)
    
    def on_round(
        self,
        callback: Callable[[RoundResult], None],
    ) -> None:
        """Register callback for round completion events."""
        self._on_round_callbacks.append(callback)
    
    @tracer.start_as_current_span("protocol.run_discussion")
    async def run_discussion(self) -> DiscussionSummary:
        """Run a complete discussion and return summary."""
        span = trace.get_current_span()
        span.set_attribute("discussion.topic", self.config.topic)
        span.set_attribute("discussion.participants", len(self._participants))
        
        if not self._participants:
            raise ValueError("No participants registered")
        
        self._is_active = True
        self._round_results = []
        
        try:
            # Opening phase
            await self.moderator.start_discussion(
                topic=self.config.topic,
            )
            
            # Main discussion rounds
            for round_num in range(1, self.config.max_rounds + 1):
                result = await self._run_round(round_num)
                self._round_results.append(result)
                
                # Notify callbacks
                for callback in self._on_round_callbacks:
                    callback(result)
                
                # Check for early completion
                if self._should_end_early(result):
                    break
            
            # Rebuttal rounds if configured
            if self.config.allow_rebuttals:
                for _ in range(self.config.rebuttal_rounds):
                    await self._run_rebuttal_round()
            
            # Synthesis
            summary = await self.moderator.synthesize()
            
        finally:
            self._is_active = False
        
        return summary
    
    async def _run_round(self, round_number: int) -> RoundResult:
        """Run a single discussion round."""
        start_time = time.time()
        
        # Sort participants by priority
        ordered = sorted(
            self._participants.values(),
            key=lambda p: p.priority,
            reverse=True,
        )
        
        # Collect turns
        turns = await self.moderator.run_round()
        
        # Notify turn callbacks
        for turn in turns:
            for callback in self._on_turn_callbacks:
                callback(turn)
        
        # Calculate metrics
        duration = time.time() - start_time
        participation = len(turns) / len(self._participants)
        conflicts = len([
            c for c in self.moderator._conflicts
            if any(
                t.round_number == round_number
                for t in self.moderator._turns
            )
        ])
        
        return RoundResult(
            round_number=round_number,
            turns=turns,
            conflicts_detected=conflicts,
            duration_seconds=duration,
            participation_rate=participation,
        )
    
    async def _run_rebuttal_round(self) -> list[DiscussionTurn]:
        """Run a rebuttal round."""
        rebuttal_prompt = f"""This is the rebuttal phase of the discussion on: {self.config.topic}

Please respond to any arguments or points you disagree with.
Be specific about which points you are addressing.
Focus on your strongest counter-arguments."""
        
        return await self.moderator.run_round(rebuttal_prompt)
    
    def _should_end_early(self, result: RoundResult) -> bool:
        """Check if discussion should end early."""
        # End if participation drops significantly
        if result.participation_rate < 0.5:
            return True
        
        # End if no new content (all turns are short)
        avg_length = sum(
            len(t.content) for t in result.turns
        ) / max(len(result.turns), 1)
        if avg_length < 50:
            return True
        
        return False
    
    def get_participant_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for each participant."""
        stats: dict[str, dict[str, Any]] = {}
        
        for name, participant in self._participants.items():
            turns = [
                t for t in self.moderator.turn_history
                if t.participant == name
            ]
            
            stats[name] = {
                "role": participant.role.value,
                "total_turns": len(turns),
                "total_words": sum(
                    len(t.content.split()) for t in turns
                ),
                "references_made": sum(
                    len(t.references) for t in turns
                ),
                "rounds_participated": len(set(t.round_number for t in turns)),
            }
        
        return stats
    
    def get_cross_references(self) -> dict[str, list[str]]:
        """Get cross-references between participants."""
        references: dict[str, list[str]] = {
            name: [] for name in self._participants
        }
        
        for turn in self.moderator.turn_history:
            for ref in turn.references:
                if ref in references:
                    references[turn.participant].append(ref)
        
        return references


class DebateProtocol(DiscussionProtocol):
    """Specialized protocol for formal debates."""
    
    def __init__(
        self,
        topic: str,
        max_rounds: int = 3,
        rebuttal_rounds: int = 2,
    ) -> None:
        """Initialize debate protocol."""
        config = DiscussionConfig(
            topic=topic,
            max_rounds=max_rounds,
            require_references=True,
            allow_rebuttals=True,
            rebuttal_rounds=rebuttal_rounds,
            conflict_strategy=ConflictStrategy.EXPLORE,
        )
        super().__init__(config)
        
        self._proponent: Optional[Participant] = None
        self._opponent: Optional[Participant] = None
    
    def set_proponent(self, agent: AgentProtocol) -> None:
        """Set the proponent (argues for the topic)."""
        self.register_participant(
            agent,
            role=DiscussionRole.PROPONENT,
            priority=10,
        )
        self._proponent = self._participants[agent.name]
    
    def set_opponent(self, agent: AgentProtocol) -> None:
        """Set the opponent (argues against the topic)."""
        self.register_participant(
            agent,
            role=DiscussionRole.OPPONENT,
            priority=10,
        )
        self._opponent = self._participants[agent.name]
    
    def add_judge(self, agent: AgentProtocol) -> None:
        """Add a judge to evaluate the debate."""
        self.register_participant(
            agent,
            role=DiscussionRole.NEUTRAL,
            priority=5,
        )


class RoundRobinProtocol(DiscussionProtocol):
    """Protocol where each participant speaks in order."""
    
    def __init__(
        self,
        topic: str,
        rounds: int = 3,
    ) -> None:
        """Initialize round-robin protocol."""
        config = DiscussionConfig(
            topic=topic,
            max_rounds=rounds,
            max_turns_per_round=100,  # Allow all participants
            allow_rebuttals=False,
        )
        super().__init__(config)
    
    async def _run_round(self, round_number: int) -> RoundResult:
        """Run round with strict ordering."""
        start_time = time.time()
        
        # Ensure each participant speaks in registration order
        ordered = list(self._participants.values())
        
        turns = []
        for participant in ordered:
            # Create individual prompt with context
            prompt = self._build_individual_prompt(participant, round_number)
            
            try:
                response = await participant.agent.run(prompt)
                turn = DiscussionTurn(
                    participant=participant.name,
                    content=response,
                    phase=DiscussionPhase.DISCUSSION,
                    round_number=round_number,
                )
                turns.append(turn)
                self.moderator._turns.append(turn)
                
                # Notify callbacks
                for callback in self._on_turn_callbacks:
                    callback(turn)
                    
            except Exception as e:
                span = trace.get_current_span()
                span.add_event(
                    "turn_error",
                    {"participant": participant.name, "error": str(e)},
                )
        
        duration = time.time() - start_time
        participation = len(turns) / max(len(ordered), 1)
        
        return RoundResult(
            round_number=round_number,
            turns=turns,
            conflicts_detected=0,
            duration_seconds=duration,
            participation_rate=participation,
        )
    
    def _build_individual_prompt(
        self,
        participant: Participant,
        round_number: int,
    ) -> str:
        """Build prompt for individual participant."""
        # Get previous turns from this round
        previous = [
            t for t in self.moderator._turns
            if t.round_number == round_number
        ]
        
        prompt = f"""Discussion Topic: {self.config.topic}
Round: {round_number}

"""
        if previous:
            prompt += "Previous statements this round:\n"
            for turn in previous:
                prompt += f"\n{turn.participant}: {turn.content[:300]}..."
            prompt += "\n\n"
        
        prompt += "Please share your perspective on this topic."
        
        return prompt


def create_debate(
    topic: str,
    proponent: AgentProtocol,
    opponent: AgentProtocol,
    judges: Optional[list[AgentProtocol]] = None,
) -> DebateProtocol:
    """Factory function to create a structured debate."""
    debate = DebateProtocol(topic)
    debate.set_proponent(proponent)
    debate.set_opponent(opponent)
    
    if judges:
        for judge in judges:
            debate.add_judge(judge)
    
    return debate


def create_roundtable(
    topic: str,
    participants: list[AgentProtocol],
    rounds: int = 3,
) -> RoundRobinProtocol:
    """Factory function to create a round-table discussion."""
    roundtable = RoundRobinProtocol(topic, rounds)
    
    for agent in participants:
        roundtable.register_participant(agent)
    
    return roundtable
