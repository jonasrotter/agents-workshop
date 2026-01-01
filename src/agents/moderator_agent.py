"""
Moderator agent for facilitating multi-agent discussions.

Implements turn-taking, conflict resolution, and conclusion synthesis
for structured debates between multiple agents.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol
from enum import Enum
import asyncio

from opentelemetry import trace

tracer = trace.get_tracer(__name__)


class DiscussionPhase(str, Enum):
    """Phases of a moderated discussion."""
    
    OPENING = "opening"
    DISCUSSION = "discussion"
    REBUTTAL = "rebuttal"
    SYNTHESIS = "synthesis"
    CLOSING = "closing"


class ConflictStrategy(str, Enum):
    """Strategies for handling conflicts between agents."""
    
    ACKNOWLEDGE = "acknowledge"  # Simply note disagreement
    EXPLORE = "explore"  # Ask for clarification
    VOTE = "vote"  # Let all agents weigh in
    DEFER = "defer"  # Move on, revisit later


@dataclass
class DiscussionTurn:
    """A single turn in the discussion."""
    
    participant: str
    content: str
    phase: DiscussionPhase
    round_number: int
    references: list[str] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class DiscussionConflict:
    """Records a conflict between participants."""
    
    participants: list[str]
    topic: str
    positions: dict[str, str]
    resolution: Optional[str] = None
    strategy_used: Optional[ConflictStrategy] = None


@dataclass
class DiscussionSummary:
    """Summary of a completed discussion."""
    
    topic: str
    participants: list[str]
    total_rounds: int
    total_turns: int
    key_points: list[str]
    conflicts: list[DiscussionConflict]
    conclusion: str
    consensus_reached: bool


class AgentProtocol(Protocol):
    """Protocol for agents participating in discussions."""
    
    @property
    def name(self) -> str:
        """Agent name."""
        ...
    
    async def run(self, prompt: str) -> str:
        """Run the agent with a prompt."""
        ...


class ModeratorAgent:
    """Agent that moderates discussions between other agents."""
    
    def __init__(
        self,
        name: str = "moderator",
        conflict_strategy: ConflictStrategy = ConflictStrategy.EXPLORE,
        max_rounds: int = 5,
        synthesis_prompt: Optional[str] = None,
    ) -> None:
        """Initialize the moderator agent."""
        self.name = name
        self.conflict_strategy = conflict_strategy
        self.max_rounds = max_rounds
        self.synthesis_prompt = synthesis_prompt or self._default_synthesis_prompt()
        
        self._participants: dict[str, AgentProtocol] = {}
        self._turns: list[DiscussionTurn] = []
        self._conflicts: list[DiscussionConflict] = []
        self._current_round = 0
        self._current_phase = DiscussionPhase.OPENING
        self._topic = ""
    
    def _default_synthesis_prompt(self) -> str:
        """Default prompt for synthesizing conclusions."""
        return """Based on the following discussion, synthesize a conclusion:

Topic: {topic}

Discussion Summary:
{discussion_summary}

Key Points from Each Participant:
{participant_summaries}

Conflicts and Resolutions:
{conflicts}

Please provide:
1. Main conclusion
2. Areas of consensus
3. Remaining disagreements
4. Recommended next steps"""
    
    def register_participant(
        self,
        agent: AgentProtocol,
        role: Optional[str] = None,
    ) -> None:
        """Register an agent as a discussion participant."""
        self._participants[agent.name] = agent
    
    def unregister_participant(self, name: str) -> None:
        """Remove a participant from the discussion."""
        self._participants.pop(name, None)
    
    @property
    def participants(self) -> list[str]:
        """List of participant names."""
        return list(self._participants.keys())
    
    @property
    def turn_history(self) -> list[DiscussionTurn]:
        """Get all discussion turns."""
        return self._turns.copy()
    
    @tracer.start_as_current_span("moderator.start_discussion")
    async def start_discussion(
        self,
        topic: str,
        opening_prompt: Optional[str] = None,
    ) -> None:
        """Start a new discussion on a topic."""
        span = trace.get_current_span()
        span.set_attribute("discussion.topic", topic)
        span.set_attribute("discussion.participants", len(self._participants))
        
        self._topic = topic
        self._turns = []
        self._conflicts = []
        self._current_round = 1
        self._current_phase = DiscussionPhase.OPENING
        
        # Generate opening statements
        prompt = opening_prompt or self._opening_prompt(topic)
        await self._collect_responses(prompt)
        
        self._current_phase = DiscussionPhase.DISCUSSION
    
    @tracer.start_as_current_span("moderator.run_round")
    async def run_round(
        self,
        prompt: Optional[str] = None,
    ) -> list[DiscussionTurn]:
        """Run a single round of discussion."""
        if self._current_round > self.max_rounds:
            raise ValueError(f"Maximum rounds ({self.max_rounds}) exceeded")
        
        span = trace.get_current_span()
        span.set_attribute("discussion.round", self._current_round)
        
        round_prompt = prompt or self._round_prompt()
        turns = await self._collect_responses(round_prompt)
        
        # Check for conflicts
        conflicts = self._detect_conflicts(turns)
        self._conflicts.extend(conflicts)
        
        # Handle conflicts according to strategy
        for conflict in conflicts:
            await self._handle_conflict(conflict)
        
        self._current_round += 1
        return turns
    
    @tracer.start_as_current_span("moderator.synthesize")
    async def synthesize(
        self,
        synthesizer: Optional[AgentProtocol] = None,
    ) -> DiscussionSummary:
        """Synthesize a conclusion from the discussion."""
        self._current_phase = DiscussionPhase.SYNTHESIS
        
        # Build synthesis context
        discussion_summary = self._build_discussion_summary()
        participant_summaries = self._build_participant_summaries()
        conflict_summary = self._build_conflict_summary()
        
        synthesis_prompt = self.synthesis_prompt.format(
            topic=self._topic,
            discussion_summary=discussion_summary,
            participant_summaries=participant_summaries,
            conflicts=conflict_summary,
        )
        
        # Use provided synthesizer or first participant
        if synthesizer:
            conclusion = await synthesizer.run(synthesis_prompt)
        elif self._participants:
            first = next(iter(self._participants.values()))
            conclusion = await first.run(synthesis_prompt)
        else:
            conclusion = "No participants available for synthesis."
        
        # Extract key points
        key_points = self._extract_key_points()
        
        # Determine consensus
        consensus = len(self._conflicts) == 0 or all(
            c.resolution is not None for c in self._conflicts
        )
        
        self._current_phase = DiscussionPhase.CLOSING
        
        return DiscussionSummary(
            topic=self._topic,
            participants=self.participants,
            total_rounds=self._current_round - 1,
            total_turns=len(self._turns),
            key_points=key_points,
            conflicts=self._conflicts,
            conclusion=conclusion,
            consensus_reached=consensus,
        )
    
    async def _collect_responses(
        self,
        prompt: str,
    ) -> list[DiscussionTurn]:
        """Collect responses from all participants."""
        turns = []
        
        for name, agent in self._participants.items():
            # Include context of previous turns
            context_prompt = self._add_context(prompt, name)
            
            try:
                response = await agent.run(context_prompt)
                
                turn = DiscussionTurn(
                    participant=name,
                    content=response,
                    phase=self._current_phase,
                    round_number=self._current_round,
                    references=self._find_references(response),
                )
                turns.append(turn)
                self._turns.append(turn)
            except Exception as e:
                # Log error but continue with other participants
                span = trace.get_current_span()
                span.add_event(
                    "participant_error",
                    {"participant": name, "error": str(e)},
                )
        
        return turns
    
    def _opening_prompt(self, topic: str) -> str:
        """Generate opening prompt."""
        return f"""You are participating in a moderated discussion.

Topic: {topic}

Please provide your opening statement on this topic. Be clear about your position and key arguments."""
    
    def _round_prompt(self) -> str:
        """Generate round discussion prompt."""
        return f"""Continue the discussion on: {self._topic}

Consider the points made by other participants and respond with your thoughts.
You may agree, disagree, or build upon their arguments.

Please be constructive and cite specific points when responding to others."""
    
    def _add_context(self, prompt: str, current_participant: str) -> str:
        """Add discussion context to prompt."""
        if not self._turns:
            return prompt
        
        # Get recent turns (last round or last 5)
        recent_turns = [
            t for t in self._turns
            if t.round_number >= self._current_round - 1
            and t.participant != current_participant
        ][-5:]
        
        if not recent_turns:
            return prompt
        
        context = "\n\nPrevious statements:\n"
        for turn in recent_turns:
            context += f"\n{turn.participant}: {turn.content[:500]}..."
        
        return prompt + context
    
    def _find_references(self, content: str) -> list[str]:
        """Find references to other participants in content."""
        references = []
        for name in self._participants:
            if name.lower() in content.lower():
                references.append(name)
        return references
    
    def _detect_conflicts(
        self,
        turns: list[DiscussionTurn],
    ) -> list[DiscussionConflict]:
        """Detect conflicts between participants in this round."""
        conflicts: list[DiscussionConflict] = []
        
        # Simple heuristic: look for disagreement keywords
        disagreement_markers = [
            "disagree",
            "incorrect",
            "wrong",
            "contrary",
            "however",
            "but",
            "on the other hand",
        ]
        
        for turn in turns:
            content_lower = turn.content.lower()
            has_disagreement = any(
                marker in content_lower
                for marker in disagreement_markers
            )
            
            if has_disagreement and turn.references:
                for ref in turn.references:
                    conflict = DiscussionConflict(
                        participants=[turn.participant, ref],
                        topic=self._topic,
                        positions={
                            turn.participant: turn.content[:200],
                        },
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    async def _handle_conflict(
        self,
        conflict: DiscussionConflict,
    ) -> None:
        """Handle a detected conflict according to strategy."""
        conflict.strategy_used = self.conflict_strategy
        
        if self.conflict_strategy == ConflictStrategy.ACKNOWLEDGE:
            conflict.resolution = (
                f"Disagreement noted between {', '.join(conflict.participants)}"
            )
        
        elif self.conflict_strategy == ConflictStrategy.EXPLORE:
            # Ask for clarification (would need additional round)
            conflict.resolution = "Further exploration requested"
        
        elif self.conflict_strategy == ConflictStrategy.VOTE:
            conflict.resolution = "Put to vote among all participants"
        
        elif self.conflict_strategy == ConflictStrategy.DEFER:
            conflict.resolution = "Deferred for later discussion"
    
    def _build_discussion_summary(self) -> str:
        """Build a summary of the discussion so far."""
        summary_parts = []
        for round_num in range(1, self._current_round):
            round_turns = [
                t for t in self._turns
                if t.round_number == round_num
            ]
            if round_turns:
                summary_parts.append(f"Round {round_num}:")
                for turn in round_turns:
                    summary_parts.append(
                        f"  {turn.participant}: {turn.content[:200]}..."
                    )
        return "\n".join(summary_parts)
    
    def _build_participant_summaries(self) -> str:
        """Build summaries of each participant's contributions."""
        summaries = []
        for name in self._participants:
            turns = [t for t in self._turns if t.participant == name]
            if turns:
                summaries.append(f"{name}:")
                for turn in turns[:3]:  # First 3 turns
                    summaries.append(f"  - {turn.content[:150]}...")
        return "\n".join(summaries)
    
    def _build_conflict_summary(self) -> str:
        """Build summary of conflicts."""
        if not self._conflicts:
            return "No significant conflicts detected."
        
        summary = []
        for i, conflict in enumerate(self._conflicts, 1):
            summary.append(
                f"{i}. {', '.join(conflict.participants)}: "
                f"{conflict.resolution or 'Unresolved'}"
            )
        return "\n".join(summary)
    
    def _extract_key_points(self) -> list[str]:
        """Extract key points from the discussion."""
        key_points = []
        
        # Use first turn from each participant in opening
        opening_turns = [
            t for t in self._turns
            if t.phase == DiscussionPhase.OPENING
        ]
        
        for turn in opening_turns:
            # Take first sentence as key point
            first_sentence = turn.content.split(".")[0]
            if first_sentence:
                key_points.append(f"{turn.participant}: {first_sentence}")
        
        return key_points[:5]  # Limit to 5 key points
