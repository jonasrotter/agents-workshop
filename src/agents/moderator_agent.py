"""
Moderator agent for facilitating multi-agent discussions.

Implements turn-taking, conflict resolution, and conclusion synthesis
for structured debates between multiple agents.

Supports both rule-based moderation and ChatAgent-based LLM moderation
for intelligent speaker selection and conflict resolution.
"""

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional, Protocol, TYPE_CHECKING
from enum import Enum
import asyncio
import logging
import time

from opentelemetry import trace

# Conditional imports for agent-framework
if TYPE_CHECKING:
    from agent_framework import ChatAgent
    from agent_framework.azure import AzureOpenAIChatClient

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


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


# =============================================================================
# Default moderator instructions for ChatAgent-based moderation
# =============================================================================

DEFAULT_MODERATOR_INSTRUCTIONS = """You are a discussion moderator responsible for:
1. Selecting the next speaker based on the discussion context
2. Ensuring balanced participation from all participants
3. Identifying and addressing conflicts between participants
4. Guiding the discussion toward productive synthesis

When asked to select a speaker, analyze the conversation and choose who should speak next.
Return ONLY the participant name, nothing else.

When asked to synthesize, provide a balanced summary of all perspectives.

When handling conflicts, identify the core disagreement and suggest paths forward.
"""


class ModeratorAgent:
    """
    Agent that moderates discussions between other agents.
    
    Supports two moderation modes:
    1. Rule-based (default): Uses ConflictStrategy and round-robin speaker selection
    2. ChatAgent-based: Uses an LLM to intelligently select speakers and resolve conflicts
    
    The ChatAgent-based mode integrates with GroupChatBuilder for multi-agent
    orchestration via the `set_manager()` pattern.
    
    Example (rule-based):
        >>> moderator = ModeratorAgent(conflict_strategy=ConflictStrategy.EXPLORE)
        >>> moderator.register_participant(agent1)
        >>> moderator.register_participant(agent2)
        >>> await moderator.start_discussion("AI in Education")
    
    Example (ChatAgent-based):
        >>> moderator = create_moderator_agent(
        ...     name="smart_moderator",
        ...     instructions="Focus on finding common ground",
        ...     chat_client=chat_client,
        ... )
        >>> next_speaker = await moderator.select_speaker(discussion_state)
    """
    
    def __init__(
        self,
        name: str = "moderator",
        conflict_strategy: ConflictStrategy = ConflictStrategy.EXPLORE,
        max_rounds: int = 5,
        synthesis_prompt: Optional[str] = None,
        chat_agent: Optional["ChatAgent"] = None,
        moderator_instructions: Optional[str] = None,
    ) -> None:
        """
        Initialize the moderator agent.
        
        Args:
            name: Name for this moderator.
            conflict_strategy: Strategy for handling participant conflicts.
            max_rounds: Maximum discussion rounds.
            synthesis_prompt: Custom prompt for generating discussion synthesis.
            chat_agent: Optional ChatAgent for LLM-based moderation.
            moderator_instructions: Custom instructions for ChatAgent moderation.
        """
        self.name = name
        self.conflict_strategy = conflict_strategy
        self.max_rounds = max_rounds
        self.synthesis_prompt = synthesis_prompt or self._default_synthesis_prompt()
        
        # ChatAgent-based moderation (T105)
        self._chat_agent = chat_agent
        self._moderator_instructions = moderator_instructions or DEFAULT_MODERATOR_INSTRUCTIONS
        
        self._participants: dict[str, AgentProtocol] = {}
        self._turns: list[DiscussionTurn] = []
        self._conflicts: list[DiscussionConflict] = []
        self._current_round = 0
        self._current_phase = DiscussionPhase.OPENING
        self._topic = ""
    
    @property
    def chat_agent(self) -> Optional["ChatAgent"]:
        """Get the underlying ChatAgent if using LLM-based moderation."""
        return self._chat_agent
    
    @property
    def is_llm_based(self) -> bool:
        """Check if using LLM-based moderation."""
        return self._chat_agent is not None
    
    async def select_speaker(
        self,
        context: Optional[str] = None,
        exclude: Optional[list[str]] = None,
    ) -> Optional[str]:
        """
        Select the next speaker for the discussion.
        
        If using ChatAgent-based moderation, uses the LLM to intelligently
        select the next speaker. Otherwise, uses round-robin selection.
        
        Args:
            context: Discussion context for LLM-based selection.
            exclude: List of participant names to exclude from selection.
            
        Returns:
            Name of the selected speaker, or None if no valid speaker.
        """
        exclude = exclude or []
        available = [p for p in self._participants.keys() if p not in exclude]
        
        if not available:
            return None
        
        if self._chat_agent:
            # LLM-based speaker selection
            prompt = self._build_speaker_selection_prompt(context, available)
            try:
                result = await self._chat_agent.run(prompt)
                selected = self._parse_speaker_selection(result, available)
                return selected
            except Exception as e:
                logger.warning(f"LLM speaker selection failed: {e}, using round-robin")
        
        # Fall back to round-robin
        current_idx = self._current_round % len(available)
        return available[current_idx]
    
    def _build_speaker_selection_prompt(
        self,
        context: Optional[str],
        available: list[str],
    ) -> str:
        """Build prompt for LLM speaker selection."""
        recent_turns = self._turns[-5:] if self._turns else []
        turns_text = "\n".join(
            f"[{t.participant}]: {t.content[:200]}..." if len(t.content) > 200 else f"[{t.participant}]: {t.content}"
            for t in recent_turns
        )
        
        return f"""Select the next speaker for this discussion.

Topic: {self._topic}
Available speakers: {', '.join(available)}
Current round: {self._current_round}

Recent discussion:
{turns_text or 'No previous turns'}

{f'Context: {context}' if context else ''}

Who should speak next? Return ONLY the speaker name."""
    
    def _parse_speaker_selection(
        self,
        result: Any,
        available: list[str],
    ) -> Optional[str]:
        """Parse LLM response to extract speaker name."""
        # Extract text from result
        if hasattr(result, 'text'):
            text = result.text
        elif hasattr(result, 'content'):
            text = str(result.content)
        else:
            text = str(result)
        
        # Clean and match
        text = text.strip().lower()
        for speaker in available:
            if speaker.lower() in text:
                return speaker
        
        # Return first available if no match
        return available[0] if available else None
    
    async def run(self, prompt: str) -> str:
        """
        Run the moderator with a prompt.
        
        If using ChatAgent-based moderation, delegates to the ChatAgent.
        Otherwise, returns a placeholder response.
        
        This method allows the moderator to be used as a participant
        in GroupChatBuilder via set_manager().
        """
        if self._chat_agent:
            result = await self._chat_agent.run(prompt)
            if hasattr(result, 'text'):
                return result.text
            return str(result)
        
        # Rule-based moderator doesn't generate text responses
        return f"[Moderator {self.name}]: Processing discussion..."
    
    async def run_stream(self, prompt: str) -> AsyncIterator[str]:
        """
        Run the moderator with streaming output.
        
        Only available when using ChatAgent-based moderation.
        """
        if not self._chat_agent:
            yield f"[Moderator {self.name}]: Processing..."
            return
        
        async for chunk in self._chat_agent.run_stream(prompt):
            if hasattr(chunk, 'text'):
                yield chunk.text
            elif hasattr(chunk, 'delta'):
                yield str(chunk.delta)
            else:
                yield str(chunk)
    
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


# =============================================================================
# Factory Function for ChatAgent-based Moderator (T105)
# =============================================================================


def create_moderator_agent(
    name: str = "moderator",
    conflict_strategy: ConflictStrategy = ConflictStrategy.EXPLORE,
    max_rounds: int = 5,
    instructions: Optional[str] = None,
    synthesis_prompt: Optional[str] = None,
    chat_client: Optional["AzureOpenAIChatClient"] = None,
    model: str = "gpt-4o",
    temperature: float = 0.3,
) -> ModeratorAgent:
    """
    Factory function to create a ModeratorAgent with optional ChatAgent support.
    
    Creates a moderator that can use LLM-based speaker selection and
    conflict resolution when provided with a chat_client.
    
    Args:
        name: Name for this moderator.
        conflict_strategy: Strategy for handling participant conflicts.
        max_rounds: Maximum discussion rounds.
        instructions: Custom instructions for LLM-based moderation.
            Defaults to DEFAULT_MODERATOR_INSTRUCTIONS.
        synthesis_prompt: Custom prompt for generating discussion synthesis.
        chat_client: AzureOpenAIChatClient for LLM-based moderation.
            If not provided, uses rule-based moderation only.
        model: Model to use for ChatAgent (default: gpt-4o).
        temperature: Temperature for LLM responses (default: 0.3 for consistency).
        
    Returns:
        A ModeratorAgent configured for either rule-based or LLM-based moderation.
        
    Example (rule-based):
        >>> moderator = create_moderator_agent(
        ...     name="rule_mod",
        ...     conflict_strategy=ConflictStrategy.VOTE,
        ... )
    
    Example (LLM-based):
        >>> from agent_framework.azure import AzureOpenAIChatClient
        >>> chat_client = AzureOpenAIChatClient(...)
        >>> moderator = create_moderator_agent(
        ...     name="smart_mod",
        ...     instructions="Focus on synthesis and finding common ground",
        ...     chat_client=chat_client,
        ... )
    """
    chat_agent = None
    
    if chat_client is not None:
        # Create ChatAgent for LLM-based moderation
        try:
            from agent_framework import ChatAgent
        except ImportError as e:
            raise ImportError(
                "agent-framework package required for ChatAgent-based moderation. "
                "Install with: pip install agent-framework"
            ) from e
        
        full_instructions = instructions or DEFAULT_MODERATOR_INSTRUCTIONS
        
        chat_agent = chat_client.create_agent(
            name=name,
            instructions=full_instructions,
            temperature=temperature,
        )
    
    return ModeratorAgent(
        name=name,
        conflict_strategy=conflict_strategy,
        max_rounds=max_rounds,
        synthesis_prompt=synthesis_prompt,
        chat_agent=chat_agent,
        moderator_instructions=instructions,
    )
